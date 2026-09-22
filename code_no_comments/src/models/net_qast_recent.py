from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.net_qast_grounded import QASTGroundedFusion

class RecentAdaptiveTemporalPrompt(nn.Module):

    def __init__(self, d_model: int, min_strength: float=0.015, max_strength: float=0.06, init_strength: float=0.03, runtime_scale: float=1.0, dropout: float=0.05):
        super().__init__()
        if not 0.0 < min_strength < init_strength < max_strength:
            raise ValueError('Require 0 < min_strength < init_strength < max_strength')
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        self.runtime_scale = float(runtime_scale)
        ratio = (init_strength - min_strength) / (max_strength - min_strength)
        self.strength_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        hidden = max(64, d_model // 4)
        self.relation_encoder = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.question_encoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.content_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.prompt_norm = nn.LayerNorm(d_model)
        self.time_gate = nn.Linear(d_model, 1)
        self.question_strength = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        nn.init.zeros_(self.question_strength.weight)
        nn.init.zeros_(self.question_strength.bias)
        self.last_strength = None
        self.last_ratio = None

    @staticmethod
    def _zscore(value: Tensor, eps: float=1e-06) -> Tensor:
        mean = value.mean(dim=1, keepdim=True)
        std = value.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (value - mean) / std

    @staticmethod
    def _positive_delta(value: Tensor) -> Tensor:
        return F.pad(F.relu(value[:, 1:] - value[:, :-1]), (1, 0))

    def _relations(self, audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        (batch, steps) = audio_activity.shape
        audio = self._zscore(audio_activity)
        motion = self._zscore(visual_motion)
        time = torch.linspace(0.0, 1.0, steps, device=audio.device, dtype=audio.dtype).view(1, steps).expand(batch, -1)
        audio_onset = self._positive_delta(audio)
        motion_onset = self._positive_delta(motion)
        co_activity = torch.sigmoid(audio) * torch.sigmoid(motion)
        cumulative = torch.cumsum(co_activity, dim=1) / max(steps, 1)
        local_variance = F.avg_pool1d((motion - motion.mean(dim=1, keepdim=True)).square().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        audio_change = F.pad((audio[:, 1:] - audio[:, :-1]).abs(), (1, 0))
        motion_change = F.pad((motion[:, 1:] - motion[:, :-1]).abs(), (1, 0))
        return torch.stack((time, audio, motion, audio_onset, motion_onset, cumulative, co_activity, audio_change, motion_change + local_variance), dim=-1)

    def forward(self, seq_feat: Tensor, question_feat: Tensor, audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        if not self.enabled:
            self.last_strength = torch.zeros((), device=seq_feat.device)
            self.last_ratio = torch.zeros((), device=seq_feat.device)
            return seq_feat
        relations = self._relations(audio_activity, visual_motion)
        prompt = self.relation_encoder(relations)
        prompt = prompt + self.question_encoder(question_feat).unsqueeze(1)
        prompt = prompt + self.content_adapter(seq_feat)
        prompt = self.prompt_norm(prompt)
        event_prior = relations[..., 3] + relations[..., 4] + relations[..., 7] + relations[..., 8]
        event_prior = self._zscore(event_prior)
        gate = torch.sigmoid(self.time_gate(prompt).squeeze(-1) + 0.35 * event_prior)
        gate = gate / gate.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-06)
        base_rms = seq_feat.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-06)
        prompt_rms = prompt.float().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-06)
        prompt = prompt * (base_rms / prompt_rms).to(prompt.dtype)
        strength_offset = 0.5 * torch.tanh(self.question_strength(question_feat)).view(-1, 1, 1)
        strength_unit = torch.sigmoid(self.strength_logit + strength_offset)
        strength = self.runtime_scale * (self.min_strength + (self.max_strength - self.min_strength) * strength_unit)
        delta = strength * gate.unsqueeze(-1) * self.dropout(prompt)
        output = seq_feat + delta
        with torch.no_grad():
            ratio = delta.float().square().mean(dim=(1, 2)).sqrt() / base_rms.flatten()
            self.last_strength = strength.detach().mean()
            self.last_ratio = ratio.detach().mean()
        return output

class DynamicResidualRouter(nn.Module):

    def __init__(self, d_model: int, bottleneck: int=128):
        super().__init__()
        self.router = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, bottleneck), nn.GELU(), nn.Linear(bottleneck, 3))
        self.temporal_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        self.semantic_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        self.spatial_adapter = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        self.adapter_scale = nn.Parameter(torch.full((3,), -2.944439))
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor((1.5, 0.0, 0.0)))

    def forward(self, base: Tensor, question: Tensor, temporal: Tensor, semantic: Tensor, spatial: Tensor) -> tuple[Tensor, Tensor]:
        weights = F.softmax(self.router(torch.cat((base, question), dim=-1)), dim=-1)
        scales = 0.1 * torch.sigmoid(self.adapter_scale)
        experts = torch.stack((base, base + scales[1] * self.temporal_adapter(temporal), base + scales[2] * self.semantic_adapter(semantic + spatial)), dim=1)
        routed = (weights.unsqueeze(-1) * experts).sum(dim=1)
        return (routed, weights)

class QASTRecentFusion(QASTGroundedFusion):

    def __init__(self, d_model: int=512, lambda_tacl: float=0.05, lambda_sacl: float=0.05, contrastive_temperature: float=0.1, position_min_strength: float=0.015, position_max_strength: float=0.06, position_init_strength: float=0.03, position_runtime_scale: float=1.0, freeze_existing: bool=True, **kwargs):
        super().__init__(d_model=d_model, lambda_tacl=lambda_tacl, lambda_sacl=lambda_sacl, **kwargs)
        if freeze_existing:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.lambda_tacl = float(lambda_tacl)
        self.lambda_sacl = float(lambda_sacl)
        self.contrastive_temperature = float(contrastive_temperature)
        self.recent_position = RecentAdaptiveTemporalPrompt(d_model, min_strength=position_min_strength, max_strength=position_max_strength, init_strength=position_init_strength, runtime_scale=position_runtime_scale)
        self.path_router = DynamicResidualRouter(d_model)

    def _symmetric_contrastive(self, left: Tensor, right: Tensor) -> Tensor:
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        logits = left @ right.transpose(0, 1) / self.contrastive_temperature
        target = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    def forward(self, data: Dict[str, Tensor]):
        audio = self.audio_proj(data['audio'])
        video = self.video_proj(data['video'])
        patches = self.patch_proj(data['patch'])
        question = self.question_proj(data['quest'].squeeze(1))
        prompt = self.prompt_proj(data['prompt'].squeeze(1))
        audio_activity = audio.norm(p=2, dim=-1)
        visual_motion = self._motion(video)
        audio = self.recent_position(audio, question, audio_activity, visual_motion)
        video = self.recent_position(video, question, audio_activity, visual_motion)
        (audio_context, video_context) = self.av_context(audio, video)
        joint_stream = torch.cat((audio, video), dim=1)
        (q_context, _) = self.cross_modal_grounding(question.unsqueeze(1), joint_stream, joint_stream, need_weights=False)
        grounded_question = self.cross_ground_norm(question + q_context.squeeze(1))
        question_for_reasoning = question + self.cross_ground_scale * grounded_question
        (audio_events, video_events, event_indices) = self.temporal_proposal(audio, video, prompt)
        visual_events = self.spatial_grounding(audio_events, patches, prompt, event_indices, question_for_reasoning)
        (audio_grounded, visual_grounded) = self.temporal_grounding(question_for_reasoning, audio_events, visual_events)
        branches = torch.cat((audio_grounded, audio_context.mean(dim=1), audio_events.mean(dim=1), visual_grounded, video_context.mean(dim=1), visual_events.mean(dim=1)), dim=-1)
        fused = self.fusion_linear(torch.tanh(branches))
        semantic_tokens = torch.cat((question_for_reasoning.unsqueeze(1), audio_events, visual_events), dim=1)
        semantic = self.semantic_context(semantic_tokens)[:, 0]
        semantic = self.semantic_project(semantic)
        fused = fused + self.semantic_scale * semantic
        temporal_summary = 0.5 * (audio_events.mean(dim=1) + video_events.mean(dim=1))
        spatial_summary = visual_events.mean(dim=1)
        (routed, router_weights) = self.path_router(fused, question_for_reasoning, temporal_summary, semantic, spatial_summary)
        answer_feature = torch.tanh(routed * question_for_reasoning)
        primary_logits = self.answer_head(answer_feature)
        if self.complementary_model is None:
            answer_logits = primary_logits
        else:
            self.complementary_model.eval()
            with torch.no_grad():
                complement_logits = self.complementary_model(data)['out']
            consensus = self.primary_mix * primary_logits.float().softmax(dim=-1) + (1.0 - self.primary_mix) * complement_logits.float().softmax(dim=-1)
            answer_logits = consensus.clamp_min(1e-08).log()
        output = {'out': answer_logits, 'position_strength': self.recent_position.last_strength, 'position_ratio': self.recent_position.last_ratio, 'router_weights': router_weights.detach().mean(dim=0)}
        if self.training:
            audio_summary = audio_events.mean(dim=1)
            video_summary = video_events.mean(dim=1)
            joint_summary = 0.5 * (video_summary + visual_events.mean(dim=1))
            local_temporal = 1.0 - F.cosine_similarity(F.normalize(audio_events, dim=-1), F.normalize(video_events, dim=-1), dim=-1).mean()
            tacl = self._symmetric_contrastive(audio_summary, video_summary) + 0.25 * local_temporal
            sacl = self._symmetric_contrastive(question_for_reasoning, joint_summary)
            output['loss_tacl'] = self.lambda_tacl * tacl
            output['loss_sacl'] = self.lambda_sacl * sacl
            target_usage = torch.full_like(router_weights.mean(dim=0), 1.0 / 3.0)
            output['loss_router_balance'] = 0.002 * F.mse_loss(router_weights.mean(dim=0), target_usage)
        return output

class PositionConditionedAnswerAdapter(nn.Module):

    def __init__(self, d_model: int, num_labels: int, bottleneck: int=192, min_scale: float=0.04, max_scale: float=0.24, init_scale: float=0.1, runtime_scale: float=1.0, dropout: float=0.1):
        super().__init__()
        if not 0.0 <= min_scale < init_scale < max_scale:
            raise ValueError('Require 0 <= min_scale < init_scale < max_scale')
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.runtime_scale = float(runtime_scale)
        ratio = (init_scale - min_scale) / (max_scale - min_scale)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.key = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.fusion = nn.Sequential(nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, bottleneck), nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck, d_model), nn.GELU())
        self.question_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.answer_head = nn.Linear(d_model, num_labels)
        nn.init.zeros_(self.answer_head.weight)
        nn.init.zeros_(self.answer_head.bias)
        self.last_scale = None
        self.last_feature_ratio = None

    def _attend(self, delta: Tensor, question: Tensor) -> Tensor:
        delta_rms = (delta.float().square().mean(dim=(1, 2), keepdim=True) + 1e-12).sqrt()
        normalized = delta / delta_rms.to(delta.dtype)
        query = F.normalize(self.query(question), dim=-1)
        keys = F.normalize(self.key(normalized), dim=-1)
        scores = torch.einsum('bd,btd->bt', query, keys) / math.sqrt(query.size(-1))
        weights = F.softmax(scores, dim=1)
        return torch.einsum('bt,btd->bd', weights, self.value(normalized))

    def forward(self, question: Tensor, audio_delta: Tensor, video_delta: Tensor, audio_events: Tensor, visual_events: Tensor) -> Tensor:
        audio_position = self._attend(audio_delta, question)
        video_position = self._attend(video_delta, question)
        event_relation = audio_events.mean(dim=1) - visual_events.mean(dim=1)
        descriptor = torch.cat((question, audio_position, video_position, event_relation), dim=-1)
        feature = self.fusion(descriptor) * self.question_gate(question)
        scale = self.runtime_scale * (self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(self.scale_logit))
        logits = scale * self.answer_head(feature)
        with torch.no_grad():
            self.last_scale = scale.detach()
            position_norm = 0.5 * (audio_position.float().norm(dim=-1).mean() + video_position.float().norm(dim=-1).mean())
            question_norm = question.float().norm(dim=-1).mean().clamp_min(1e-06)
            self.last_feature_ratio = (position_norm / question_norm).detach()
        return logits

class QASTRecentFusionV2(QASTRecentFusion):

    def __init__(self, d_model: int=512, num_labels: int=42, position_logit_min_scale: float=0.04, position_logit_max_scale: float=0.24, position_logit_init_scale: float=0.1, position_logit_runtime_scale: float=1.0, position_utility_weight: float=0.08, position_confidence_threshold: float=1.0, position_margin_threshold: float=1.0, position_enabled: bool=True, position_qtype_mask: tuple[int, ...] | list[int] | None=None, **kwargs):
        super().__init__(d_model=d_model, num_labels=num_labels, **kwargs)
        self.position_answer_adapter = PositionConditionedAnswerAdapter(d_model=d_model, num_labels=num_labels, min_scale=position_logit_min_scale, max_scale=position_logit_max_scale, init_scale=position_logit_init_scale, runtime_scale=position_logit_runtime_scale)
        self.position_utility_weight = float(position_utility_weight)
        self.position_confidence_threshold = float(position_confidence_threshold)
        self.position_margin_threshold = float(position_margin_threshold)
        self.recent_position.enabled = bool(position_enabled)
        self.position_qtype_mask = None if position_qtype_mask is None else tuple((int(v) for v in position_qtype_mask))
        question_dim = int(kwargs.get('question_dim', 768))
        self.position_qtype_router = nn.Sequential(nn.LayerNorm(question_dim), nn.Linear(question_dim, 9))

    def forward(self, data: Dict[str, Tensor]):
        audio_base = self.audio_proj(data['audio'])
        video_base = self.video_proj(data['video'])
        patches = self.patch_proj(data['patch'])
        question = self.question_proj(data['quest'].squeeze(1))
        prompt = self.prompt_proj(data['prompt'].squeeze(1))
        audio_activity = audio_base.norm(p=2, dim=-1)
        visual_motion = self._motion(video_base)
        if self.ablate_adaptive_position:
            audio = audio_base
            video = video_base
            zero = audio_base.new_zeros(())
            audio_strength = zero
            audio_ratio = zero
            video_strength = zero
            video_ratio = zero
        else:
            audio = self.recent_position(audio_base, question, audio_activity, visual_motion)
            audio_strength = self.recent_position.last_strength
            audio_ratio = self.recent_position.last_ratio
            video = self.recent_position(video_base, question, audio_activity, visual_motion)
            video_strength = self.recent_position.last_strength
            video_ratio = self.recent_position.last_ratio
        audio_delta = audio - audio_base
        video_delta = video - video_base
        sample_position_gate = None
        qtype_logits = self.position_qtype_router(data['quest'].squeeze(1))
        if self.position_qtype_mask is not None:
            qtype = qtype_logits.argmax(dim=-1)
            sample_position_gate = torch.zeros_like(qtype, dtype=audio.dtype)
            for qtype_id in self.position_qtype_mask:
                sample_position_gate = torch.maximum(sample_position_gate, (qtype == qtype_id).to(audio.dtype))
            audio_delta = audio_delta * sample_position_gate.view(-1, 1, 1)
            video_delta = video_delta * sample_position_gate.view(-1, 1, 1)
            audio = audio_base + audio_delta
            video = video_base + video_delta
        (audio_context, video_context) = self.av_context(audio, video)
        if self.ablate_cross_modal_grounding:
            question_for_reasoning = question
        else:
            joint_stream = torch.cat((audio, video), dim=1)
            (q_context, _) = self.cross_modal_grounding(question.unsqueeze(1), joint_stream, joint_stream, need_weights=False)
            grounded_question = self.cross_ground_norm(question + q_context.squeeze(1))
            question_for_reasoning = question + self.cross_ground_scale * grounded_question
        (audio_events, video_events, event_indices) = self.temporal_proposal(audio, video, prompt)
        if self.ablate_spatial_grounding:
            visual_events = video_events
        else:
            visual_events = self.spatial_grounding(audio_events, patches, prompt, event_indices, question_for_reasoning)
        (audio_grounded, visual_grounded) = self.temporal_grounding(question_for_reasoning, audio_events, visual_events)
        branches = torch.cat((audio_grounded, audio_context.mean(dim=1), audio_events.mean(dim=1), visual_grounded, video_context.mean(dim=1), visual_events.mean(dim=1)), dim=-1)
        fused = self.fusion_linear(torch.tanh(branches))
        if self.ablate_semantic_context:
            semantic = torch.zeros_like(fused)
        else:
            semantic_tokens = torch.cat((question_for_reasoning.unsqueeze(1), audio_events, visual_events), dim=1)
            semantic = self.semantic_project(self.semantic_context(semantic_tokens)[:, 0])
            fused = fused + self.semantic_scale * semantic
        temporal_summary = 0.5 * (audio_events.mean(dim=1) + video_events.mean(dim=1))
        spatial_summary = visual_events.mean(dim=1)
        (routed, router_weights) = self.path_router(fused, question_for_reasoning, temporal_summary, semantic, spatial_summary)
        answer_feature = torch.tanh(routed * question_for_reasoning)
        primary_logits = self.answer_head(answer_feature)
        if self.complementary_model is None:
            base_logits = primary_logits
        else:
            self.complementary_model.eval()
            with torch.no_grad():
                complement_logits = self.complementary_model(data)['out']
            consensus = self.primary_mix * primary_logits.float().softmax(dim=-1) + (1.0 - self.primary_mix) * complement_logits.float().softmax(dim=-1)
            base_logits = consensus.clamp_min(1e-08).log()
        if self.recent_position.enabled and (not self.ablate_adaptive_position):
            position_logits = self.position_answer_adapter(question_for_reasoning, audio_delta, video_delta, audio_events, visual_events)
            if sample_position_gate is not None:
                position_logits = position_logits * sample_position_gate.unsqueeze(-1)
            if not self.training:
                base_probabilities = base_logits.float().softmax(dim=-1)
                top_two = base_probabilities.topk(k=2, dim=-1).values
                confidence = top_two[:, 0]
                margin = top_two[:, 0] - top_two[:, 1]
                selective_gate = ((confidence <= self.position_confidence_threshold) & (margin <= self.position_margin_threshold)).to(position_logits.dtype)
                position_logits = position_logits * selective_gate.unsqueeze(-1)
        else:
            position_logits = torch.zeros_like(base_logits)
        answer_logits = base_logits + position_logits
        with torch.no_grad():
            audio_base_rms = audio_base.float().square().mean(dim=(1, 2)).sqrt().clamp_min(1e-06)
            video_base_rms = video_base.float().square().mean(dim=(1, 2)).sqrt().clamp_min(1e-06)
            audio_delta_rms = audio_delta.float().square().mean(dim=(1, 2)).sqrt()
            video_delta_rms = video_delta.float().square().mean(dim=(1, 2)).sqrt()
            effective_position_ratio = 0.5 * (audio_delta_rms / audio_base_rms + video_delta_rms / video_base_rms).mean()
        output = {'out': answer_logits, 'out_without_position_head': base_logits.detach(), 'position_logits': position_logits.detach(), 'position_strength': 0.5 * (audio_strength + video_strength), 'position_ratio': effective_position_ratio, 'position_logit_scale': self.position_answer_adapter.last_scale, 'router_weights': router_weights.detach().mean(dim=0), 'position_qtype_logits': qtype_logits}
        if self.training:
            audio_summary = audio_events.mean(dim=1)
            video_summary = video_events.mean(dim=1)
            joint_summary = 0.5 * (video_summary + visual_events.mean(dim=1))
            local_temporal = 1.0 - F.cosine_similarity(F.normalize(audio_events, dim=-1), F.normalize(video_events, dim=-1), dim=-1).mean()
            tacl = self._symmetric_contrastive(audio_summary, video_summary) + 0.25 * local_temporal
            sacl = self._symmetric_contrastive(question_for_reasoning, joint_summary)
            output['loss_tacl'] = self.lambda_tacl * tacl
            output['loss_sacl'] = self.lambda_sacl * sacl
            target_usage = torch.full_like(router_weights.mean(dim=0), 1.0 / 3.0)
            output['loss_router_balance'] = 0.002 * F.mse_loss(router_weights.mean(dim=0), target_usage)
            if not self.ablate_adaptive_position:
                position_alignment = 1.0 - F.cosine_similarity(F.normalize(audio_delta.mean(dim=1), dim=-1), F.normalize(video_delta.mean(dim=1), dim=-1), dim=-1).mean()
                output['loss_position_alignment'] = self.position_utility_weight * position_alignment
                if 'qtype_label' in data:
                    output['loss_position_qtype'] = 0.05 * F.cross_entropy(qtype_logits, data['qtype_label'].reshape(-1))
        return output

class QASTRecentFusionV3(QASTRecentFusionV2):

    def __init__(self, d_model: int=512, num_labels: int=42, topK: int=10, audio_dim: int=128, video_dim: int=768, patch_dim: int=1024, question_dim: int=768, routing_expert_checkpoint: str | None=None, recent_mix: float=0.7, train_routing_expert: bool=False, **kwargs):
        super().__init__(d_model=d_model, num_labels=num_labels, topK=topK, audio_dim=audio_dim, video_dim=video_dim, patch_dim=patch_dim, question_dim=question_dim, **kwargs)
        if not 0.0 <= recent_mix <= 1.0:
            raise ValueError('recent_mix must be in [0, 1]')
        self.recent_mix = float(recent_mix)
        self.train_routing_expert = bool(train_routing_expert)
        self.routing_expert = QASTGroundedFusion(topK=topK, audio_dim=audio_dim, video_dim=video_dim, patch_dim=patch_dim, question_dim=question_dim, d_model=d_model, num_labels=num_labels, source_checkpoint=None, complementary_checkpoint=None, use_complementary=bool(kwargs.get('use_complementary', True)), primary_mix=self.primary_mix, freeze_shared=bool(kwargs.get('freeze_shared', True)), ablate_semantic_context=self.ablate_semantic_context, ablate_spatial_grounding=self.ablate_spatial_grounding, ablate_cross_modal_grounding=self.ablate_cross_modal_grounding, ablate_adaptive_position=self.ablate_adaptive_position)
        if routing_expert_checkpoint:
            state = torch.load(routing_expert_checkpoint, map_location='cpu')
            if 'state_dict' in state:
                state = state['state_dict']
            state = {key.removeprefix('module.'): value for (key, value) in state.items()}
            message = self.routing_expert.load_state_dict(state, strict=True)
            if message.missing_keys or message.unexpected_keys:
                raise RuntimeError(f'routing expert checkpoint mismatch: {message}')
        self.routing_expert.requires_grad_(self.train_routing_expert)

    def _routing_expert_logits(self, data: Dict[str, Tensor]) -> Tensor:
        if self.train_routing_expert:
            return self.routing_expert(data)['out']
        self.routing_expert.eval()
        with torch.no_grad():
            return self.routing_expert(data)['out']

    def forward(self, data: Dict[str, Tensor]):
        output = super().forward(data)
        expert_logits = self._routing_expert_logits(data)
        consensus = self.recent_mix * output['out'].float().softmax(dim=-1) + (1.0 - self.recent_mix) * expert_logits.float().softmax(dim=-1)
        output['out'] = consensus.clamp_min(1e-08).log()
        output['routing_expert_weight'] = 1.0 - self.recent_mix
        return output
