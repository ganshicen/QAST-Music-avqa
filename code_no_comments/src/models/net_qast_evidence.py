from __future__ import annotations
from typing import Dict, Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def _delta(sequence: Tensor) -> Tensor:
    return F.pad(sequence[:, 1:] - sequence[:, :-1], (0, 0, 1, 0))

def _normalize_signal(signal: Tensor, eps: float=1e-06) -> Tensor:
    mean = signal.mean(dim=1, keepdim=True)
    scale = signal.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (signal - mean) / scale

class QuestionAdaptiveEvidenceSelector(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        self.stream_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.stream_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.question_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, audio: Tensor, video: Tensor, question: Tensor) -> Tuple[Tensor, Tensor]:
        audio_delta = _delta(audio)
        visual_delta = _delta(video)
        co_activity = torch.sigmoid(F.cosine_similarity(audio, video, dim=-1)).unsqueeze(-1)
        candidates = (video, visual_delta, audio, audio_delta * co_activity)
        streams = [projection(norm(candidate)) for (projection, norm, candidate) in zip(self.stream_projections, self.stream_norms, candidates)]
        weights = F.softmax(self.question_gate(question), dim=-1)
        stacked = torch.stack(streams, dim=2)
        selected = (stacked * weights[:, None, :, None]).sum(dim=2)
        return (selected + self.output(selected), weights)

class MultiScaleEventEncoder(nn.Module):

    def __init__(self, d_model: int, scales: Iterable[int]=(1, 3, 5, 9), dropout: float=0.1):
        super().__init__()
        scales = tuple((int(scale) for scale in scales))
        if not scales or any((scale < 1 or scale % 2 == 0 for scale in scales)):
            raise ValueError('temporal scales must be non-empty positive odd integers')
        self.scales = scales
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=scale, padding=scale // 2, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, kernel_size=1)) for scale in scales])
        self.scale_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(scales)))
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stream: Tensor, question: Tensor) -> Tuple[Tensor, Tensor]:
        channel_first = stream.transpose(1, 2)
        branches = [branch(channel_first).transpose(1, 2) for branch in self.branches]
        weights = F.softmax(self.scale_gate(question), dim=-1)
        encoded = (torch.stack(branches, dim=2) * weights[:, None, :, None]).sum(dim=2)
        return (self.output_norm(stream + self.dropout(encoded)), weights)

class QuestionGuidedTemporalQueries(nn.Module):

    def __init__(self, d_model: int, num_queries: int=10, num_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.context = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU())
        self.attention = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, stream: Tensor, question: Tensor, prompt: Tensor) -> Tuple[Tensor, Tensor]:
        batch = stream.size(0)
        conditioned = self.context(torch.cat((question, prompt), dim=-1))
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + conditioned.unsqueeze(1)
        (events, attention) = self.attention(queries, stream, stream, need_weights=True, average_attn_weights=False)
        events = self.norm(queries + events)
        events = self.norm(events + self.ffn(events))
        attention = attention.mean(dim=1)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        return (events, attention)

class SingleAdaptiveTemporalPrompt(nn.Module):

    def __init__(self, d_model: int, min_strength: float=0.03, max_strength: float=0.12, init_strength: float=0.06, dropout: float=0.1):
        super().__init__()
        if not 0.0 <= min_strength < max_strength:
            raise ValueError('position strength interval is invalid')
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        ratio = (init_strength - min_strength) / (max_strength - min_strength)
        ratio = min(max(ratio, 0.0001), 1.0 - 0.0001)
        init_logit = torch.logit(torch.tensor(ratio))
        self.strength_bias = nn.Parameter(init_logit.reshape(1))
        self.strength_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.audio_projection = nn.Linear(1, d_model)
        self.motion_projection = nn.Linear(1, d_model)
        self.question_projection = nn.Linear(d_model, d_model)
        self.relation_projection = nn.Linear(9, d_model)
        self.phi = nn.Sequential(nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.time_gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _relations(audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        audio = _normalize_signal(audio_activity)
        motion = _normalize_signal(visual_motion)
        (batch, steps) = audio.shape
        time = torch.linspace(0.0, 1.0, steps, device=audio.device, dtype=audio.dtype).view(1, steps).expand(batch, -1)
        audio_onset = F.pad(F.relu(audio[:, 1:] - audio[:, :-1]), (1, 0))
        motion_onset = F.pad(F.relu(motion[:, 1:] - motion[:, :-1]), (1, 0))
        denominator = max(steps, 1)
        cumulative_audio = torch.cumsum(torch.sigmoid(audio), dim=1) / denominator
        cumulative_motion = torch.cumsum(torch.sigmoid(motion), dim=1) / denominator
        co_activity = torch.sigmoid(audio) * torch.sigmoid(motion)
        quiet_history = torch.cumsum(1.0 - co_activity, dim=1) / denominator
        return torch.stack((time, torch.sin(torch.pi * time), torch.cos(torch.pi * time), audio_onset, motion_onset, cumulative_audio, cumulative_motion, co_activity, quiet_history), dim=-1)

    def forward(self, sequence: Tensor, question: Tensor, audio_activity: Tensor, visual_motion: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        (batch, steps, _) = sequence.shape
        if audio_activity.shape != (batch, steps) or visual_motion.shape != (batch, steps):
            raise ValueError('activity inputs must have shape [batch, time]')
        audio = self.audio_projection(_normalize_signal(audio_activity).unsqueeze(-1))
        motion = self.motion_projection(_normalize_signal(visual_motion).unsqueeze(-1))
        question_tokens = self.question_projection(question).unsqueeze(1).expand(-1, steps, -1)
        relation = self.relation_projection(self._relations(audio_activity, visual_motion))
        phi = self.phi(torch.cat((audio, motion, question_tokens, relation), dim=-1))
        phi = phi / phi.square().mean(dim=(1, 2), keepdim=True).add(1e-06).sqrt()
        time_weight = torch.sigmoid(self.time_gate(phi))
        strength_ratio = torch.sigmoid(self.strength_bias + self.strength_gate(question))
        strength = self.min_strength + (self.max_strength - self.min_strength) * strength_ratio
        injected = strength[:, None, :] * time_weight * self.dropout(phi)
        output = sequence + injected
        position_ratio = injected.norm(dim=-1).mean() / sequence.norm(dim=-1).mean().clamp_min(1e-06)
        diagnostics = {'mean_strength': strength.mean(), 'position_ratio': position_ratio, 'time_gate_mean': time_weight.mean()}
        return (output, diagnostics)

class QASTSingleEvidence(nn.Module):

    def __init__(self, topK: int=10, audio_dim: int=128, video_dim: int=768, patch_dim: int=1024, question_dim: int=768, d_model: int=512, num_labels: int=42, num_heads: int=8, num_temporal_queries: int=10, temporal_scales: Iterable[int]=(1, 3, 5, 9), lambda_tacl: float=0.035, lambda_sacl: float=0.035, lambda_evidence: float=0.05, lambda_diversity: float=0.005, aux_warmup_epochs: int=5, position_min_strength: float=0.03, position_max_strength: float=0.12, position_init_strength: float=0.06, evidence_margin: float=0.2, dropout: float=0.1, **_: object):
        super().__init__()
        self.topK = int(topK)
        self.lambda_tacl = float(lambda_tacl)
        self.lambda_sacl = float(lambda_sacl)
        self.lambda_evidence = float(lambda_evidence)
        self.lambda_diversity = float(lambda_diversity)
        self.aux_warmup_epochs = max(int(aux_warmup_epochs), 1)
        self.evidence_margin = float(evidence_margin)
        self.current_epoch = 0
        self.auxiliary_ramp = 0.0
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)
        self.patch_projection = nn.Linear(patch_dim, d_model)
        self.question_projection = nn.Linear(question_dim, d_model)
        self.prompt_projection = nn.Linear(question_dim, d_model)
        self.adaptive_position = SingleAdaptiveTemporalPrompt(d_model, min_strength=position_min_strength, max_strength=position_max_strength, init_strength=position_init_strength, dropout=dropout)
        self.evidence_selector = QuestionAdaptiveEvidenceSelector(d_model, dropout)
        self.event_encoder = MultiScaleEventEncoder(d_model, temporal_scales, dropout)
        self.cross_modal_grounding = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.question_norm = nn.LayerNorm(d_model)
        self.temporal_queries = QuestionGuidedTemporalQueries(d_model, num_temporal_queries, num_heads, dropout)
        self.spatial_query = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU())
        self.spatial_key = nn.Linear(d_model, d_model)
        self.spatial_value = nn.Linear(d_model, d_model)
        semantic_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 3, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.semantic_context = nn.TransformerEncoder(semantic_layer, num_layers=1)
        self.fusion_projection = nn.Sequential(nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, d_model), nn.GELU(), nn.Dropout(dropout))
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.answer_head = nn.Linear(d_model, num_labels)

    def set_training_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.auxiliary_ramp = min(1.0, max(0.0, self.current_epoch / self.aux_warmup_epochs))

    @staticmethod
    def _motion(video: Tensor) -> Tensor:
        return F.pad((video[:, 1:] - video[:, :-1]).norm(dim=-1), (1, 0))

    def _spatial_grounding(self, patches: Tensor, audio: Tensor, question: Tensor, event_stream: Tensor, temporal_importance: Tensor) -> Tensor:
        query = self.spatial_query(torch.cat((audio, event_stream, question.unsqueeze(1).expand_as(audio)), dim=-1))
        keys = self.spatial_key(patches)
        values = self.spatial_value(patches)
        patch_attention = F.softmax(torch.einsum('btd,btpd->btp', query, keys) / query.size(-1) ** 0.5, dim=-1)
        grounded = torch.einsum('btp,btpd->btd', patch_attention, values)
        return torch.einsum('bt,btd->bd', temporal_importance, grounded)

    def _fuse(self, question: Tensor, grounded_question: Tensor, semantic: Tensor, spatial: Tensor) -> Tensor:
        proposal = self.fusion_projection(torch.cat((question, grounded_question, semantic, spatial), dim=-1))
        gate = self.fusion_gate(torch.cat((question, proposal), dim=-1))
        return gate * proposal + (1.0 - gate) * question

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        audio = self.audio_projection(data['audio'])
        video = self.video_projection(data['video'])
        patches = self.patch_projection(data['patch'])
        question = self.question_projection(data['quest'].squeeze(1))
        prompt = self.prompt_projection(data['prompt'].squeeze(1))
        audio_activity = audio.norm(dim=-1)
        visual_motion = self._motion(video)
        (audio, position_diagnostics_a) = self.adaptive_position(audio, question, audio_activity, visual_motion)
        (video, position_diagnostics_v) = self.adaptive_position(video, question, audio_activity, visual_motion)
        (evidence_stream, static_dynamic_weights) = self.evidence_selector(audio, video, question)
        (event_stream, scale_weights) = self.event_encoder(evidence_stream, question)
        joint = torch.cat((audio, video, event_stream), dim=1)
        (grounded_question, _) = self.cross_modal_grounding(question.unsqueeze(1), joint, joint, need_weights=False)
        grounded_question = self.question_norm(question + grounded_question.squeeze(1))
        (event_tokens, temporal_attention) = self.temporal_queries(event_stream, grounded_question, prompt)
        temporal_importance = temporal_attention.mean(dim=1)
        temporal_importance = temporal_importance / temporal_importance.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        spatial = self._spatial_grounding(patches, audio, grounded_question, event_stream, temporal_importance)
        audio_pool = torch.einsum('bt,btd->bd', temporal_importance, audio)
        video_pool = torch.einsum('bt,btd->bd', temporal_importance, video)
        semantic_tokens = torch.cat((grounded_question.unsqueeze(1), prompt.unsqueeze(1), audio_pool.unsqueeze(1), video_pool.unsqueeze(1), event_tokens, spatial.unsqueeze(1)), dim=1)
        semantic = self.semantic_context(semantic_tokens)[:, 0]
        fused = self._fuse(question, grounded_question, semantic, spatial)
        logits = self.answer_head(fused)
        tacl = F.mse_loss(F.normalize(audio_pool, dim=-1), F.normalize(video_pool, dim=-1))
        sacl = (1.0 - F.cosine_similarity(F.normalize(grounded_question, dim=-1), F.normalize(spatial, dim=-1), dim=-1)).mean()
        normalized_attention = temporal_attention / temporal_attention.norm(dim=-1, keepdim=True).clamp_min(1e-06)
        query_similarity = torch.bmm(normalized_attention, normalized_attention.transpose(1, 2))
        identity = torch.eye(query_similarity.size(1), device=query_similarity.device, dtype=query_similarity.dtype).unsqueeze(0)
        diversity = ((query_similarity - identity) ** 2).mean()
        low_attention = (1.0 - temporal_importance).clamp_min(0.0)
        low_attention = low_attention / low_attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        counterfactual_spatial = torch.einsum('bt,btd->bd', low_attention.detach(), event_stream.detach())
        counterfactual_feature = self._fuse(question, grounded_question, semantic, counterfactual_spatial)
        counterfactual_logits = self.answer_head(counterfactual_feature)
        if 'label' in data:
            labels = data['label'].long().view(-1, 1)
            positive = logits.gather(1, labels).squeeze(1)
            counterfactual = counterfactual_logits.gather(1, labels).squeeze(1)
            evidence_loss = F.relu(self.evidence_margin - positive + counterfactual).mean()
        else:
            evidence_loss = (counterfactual_logits - logits).square().mean()
        position_ratio = 0.5 * (position_diagnostics_a['position_ratio'] + position_diagnostics_v['position_ratio'])
        return {'out': logits, 'temporal_attention': temporal_attention, 'static_dynamic_weights': static_dynamic_weights, 'scale_weights': scale_weights, 'position_ratio': position_ratio, 'loss_tacl': self.lambda_tacl * self.auxiliary_ramp * tacl, 'loss_sacl': self.lambda_sacl * self.auxiliary_ramp * sacl, 'loss_evidence': self.lambda_evidence * evidence_loss, 'loss_query_diversity': self.lambda_diversity * diversity}
