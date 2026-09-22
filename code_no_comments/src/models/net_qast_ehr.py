from __future__ import annotations
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.qast_ehr_components import AdaptivePositionController, HierarchicalEventMemory, MotionSoundQuestionPatchGrounder, QueryAdaptiveEvidenceRouter, temporal_delta

class TemporalQueryReader(nn.Module):

    def __init__(self, d_model: int, num_queries: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.condition = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU())
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, stream: Tensor, question: Tensor, prompt: Tensor) -> Tuple[Tensor, Tensor]:
        batch = stream.size(0)
        condition = self.condition(torch.cat((question, prompt), dim=-1))
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + condition.unsqueeze(1)
        (output, attention) = self.attention(queries, stream, stream, need_weights=True, average_attn_weights=False)
        output = self.norm(queries + output)
        output = self.norm(output + self.ffn(output))
        attention = attention.mean(dim=1)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        return (output, attention)

class QASTEHR(nn.Module):

    def __init__(self, audio_dim: int=128, video_dim: int=768, patch_dim: int=1024, question_dim: int=768, d_model: int=512, num_labels: int=42, num_heads: int=8, num_temporal_queries: int=10, event_scales: Sequence[int]=(3, 7, 15), keep_patches: int=4, lambda_tacl: float=0.035, lambda_sacl: float=0.035, lambda_event: float=0.03, lambda_evidence: float=0.06, lambda_diversity: float=0.005, lambda_position: float=0.02, aux_warmup_epochs: int=3, position_min_strength: float=0.05, position_max_strength: float=0.1, position_init_strength: float=0.07, evidence_margin: float=0.2, dropout: float=0.1, enable_cmg: bool=True, enable_scm: bool=True, enable_adaptive_position: bool=True, enable_patch_grounder: bool=True, **unused: object):
        super().__init__()
        del unused
        self.lambda_tacl = float(lambda_tacl)
        self.lambda_sacl = float(lambda_sacl)
        self.lambda_event = float(lambda_event)
        self.lambda_evidence = float(lambda_evidence)
        self.lambda_diversity = float(lambda_diversity)
        self.lambda_position = float(lambda_position)
        self.aux_warmup_epochs = max(int(aux_warmup_epochs), 1)
        self.evidence_margin = float(evidence_margin)
        self.current_epoch = 0
        self.auxiliary_ramp = 0.0
        self.audio_projection = nn.Linear(audio_dim, d_model)
        self.video_projection = nn.Linear(video_dim, d_model)
        self.patch_projection = nn.Linear(patch_dim, d_model)
        self.question_projection = nn.Linear(question_dim, d_model)
        self.prompt_projection = nn.Linear(question_dim, d_model)
        self.evidence_router = QueryAdaptiveEvidenceRouter(d_model, dropout)
        self.patch_grounder = MotionSoundQuestionPatchGrounder(d_model, keep_patches, dropout) if enable_patch_grounder else None
        self.evidence_norm = nn.LayerNorm(d_model)
        self.event_memory = HierarchicalEventMemory(d_model, event_scales, num_heads, dropout)
        self.adaptive_position = AdaptivePositionController(d_model, min_strength=position_min_strength, max_strength=position_max_strength, init_strength=position_init_strength, dropout=dropout) if enable_adaptive_position else None
        self.cmg = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True) if enable_cmg else None
        self.cmg_norm = nn.LayerNorm(d_model) if enable_cmg else None
        self.temporal_reader = TemporalQueryReader(d_model, num_temporal_queries, num_heads, dropout)
        self.semantic_context = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 3, dropout=dropout, activation='gelu', batch_first=True, norm_first=True), num_layers=1) if enable_scm else None
        self.fusion_gate = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.fusion_refine = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.answer_head = nn.Linear(d_model, num_labels)

    def set_training_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        self.auxiliary_ramp = min(1.0, max(0.0, self.current_epoch / self.aux_warmup_epochs))

    @staticmethod
    def _motion(visual: Tensor) -> Tensor:
        return temporal_delta(visual).norm(dim=-1)

    def _cross_modal_ground(self, question: Tensor, audio: Tensor, visual: Tensor, events: Tensor) -> Tensor:
        joint = torch.cat((audio, visual, events), dim=1)
        (context, _) = self.cmg(question.unsqueeze(1), joint, joint, need_weights=False)
        return self.cmg_norm(question + context.squeeze(1))

    def _fuse(self, question: Tensor, audio: Tensor, visual: Tensor, semantic: Tensor, patch: Tensor) -> Tensor:
        weights = F.softmax(self.fusion_gate(torch.cat((question, semantic), dim=-1)), dim=-1)
        branches = torch.stack((audio, visual, semantic, patch), dim=1)
        fused = (branches * weights.unsqueeze(-1)).sum(dim=1)
        return self.fusion_refine(fused + question)

    @staticmethod
    def _query_diversity(attention: Tensor) -> Tensor:
        normalized = attention / attention.norm(dim=-1, keepdim=True).clamp_min(1e-06)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        identity = torch.eye(similarity.size(1), device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
        return ((similarity - identity) ** 2).mean()

    @staticmethod
    def _event_consistency(stream: Tensor, memory_tokens: Tensor) -> Tensor:
        smoothness = temporal_delta(stream).square().mean()
        normalized = F.normalize(memory_tokens, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        identity = torch.eye(similarity.size(1), device=similarity.device, dtype=similarity.dtype).unsqueeze(0)
        separation = ((similarity - identity) ** 2).mean()
        return smoothness + separation

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        audio = self.audio_projection(data['audio'])
        visual = self.video_projection(data['video'])
        patches = self.patch_projection(data['patch'])
        question = self.question_projection(data['quest'].squeeze(1))
        prompt = self.prompt_projection(data['prompt'].squeeze(1))
        (router_stream, router_weights) = self.evidence_router(audio, visual, question)
        if self.patch_grounder is None:
            patch_stream = patches.mean(dim=2)
            patch_attention = patches.new_full(patches.shape[:3], 1.0 / patches.size(2))
        else:
            (patch_stream, patch_attention) = self.patch_grounder(patches, audio, visual, question)
        evidence_stream = self.evidence_norm(router_stream + patch_stream)
        (event_stream, memory_tokens, event_gates) = self.event_memory(evidence_stream, question)
        if self.adaptive_position is None:
            position_ratio = event_stream.sum() * 0.0
        else:
            (event_stream, position_diagnostics) = self.adaptive_position(event_stream, question, audio.norm(dim=-1), self._motion(visual))
            position_ratio = position_diagnostics['position_ratio']
        if self.cmg is None:
            grounded_question = question
        else:
            grounded_question = self._cross_modal_ground(question, audio, visual, event_stream)
        (temporal_tokens, temporal_attention) = self.temporal_reader(event_stream, grounded_question, prompt)
        importance = temporal_attention.mean(dim=1)
        importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        audio_pool = torch.einsum('bt,btd->bd', importance, audio)
        visual_pool = torch.einsum('bt,btd->bd', importance, visual)
        patch_pool = torch.einsum('bt,btd->bd', importance, patch_stream)
        semantic_tokens = torch.cat((grounded_question.unsqueeze(1), prompt.unsqueeze(1), memory_tokens, temporal_tokens), dim=1)
        if self.semantic_context is None:
            semantic = semantic_tokens.mean(dim=1)
        else:
            semantic = self.semantic_context(semantic_tokens)[:, 0]
        fused = self._fuse(grounded_question, audio_pool, visual_pool, semantic, patch_pool)
        logits = self.answer_head(fused)
        tacl = F.mse_loss(F.normalize(audio_pool, dim=-1), F.normalize(visual_pool, dim=-1))
        sacl = (1.0 - F.cosine_similarity(F.normalize(grounded_question, dim=-1), F.normalize(patch_pool, dim=-1), dim=-1)).mean()
        event_loss = self._event_consistency(event_stream, memory_tokens)
        diversity = self._query_diversity(temporal_attention)
        low_attention = (1.0 - importance).clamp_min(0.0)
        low_attention = low_attention / low_attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        low_event = torch.einsum('bt,btd->bd', low_attention.detach(), event_stream.detach())
        counterfactual_feature = self._fuse(grounded_question, audio_pool, visual_pool, low_event, patch_pool)
        counterfactual_logits = self.answer_head(counterfactual_feature)
        if 'label' in data:
            labels = data['label'].long().view(-1, 1)
            positive = logits.gather(1, labels).squeeze(1)
            counterfactual = counterfactual_logits.gather(1, labels).squeeze(1)
            evidence_loss = F.relu(self.evidence_margin - positive + counterfactual).mean()
        else:
            evidence_loss = (counterfactual_logits - logits).square().mean()
        if self.adaptive_position is None:
            position_loss = position_ratio
        else:
            position_loss = F.relu(0.05 - position_ratio).square() + F.relu(position_ratio - 0.1).square()
        return {'out': logits, 'router_weights': router_weights, 'patch_attention': patch_attention, 'event_gates': event_gates, 'temporal_attention': temporal_attention, 'position_ratio': position_ratio, 'loss_tacl': self.lambda_tacl * self.auxiliary_ramp * tacl, 'loss_sacl': self.lambda_sacl * self.auxiliary_ramp * sacl, 'loss_event': self.lambda_event * event_loss, 'loss_evidence': self.lambda_evidence * evidence_loss, 'loss_query_diversity': self.lambda_diversity * diversity, 'loss_position': self.lambda_position * position_loss}
