from __future__ import annotations
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def temporal_delta(sequence: Tensor) -> Tensor:
    return F.pad(sequence[:, 1:] - sequence[:, :-1], (0, 0, 1, 0))

def normalize_signal(signal: Tensor, eps: float=1e-06) -> Tensor:
    mean = signal.mean(dim=1, keepdim=True)
    scale = signal.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (signal - mean) / scale

class QueryAdaptiveEvidenceRouter(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.question_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.refine = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, audio: Tensor, visual: Tensor, question: Tensor) -> Tuple[Tensor, Tensor]:
        audio_delta = temporal_delta(audio)
        visual_delta = temporal_delta(visual)
        co_activity = torch.sigmoid(F.cosine_similarity(audio, visual, dim=-1)).unsqueeze(-1)
        candidates = (visual, visual_delta, audio, audio_delta * co_activity)
        streams = [projection(norm(candidate)) for (norm, projection, candidate) in zip(self.norms, self.projections, candidates)]
        weights = F.softmax(self.question_gate(question), dim=-1)
        selected = (torch.stack(streams, dim=2) * weights[:, None, :, None]).sum(dim=2)
        return (selected + self.refine(selected), weights)

class MotionSoundQuestionPatchGrounder(nn.Module):

    def __init__(self, d_model: int, keep_patches: int=4, dropout: float=0.1):
        super().__init__()
        self.keep_patches = max(int(keep_patches), 1)
        self.patch_key = nn.Linear(d_model, d_model)
        self.audio_query = nn.Linear(d_model, d_model)
        self.question_query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.score_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, patches: Tensor, audio: Tensor, visual: Tensor, question: Tensor) -> Tuple[Tensor, Tensor]:
        del visual
        dimension = patches.size(-1)
        keys = self.patch_key(patches)
        patch_delta = temporal_delta(patches.flatten(2)).view_as(patches)
        motion_score = patch_delta.norm(dim=-1)
        audio_score = torch.einsum('btd,btpd->btp', self.audio_query(audio), keys) / dimension ** 0.5
        question_score = torch.einsum('bd,btpd->btp', self.question_query(question), keys) / dimension ** 0.5
        score_parts = torch.stack((normalize_signal(motion_score.flatten(0, 1)).view_as(motion_score), normalize_signal(audio_score.flatten(0, 1)).view_as(audio_score), normalize_signal(question_score.flatten(0, 1)).view_as(question_score)), dim=-1)
        score_weights = F.softmax(self.score_gate(question), dim=-1)
        scores = (score_parts * score_weights[:, None, None, :]).sum(dim=-1)
        soft = F.softmax(scores, dim=-1)
        count = min(self.keep_patches, scores.size(-1))
        indices = scores.topk(count, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
        mask = hard + soft - soft.detach()
        attention = mask * soft
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        grounded = torch.einsum('btp,btpd->btd', attention, self.value(patches))
        return (grounded + self.output(grounded), attention)

class HierarchicalEventMemory(nn.Module):

    def __init__(self, d_model: int, scales: Sequence[int]=(3, 7, 15), num_heads: int=8, dropout: float=0.1):
        super().__init__()
        del num_heads
        self.scales = tuple((int(scale) for scale in scales))
        if len(self.scales) != 3 or any((scale < 1 or scale % 2 == 0 for scale in self.scales)):
            raise ValueError('event memory requires three positive odd scales')
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(d_model, d_model, scale, padding=scale // 2, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, 1)) for scale in self.scales])
        self.question_queries = nn.ModuleList([nn.Linear(d_model, d_model) for _ in self.scales])
        self.scale_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(self.scales)))
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stream: Tensor, question: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        branches = [branch(stream.transpose(1, 2)).transpose(1, 2) for branch in self.branches]
        tokens = []
        for (branch, query_layer) in zip(branches, self.question_queries):
            query = query_layer(question)
            attention = F.softmax(torch.einsum('bd,btd->bt', query, branch) / query.size(-1) ** 0.5, dim=-1)
            tokens.append(torch.einsum('bt,btd->bd', attention, branch))
        memory_tokens = torch.stack(tokens, dim=1)
        gates = F.softmax(self.scale_gate(question), dim=-1)
        timeline = (torch.stack(branches, dim=2) * gates[:, None, :, None]).sum(dim=2)
        return (self.output_norm(stream + self.dropout(timeline)), memory_tokens, gates)

class AdaptivePositionController(nn.Module):

    def __init__(self, d_model: int, min_strength: float=0.05, max_strength: float=0.1, init_strength: float=0.07, dropout: float=0.1):
        super().__init__()
        if not 0.0 <= min_strength < init_strength < max_strength:
            raise ValueError('position strengths must satisfy min < init < max')
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        fraction = (init_strength - min_strength) / (max_strength - min_strength)
        self.strength_bias = nn.Parameter(torch.logit(torch.tensor([fraction])))
        self.strength_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        nn.init.zeros_(self.strength_gate[-1].weight)
        nn.init.zeros_(self.strength_gate[-1].bias)
        self.audio_projection = nn.Linear(1, d_model)
        self.motion_projection = nn.Linear(1, d_model)
        self.question_projection = nn.Linear(d_model, d_model)
        self.relation_projection = nn.Linear(9, d_model)
        self.phi = nn.Sequential(nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.time_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.time_gate.weight)
        nn.init.constant_(self.time_gate.bias, torch.logit(torch.tensor(0.8)).item())
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def relation_features(audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        audio = normalize_signal(audio_activity)
        motion = normalize_signal(visual_motion)
        (batch, steps) = audio.shape
        time = torch.linspace(0.0, 1.0, steps, dtype=audio.dtype, device=audio.device).view(1, steps).expand(batch, -1)
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
        audio = self.audio_projection(normalize_signal(audio_activity).unsqueeze(-1))
        motion = self.motion_projection(normalize_signal(visual_motion).unsqueeze(-1))
        question_tokens = self.question_projection(question).unsqueeze(1).expand(-1, steps, -1)
        relations = self.relation_projection(self.relation_features(audio_activity, visual_motion))
        phi = self.phi(torch.cat((audio, motion, question_tokens, relations), dim=-1))
        phi = phi / phi.square().mean(dim=(1, 2), keepdim=True).add(1e-06).sqrt()
        time_gate = 0.25 + 0.75 * torch.sigmoid(self.time_gate(phi))
        strength_fraction = torch.sigmoid(self.strength_bias + self.strength_gate(question))
        strength = self.min_strength + (self.max_strength - self.min_strength) * strength_fraction
        raw_injection = time_gate * self.dropout(phi)
        raw_norm = raw_injection.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-06)
        sequence_norm = sequence.norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-06)
        scale = strength * sequence_norm / raw_norm
        injected = raw_injection * scale.unsqueeze(-1)
        ratio = injected.norm(dim=-1).mean() / sequence.norm(dim=-1).mean().clamp_min(1e-06)
        return (sequence + injected, {'mean_strength': strength.mean(), 'position_ratio': ratio, 'time_gate_mean': time_gate.mean()})
