import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
sys.path.append(ROOT.as_posix())
from torch import Tensor
from typing import List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class Projection(nn.Module):

    def __init__(self, inp_dim: int=512, d_model: int=512):
        super(Projection, self).__init__()
        self.proj = nn.Linear(inp_dim, d_model)

    def forward(self, inp: Tensor) -> Tensor:
        return self.proj(inp)

class AVCM_Module(nn.Module):

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, feat, text_feat, key_padding_mask=None):
        (attn_out, _) = self.attn(feat, text_feat, text_feat, key_padding_mask=key_padding_mask)
        feat = self.norm1(feat + attn_out)
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(feat))))
        feat = self.norm2(feat + ffn_out)
        return feat

class MultiModalSpatialAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.proj_patch = nn.Linear(d_model, d_model)
        self.proj_audio = nn.Linear(d_model, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, patch_feat, audio_feat):
        (B, T, N, D) = patch_feat.shape
        p_proj = self.proj_patch(patch_feat)
        a_proj = self.proj_audio(audio_feat).unsqueeze(2)
        attn_map = torch.sum(p_proj * a_proj, dim=-1) / D ** 0.5
        attn_weights = F.softmax(attn_map, dim=-1).unsqueeze(-1)
        visual_focused = torch.sum(p_proj * attn_weights, dim=2)
        return visual_focused

class AdaptivePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        self.audio_proj = nn.Linear(1, d_model)
        self.motion_proj = nn.Linear(1, d_model)
        self.question_proj = nn.Linear(d_model, d_model)
        self.relation_proj = nn.Linear(6, d_model)
        self.phi_mlp = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.time_gate = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.full((1,), 0.02))

    @staticmethod
    def _normalize_per_clip(signal: Tensor, eps: float=1e-06) -> Tensor:
        mean = signal.mean(dim=1, keepdim=True)
        std = signal.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (signal - mean) / std

    @staticmethod
    def _positive_delta(signal: Tensor) -> Tensor:
        delta = signal[:, 1:] - signal[:, :-1]
        return F.pad(F.relu(delta), (1, 0))

    def _relation_features(self, audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        (batch, steps) = audio_activity.shape
        audio_z = self._normalize_per_clip(audio_activity)
        motion_z = self._normalize_per_clip(visual_motion)
        relative_time = torch.linspace(0.0, 1.0, steps, device=audio_activity.device, dtype=audio_activity.dtype).view(1, steps).expand(batch, -1)
        audio_onset = self._positive_delta(audio_z)
        motion_onset = self._positive_delta(motion_z)
        active = torch.sigmoid(audio_z) * torch.sigmoid(motion_z)
        cumulative_activity = torch.cumsum(active, dim=1) / max(steps, 1)
        co_activity = audio_z * motion_z
        quiet_gap = torch.zeros_like(active)
        if steps > 0:
            quiet_gap[:, 0] = 1.0 - active[:, 0]
        for step in range(1, steps):
            quiet_gap[:, step] = (1.0 - active[:, step]) * (quiet_gap[:, step - 1] + 1.0 / max(steps, 1))
        return torch.stack([relative_time, audio_onset, motion_onset, cumulative_activity, co_activity, quiet_gap], dim=-1)

    def forward(self, seq_feat: Tensor, question_feat: Tensor, audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        if seq_feat.ndim != 3:
            raise ValueError(f'seq_feat must have shape [B, T, D], got {tuple(seq_feat.shape)}')
        (batch, steps, _) = seq_feat.shape
        if audio_activity.shape != (batch, steps) or visual_motion.shape != (batch, steps):
            raise ValueError('audio_activity and visual_motion must both have shape [B, T]')
        audio_embed = self.audio_proj(self._normalize_per_clip(audio_activity).unsqueeze(-1))
        motion_embed = self.motion_proj(self._normalize_per_clip(visual_motion).unsqueeze(-1))
        question_embed = self.question_proj(question_feat).unsqueeze(1).expand(-1, steps, -1)
        relation_embed = self.relation_proj(self._relation_features(audio_activity, visual_motion))
        phi = self.phi_mlp(torch.cat([audio_embed, motion_embed, question_embed, relation_embed], dim=-1))
        alpha = F.softmax(self.time_gate(phi), dim=1) * steps
        return seq_feat + self.residual_scale * self.dropout(alpha * phi)

class QuestionGuidedTemporalAttention(nn.Module):

    def __init__(self, d_model, max_len: int=60):
        super().__init__()
        self.pos_encoder = AdaptivePositionalEncoding(d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_att = nn.Linear(d_model, 1)

    def forward(self, seq_feat, question_feat, audio_activity, visual_motion):
        seq_feat = self.pos_encoder(seq_feat, question_feat, audio_activity, visual_motion)
        q_emb = self.w_q(question_feat).unsqueeze(1)
        v_emb = self.w_v(seq_feat)
        energy = self.w_att(torch.tanh(q_emb + v_emb))
        weights = F.softmax(energy, dim=1)
        context = torch.sum(seq_feat * weights, dim=1)
        return context

class BiDirectionalCrossAttention(nn.Module):

    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attn_a2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.attn_v2a = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_v = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.ffn_v = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm_ffn_a = nn.LayerNorm(d_model)
        self.norm_ffn_v = nn.LayerNorm(d_model)

    def forward(self, feat_a, feat_v):
        (a_context, _) = self.attn_a2v(feat_a, feat_v, feat_v)
        (v_context, _) = self.attn_v2a(feat_v, feat_a, feat_a)
        feat_a = self.norm_a(feat_a + a_context)
        feat_v = self.norm_v(feat_v + v_context)
        feat_a = self.norm_ffn_a(feat_a + self.ffn_a(feat_a))
        feat_v = self.norm_ffn_v(feat_v + self.ffn_v(feat_v))
        return (feat_a, feat_v)

def temporal_alignment_loss(audio_feat, visual_feat):
    a_norm = F.normalize(audio_feat, dim=-1)
    v_norm = F.normalize(visual_feat, dim=-1)
    sim_aa = torch.matmul(a_norm, a_norm.transpose(1, 2))
    sim_vv = torch.matmul(v_norm, v_norm.transpose(1, 2))
    sim_av = torch.matmul(a_norm, v_norm.transpose(1, 2))
    loss = F.l1_loss(sim_av, sim_aa) + F.l1_loss(sim_av, sim_vv)
    return loss

def spatial_alignment_loss(patch_feat, sentence_feat, top_ratio=0.1, temp=0.07):
    patch_norm = F.normalize(patch_feat, dim=-1)
    sent_norm = F.normalize(sentence_feat, dim=-1).unsqueeze(1)
    logits = (patch_norm * sent_norm).sum(-1) / temp
    k = max(1, int(logits.size(1) * top_ratio))
    return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.topk(k, 1).values, 1)).mean()
