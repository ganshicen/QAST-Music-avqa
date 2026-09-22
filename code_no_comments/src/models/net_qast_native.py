from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.modules import AdaptivePositionalEncoding
from src.models.tspm import AVHanLayer, AV_Attn, TemporalPerception, SpatioPerceptionModule, QstTemporalGrounding

class QASTNative(nn.Module):

    def __init__(self, topK: int=10, audio_dim: int=128, video_dim: int=768, patch_dim: int=1024, question_dim: int=768, d_model: int=512, num_labels: int=42, alpha: float=0.05, beta: float=0.02, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.question_proj = nn.Linear(question_dim, d_model)
        self.prompt_proj = nn.Linear(question_dim, d_model)
        self.adaptive_pe = AdaptivePositionalEncoding(d_model)
        self.av_context = AV_Attn(AVHanLayer(d_model=d_model, nhead=1, dim_feedforward=d_model), num_layers=1)
        self.temporal_selector = TemporalPerception(topK)
        self.spatial_perception = SpatioPerceptionModule(topK)
        self.temporal_grounding = QstTemporalGrounding()
        self.spatial_gate = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
        semantic_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.semantic_context = nn.TransformerEncoder(semantic_layer, num_layers=1)
        self.semantic_norm = nn.LayerNorm(d_model)
        self.semantic_scale = nn.Parameter(torch.zeros(1))
        self.fusion = nn.Sequential(nn.Linear(d_model * 6, d_model), nn.Tanh())
        self.answer_head = nn.Linear(d_model, num_labels)

    @staticmethod
    def _motion(video: Tensor) -> Tensor:
        return torch.cat((torch.zeros_like(video[:, :1, 0]), (video[:, 1:] - video[:, :-1]).norm(p=2, dim=-1)), dim=1)

    def forward(self, data: Dict[str, Tensor]):
        audio = self.audio_proj(data['audio'])
        video = self.video_proj(data['video'])
        patch = self.patch_proj(data['patch'])
        question = self.question_proj(data['quest'].squeeze(1))
        prompt = self.prompt_proj(data['prompt'].squeeze(1))
        audio_activity = audio.norm(p=2, dim=-1)
        visual_motion = self._motion(video)
        audio = self.adaptive_pe(audio, question, audio_activity, visual_motion)
        video = self.adaptive_pe(video, question, audio_activity, visual_motion)
        (audio_context, video_context) = self.av_context(audio, video)
        (audio_selected, video_selected, selected_indices) = self.temporal_selector(audio, video, prompt)
        visual_spatial = self.spatial_perception(audio_selected, patch, prompt, selected_indices)
        q_tokens = question.unsqueeze(1).expand(-1, visual_spatial.size(1), -1)
        spatial_delta = torch.tanh(self.spatial_gate(torch.cat((visual_spatial, audio_selected, q_tokens), dim=-1)))
        visual_spatial = visual_spatial * (1.0 + 0.1 * spatial_delta)
        (audio_grounded, visual_grounded) = self.temporal_grounding(question, audio_selected, visual_spatial)
        branches = torch.cat((audio_grounded, audio_context.mean(dim=1), audio_selected.mean(dim=1), visual_grounded, video_context.mean(dim=1), visual_spatial.mean(dim=1)), dim=-1)
        fused = self.fusion(branches)
        semantic_tokens = torch.cat((question.unsqueeze(1), audio_selected, visual_spatial), dim=1)
        semantic = self.semantic_norm(self.semantic_context(semantic_tokens)[:, 0])
        fused = torch.tanh(fused + self.semantic_scale * semantic)
        logits = self.answer_head(fused * question)
        output = {'out': logits}
        if self.training:
            tacl = F.mse_loss(F.normalize(audio, dim=-1), F.normalize(video, dim=-1))
            sacl = (1.0 - F.cosine_similarity(F.normalize(question, dim=-1), F.normalize(visual_spatial.mean(dim=1), dim=-1), dim=-1)).mean()
            output['loss_tacl'] = self.alpha * tacl
            output['loss_sacl'] = self.beta * sacl
        return output
