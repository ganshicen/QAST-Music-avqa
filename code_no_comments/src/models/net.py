import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
from src.models.encoders import CLIP_TEncoder
from src.models.modules import Projection, AVCM_Module, temporal_alignment_loss, spatial_alignment_loss, MultiModalSpatialAttention, QuestionGuidedTemporalAttention, BiDirectionalCrossAttention

class QA_AVAF(nn.Module):

    def __init__(self, d_model: int=512, video_dim: int=768, patch_dim: int=1024, audio_dim: int=128, text_dim: int=768, encoder_type: str='ViT-L/14@336px', alpha: float=0.2, beta: float=0.2, max_temporal_len: int=60, **kwargs):
        super(QA_AVAF, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.audio_proj = Projection(audio_dim, d_model)
        self.video_proj = Projection(video_dim, d_model)
        self.patch_proj = Projection(patch_dim, d_model)
        self.text_proj = Projection(text_dim, d_model)
        self.quest_encoder = CLIP_TEncoder(encoder_type)
        self.quest_encoder.freeze()
        self.av_interaction = BiDirectionalCrossAttention(d_model)
        self.spatial_attn = MultiModalSpatialAttention(d_model)
        self.avcm_visual = AVCM_Module(d_model)
        self.avcm_audio = AVCM_Module(d_model)
        self.weight_gen = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 2), nn.Sigmoid())
        self.fusion_fc = nn.Linear(d_model * 2, d_model)
        self.temporal_attn = QuestionGuidedTemporalAttention(d_model, max_len=max_temporal_len)
        self.head = nn.Linear(d_model, 42)
        for module in (self.audio_proj, self.video_proj, self.patch_proj, self.text_proj, self.av_interaction, self.spatial_attn, self.avcm_visual, self.avcm_audio, self.weight_gen, self.fusion_fc, self.temporal_attn, self.head):
            module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def process_text(self, reshaped_data):
        quest = reshaped_data['quest']
        if quest.dtype in (torch.float32, torch.float64):
            sentence_feat = quest.squeeze(1)
            word_feat = quest
        else:
            (sentence_feat, word_feat) = self.quest_encoder(quest)
        return (sentence_feat, word_feat)

    def forward(self, reshaped_data: Dict[str, Tensor]):
        return_dict = {}
        (sentence_feat, word_feat) = self.process_text(reshaped_data)
        f_s = self.text_proj(sentence_feat)
        f_w = self.text_proj(word_feat)
        audio = reshaped_data['audio']
        video = reshaped_data['video']
        f_a = self.audio_proj(audio)
        f_v_global = self.video_proj(video)
        patch = reshaped_data.get('patch', None)
        f_p = None
        if patch is not None:
            f_p = self.patch_proj(patch)
        loss_tacl = torch.tensor(0.0, device=f_a.device)
        loss_sacl = torch.tensor(0.0, device=f_a.device)
        if self.training:
            loss_tacl = temporal_alignment_loss(f_a, f_v_global)
            if f_p is not None:
                (B, T, N, D) = f_p.shape
                f_p_flat = f_p.view(B, T * N, D)
                loss_sacl = spatial_alignment_loss(f_p_flat, f_s)
        f_a_perm = f_a.permute(1, 0, 2)
        f_v_perm = f_v_global.permute(1, 0, 2)
        f_a_slt = f_a_perm
        f_v_slt = f_v_perm
        (f_a_slt, f_v_slt) = self.av_interaction(f_a_slt, f_v_slt)
        if f_p is not None:
            f_v_spatial = self.spatial_attn(f_p, f_a)
            f_v_spatial_perm = f_v_spatial.permute(1, 0, 2)
            f_v_slt = f_v_slt + f_v_spatial_perm
        f_w_perm = f_w.permute(1, 0, 2)
        f_a_mined = self.avcm_audio(f_a_slt, f_w_perm)
        f_v_mined = self.avcm_visual(f_v_slt, f_w_perm)
        f_a_final = f_a_mined.permute(1, 0, 2)
        f_v_final = f_v_mined.permute(1, 0, 2)
        weights = F.softmax(self.weight_gen(f_s), dim=-1)
        delta_v = weights[:, 0].view(-1, 1, 1)
        delta_a = weights[:, 1].view(-1, 1, 1)
        f_fused_seq = torch.cat([f_v_final * delta_v, f_a_final * delta_a], dim=-1)
        f_fused_seq = self.fusion_fc(f_fused_seq)
        audio_activity = f_a.norm(p=2, dim=-1)
        visual_motion = torch.cat((torch.zeros_like(f_v_global[:, :1, 0]), (f_v_global[:, 1:] - f_v_global[:, :-1]).norm(p=2, dim=-1)), dim=1)
        out_feat = self.temporal_attn(f_fused_seq, f_s, audio_activity, visual_motion)
        logits = self.head(out_feat)
        return_dict['out'] = logits
        if self.training:
            return_dict['loss_tacl'] = loss_tacl * self.alpha
            return_dict['loss_sacl'] = loss_sacl * self.beta
        return return_dict
