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

class AVCrossAttn(nn.Module):

    def __init__(self, d_model: int=512, nhead: int=8, dropout: float=0.1):
        super(AVCrossAttn, self).__init__()
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_mask = None
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def sub_forward(self, src_q: Tensor, src_v: Tensor, query: Optional[Tensor]=None) -> Tensor:
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        src_q = src_q + self.dropout(slf_attn) + self.dropout(crs_attn)
        src_q = self.norm1(src_q)
        src_q = src_q + self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

    def forward(self, src_q: Tensor, src_v: Tensor, query: Optional[Tensor]=None, visualize: bool=False) -> List[Tensor]:
        src1 = self.sub_forward(src_q, src_v)
        src2 = self.sub_forward(src_v, src_q)
        if visualize:
            return (src1, src2, None)
        return (src1, src2)

class AVQCrossAttn(nn.Module):

    def __init__(self, d_model: int=512, nhead: int=8, dropout: float=0.1):
        super(AVQCrossAttn, self).__init__()
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def sub_forward(self, src_q: Tensor, src_v: Tensor, query: Tensor, visualize: bool=False) -> Tensor:
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        (qst_attn, weight) = self.qst_attn(src_q, query, query)
        slf_attn = self.slf_attn(src_q, src_q, src_q)[0]
        crs_attn = self.crs_attn(src_q, src_v, src_v)[0]
        src_q = src_q + self.dropout(slf_attn) + self.dropout(crs_attn) + self.dropout(qst_attn)
        src_q = self.norm1(src_q)
        src_q = src_q + self.dropout(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return (src_q.permute(1, 0, 2), weight)

    def forward(self, src_q: Tensor, src_v: Tensor, query: Tensor, visualize: bool=False) -> List[Tensor]:
        (src1, a_weight) = self.sub_forward(src_q, src_v, query, visualize)
        (src2, v_weight) = self.sub_forward(src_v, src_q, query, visualize)
        if visualize:
            return (src1, src2, [a_weight, v_weight])
        return (src1, src2)

class QstGrounding(nn.Module):

    def __init__(self, d_model: int=512, nhead: int=8, dropout: float=0.1):
        super(QstGrounding, self).__init__()
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model))
        self.dropout = nn.Dropout(dropout)
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, qst: Tensor, data: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(data, list):
            data = [d.permute(1, 0, 2) for d in data]
            data = torch.cat(data, dim=0)
        else:
            data = data.permute(1, 0, 2)
        qst = qst.unsqueeze(0)
        attn = self.attn(qst, data, data)[0].squeeze(0)
        feat = data.mean(dim=0) + self.dropout(self.mlp(attn))
        feat = self.norm(feat)
        return feat

class TempMoE(nn.Module):

    def __init__(self, d_model: int=512, nhead: int=8, topK: int=5, n_experts: int=10, sigma: int=9, dropout: float=0.1, vis_branch: bool=False):
        super(TempMoE, self).__init__()
        self.sigma = sigma
        self.topK = topK
        self.n_experts = n_experts
        if vis_branch:
            self.anorm = nn.LayerNorm(d_model)
            self.vnorm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.qst_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.gauss_pred = nn.Sequential(nn.Linear(d_model, 2 * n_experts))
        self.router = nn.Sequential(nn.Linear(d_model, n_experts))
        self.experts = nn.ModuleList([nn.Sequential(*[nn.Linear(d_model, int(d_model // 2)), nn.ReLU(), nn.Linear(int(d_model // 2), d_model)]) for _ in range(n_experts)])
        self.experts.apply(self._init_weights)
        self.margin = 1 / (n_experts * 2)
        self.center = torch.linspace(self.margin, 1 - self.margin, self.n_experts)
        self.center.requires_grad_(False)
        self.router.apply(self._init_weights)
        self.gauss_pred.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_gaussian(self, pred: torch.Tensor, topk_inds: torch.Tensor, T: int=60) -> Tensor:
        weights = []
        centers = self.center.unsqueeze(0).repeat(pred.size(0), 1).to(pred.device)
        centers = centers + pred[:, :, 0]
        centers = torch.gather(centers, 1, topk_inds)
        widths = torch.gather(pred[:, :, 1], 1, topk_inds)
        for i in range(self.topK):
            center = centers[:, i]
            width = widths[:, i]
            weight = torch.linspace(0, 1, T)
            weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
            center = torch.clamp(center.unsqueeze(-1), min=0, max=1)
            width = torch.clamp(width.unsqueeze(-1), min=0.09) / self.sigma
            w = 0.3989422804014327
            weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))
            weights.append(weight / weight.max(dim=-1, keepdim=True)[0])
        return torch.stack(weights, dim=1)

    def get_output(self, experts_logits: Tensor, gauss_weight: Tensor, topk_inds: Tensor, topk_probs: Tensor, shape: tuple) -> Tensor:
        (B, T, C) = shape
        experts_logits = torch.gather(experts_logits.permute(1, 0, 2, 3).reshape(B * T, self.n_experts, -1), 1, topk_inds.repeat(T, 1).unsqueeze(-1).repeat(1, 1, C))
        experts_logits = experts_logits.reshape(B, T, self.topK, -1).contiguous()
        output = [gauss_weight[:, i, :].unsqueeze(1) @ experts_logits[:, :, i, :] for i in range(self.topK)]
        output = torch.cat(output, dim=1)
        output = topk_probs.unsqueeze(1) @ output
        return output

    def forward(self, qst: Tensor, data: Tensor, sub_data: Optional[Tensor]=None) -> Union[Tensor, List[Tensor]]:
        (B, T, C) = data.size()
        data = data.permute(1, 0, 2)
        qst = qst.unsqueeze(0)
        temp_w = self.qst_attn(qst, data, data)[0]
        temp_w = temp_w.squeeze(0)
        router_logits = self.router(temp_w)
        router_probs = F.softmax(router_logits, dim=-1)
        (topk_probs, topk_inds) = torch.topk(router_probs, self.topK, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        gauss_cw = self.gauss_pred(temp_w)
        gauss_cw = gauss_cw.view(B, self.n_experts, 2)
        gauss_cw[:, :, 0] = torch.tanh(gauss_cw[:, :, 0]) * self.margin
        gauss_cw[:, :, 1] = torch.sigmoid(gauss_cw[:, :, 1])
        gauss_weight = self.generate_gaussian(gauss_cw, topk_inds=topk_inds, T=T)
        if sub_data is not None:
            a_data = sub_data[0].permute(1, 0, 2)
            a_data = data + a_data
            a_outs = torch.stack([exprt(a_data) for exprt in self.experts], dim=2)
            a_outs = self.get_output(a_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))
            v_data = sub_data[1].permute(1, 0, 2)
            v_data = data + v_data
            v_outs = torch.stack([exprt(v_data) for exprt in self.experts], dim=2)
            v_outs = self.get_output(v_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))
            return (self.anorm(a_outs), self.vnorm(v_outs))
        else:
            main_outs = torch.stack([exprt(data) for exprt in self.experts], dim=2)
            main_outs = self.get_output(main_outs, gauss_weight, topk_inds, topk_probs, (B, T, C))
            return self.norm(main_outs)

class PatchSelecter(nn.Module):

    def __init__(self, d_model: int=512, nhead: int=8, dropout: float=0.1):
        super(PatchSelecter, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.vnorm = nn.LayerNorm(d_model)
        self.anorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.slf_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.crs_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, d_model))
        self.mlp.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, patch: Tensor, audio: Tensor, video: Tensor) -> List[Tensor]:
        (B, T, P, D) = patch.size()
        audio = audio.reshape(B * T, 1, D)
        video = video.reshape(B * T, 1, D)
        patch = patch.reshape(B * T, P, D)
        video = video.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)
        patch = patch.permute(1, 0, 2)
        patch = patch + self.slf_attn(patch, patch, patch)[0]
        data = torch.cat([video, audio], dim=0)
        attn = self.crs_attn(data, patch, patch)[0]
        attn = attn.permute(1, 0, 2)
        attn = self.mlp(self.dropout(attn))
        (v, a) = torch.chunk(attn, 2, dim=1)
        v = v.reshape(B, T, D).permute(1, 0, 2)
        a = a.reshape(B, T, D).permute(1, 0, 2)
        return [self.anorm(a.permute(1, 0, 2)), self.vnorm(v.permute(1, 0, 2))]

class TSPM_topKSelection(nn.Module):

    def __init__(self, topK: int=10):
        super(TSPM_topKSelection, self).__init__()
        self.topK = topK
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_visual_norm = nn.LayerNorm(512)

    def QstQueryClipAttn(self, query_feat, kv_feat):
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        (attn_feat, temp_weights) = self.attn_qst_query(query_feat, kv_feat, kv_feat, attn_mask=None, key_padding_mask=None)
        attn_feat = attn_feat.squeeze(0)
        src = self.qst_query_linear1(attn_feat)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)
        attn = attn_feat + src
        attn = self.qst_query_visual_norm(attn)
        return (attn, temp_weights)

    def SelectTopK(self, temp_weights, audio_input, visual_input, patch_inputs, B, C):
        sort_index = torch.argsort(temp_weights, dim=-1)
        top_k_index = sort_index[:, :, -self.topK:]
        (top_k_index_sort, indices) = torch.sort(top_k_index)
        top_k_index_sort = top_k_index_sort.cpu().numpy()
        output_audio = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_a_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        out_v_patches = torch.zeros(B, self.topK, C).to(audio_input.device)
        for batch_idx in range(B):
            idx = 0
            for temp_idx in top_k_index_sort.tolist()[batch_idx][0]:
                output_audio[batch_idx, idx, :] = audio_input[batch_idx, temp_idx, :]
                out_a_patches[batch_idx, idx, :] = patch_inputs[0][batch_idx, temp_idx, :]
                out_v_patches[batch_idx, idx, :] = patch_inputs[1][batch_idx, temp_idx, :]
                idx = idx + 1
        return (output_audio, (out_a_patches, out_v_patches))

    def forward(self, audio_input, visual_input, patch_inputs, qst_input):
        (B, T, C) = audio_input.size()
        (temp_clip_attn_feat, temp_weights) = self.QstQueryClipAttn(qst_input, visual_input)
        (output_audio, output_patches) = self.SelectTopK(temp_weights, audio_input, visual_input, patch_inputs, B, C)
        return (output_audio, output_patches)

class AdaptivePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.1):
        super().__init__()
        hidden = max(d_model // 2, 128)
        self.audio_proj = nn.Linear(1, hidden)
        self.motion_proj = nn.Linear(1, hidden)
        self.question_proj = nn.Linear(d_model, hidden)
        self.relation_proj = nn.Linear(6, hidden)
        self.phi_mlp = nn.Sequential(nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.time_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _normalize(x: Tensor) -> Tensor:
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True, unbiased=False) + 1e-06)

    def forward(self, seq_feat: Tensor, question_feat: Tensor, audio_activity: Tensor, visual_motion: Tensor) -> Tensor:
        (bsz, steps, _) = seq_feat.shape
        a = self._normalize(audio_activity).unsqueeze(-1)
        m = self._normalize(visual_motion).unsqueeze(-1)
        rel_time = torch.linspace(0.0, 1.0, steps, device=seq_feat.device, dtype=seq_feat.dtype).view(1, steps).expand(bsz, -1)
        audio_onset = F.relu(audio_activity[:, 1:] - audio_activity[:, :-1])
        motion_onset = F.relu(visual_motion[:, 1:] - visual_motion[:, :-1])
        audio_onset = F.pad(self._normalize(audio_onset), (1, 0))
        motion_onset = F.pad(self._normalize(motion_onset), (1, 0))
        activity = self._normalize(audio_activity + visual_motion)
        cumulative = torch.cumsum(activity, dim=1) / torch.arange(1, steps + 1, device=seq_feat.device, dtype=seq_feat.dtype).view(1, -1)
        co_activity = self._normalize(audio_activity) * self._normalize(visual_motion)
        active = (co_activity > 0).to(seq_feat.dtype)
        last = torch.cummax(torch.where(active > 0, torch.arange(steps, device=seq_feat.device).view(1, -1), torch.zeros(1, device=seq_feat.device, dtype=torch.long)), dim=1).values
        quiet_gap = (torch.arange(steps, device=seq_feat.device).view(1, -1) - last).to(seq_feat.dtype) / max(steps - 1, 1)
        relations = torch.stack((rel_time, audio_onset, motion_onset, cumulative, co_activity, quiet_gap), dim=-1)
        q = self.question_proj(question_feat).unsqueeze(1).expand(-1, steps, -1)
        phi = self.phi_mlp(torch.cat((self.audio_proj(a), self.motion_proj(m), q, self.relation_proj(relations)), dim=-1))
        gate = F.softmax(self.time_gate(phi).squeeze(-1), dim=1).unsqueeze(-1) * steps
        return seq_feat + self.residual_scale * self.dropout(gate * phi)
