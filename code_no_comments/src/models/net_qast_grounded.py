from __future__ import annotations
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.models.modules import AdaptivePositionalEncoding
from src.models.tspm import AVHanLayer, AV_Attn, QstTemporalGrounding, SpatioPerceptionModule, TSPM, TemporalPerception

class MotionSoundQuestionSpatialGrounding(SpatioPerceptionModule):

    def __init__(self, topK: int=10, d_model: int=512):
        super().__init__(topK=topK)
        self.question_to_query = nn.Linear(d_model, d_model)
        self.prompt_to_query = nn.Linear(d_model, d_model)
        self.motion_to_query = nn.Linear(1, d_model)
        self.query_scale = nn.Parameter(torch.zeros(1))
        self.context_refine = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.refine_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _patch_motion(patches: Tensor) -> Tensor:
        delta = (patches[:, 1:] - patches[:, :-1]).norm(p=2, dim=-1).mean(dim=-1)
        return F.pad(delta, (1, 0))

    def forward(self, audio_selected: Tensor, visual_patch: Tensor, question_prompt: Tensor, top_k_index_sort: Tensor, question: Tensor) -> Tensor:
        selected_patch = self.TopKSegs(visual_patch, top_k_index_sort)
        motion = self._patch_motion(visual_patch)
        if top_k_index_sort is not None:
            indices = torch.as_tensor(top_k_index_sort[:, 0, :], device=motion.device, dtype=torch.long)
            motion = torch.gather(motion, 1, indices)
        else:
            motion = motion[:, :audio_selected.size(1)]
        query_delta = self.question_to_query(question).unsqueeze(1) + self.prompt_to_query(question_prompt).unsqueeze(1) + self.motion_to_query(motion.unsqueeze(-1))
        grounded_query = audio_selected + self.query_scale * torch.tanh(query_delta)
        grounded_visual = self.AudioGuidedPatchAttn(grounded_query, selected_patch)
        q_tokens = question.unsqueeze(1).expand(-1, grounded_visual.size(1), -1)
        refinement = self.context_refine(torch.cat((grounded_visual, audio_selected, q_tokens), dim=-1))
        return grounded_visual + self.refine_scale * refinement

class QASTGroundedFusion(nn.Module):

    def __init__(self, topK: int=10, audio_dim: int=128, video_dim: int=768, patch_dim: int=1024, question_dim: int=768, d_model: int=512, num_labels: int=42, lambda_tacl: float=0.2, lambda_sacl: float=0.2, source_checkpoint: str | None=None, complementary_checkpoint: str | None=None, use_complementary: bool=False, primary_mix: float=0.4, freeze_shared: bool=False, ablate_semantic_context: bool=False, ablate_spatial_grounding: bool=False, ablate_cross_modal_grounding: bool=False, ablate_adaptive_position: bool=False, **kwargs):
        super().__init__()
        self.lambda_tacl = lambda_tacl
        self.lambda_sacl = lambda_sacl
        self.primary_mix = float(primary_mix)
        self.ablate_semantic_context = bool(ablate_semantic_context)
        self.ablate_spatial_grounding = bool(ablate_spatial_grounding)
        self.ablate_cross_modal_grounding = bool(ablate_cross_modal_grounding)
        self.ablate_adaptive_position = bool(ablate_adaptive_position)
        if not 0.0 <= self.primary_mix <= 1.0:
            raise ValueError('primary_mix must be in [0, 1]')
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.question_proj = nn.Linear(question_dim, d_model)
        self.prompt_proj = nn.Linear(question_dim, d_model)
        self.av_context = AV_Attn(AVHanLayer(d_model=d_model, nhead=1, dim_feedforward=d_model), num_layers=1)
        self.temporal_proposal = TemporalPerception(topK)
        self.spatial_grounding = MotionSoundQuestionSpatialGrounding(topK, d_model)
        self.temporal_grounding = QstTemporalGrounding()
        self.adaptive_position = AdaptivePositionalEncoding(d_model)
        with torch.no_grad():
            self.adaptive_position.residual_scale.zero_()
        self.cross_modal_grounding = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1, batch_first=True)
        self.cross_ground_norm = nn.LayerNorm(d_model)
        self.cross_ground_scale = nn.Parameter(torch.zeros(1))
        semantic_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.semantic_context = nn.TransformerEncoder(semantic_layer, num_layers=1)
        self.semantic_project = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.semantic_scale = nn.Parameter(torch.zeros(1))
        self.fusion_linear = nn.Linear(d_model * 6, d_model)
        self.answer_head = nn.Linear(d_model, num_labels)
        self.pretrained_report = None
        if source_checkpoint:
            self.pretrained_report = self._load_tspm_initialization(source_checkpoint)
        self.complementary_model = None
        if complementary_checkpoint or use_complementary:
            self.complementary_model = TSPM(topK=topK, audio_dim=audio_dim, vis_dim=video_dim, patch_dim=patch_dim, qst_dim=question_dim, hidden_size=d_model)
            if complementary_checkpoint:
                complement_state = torch.load(complementary_checkpoint, map_location='cpu')
                if 'state_dict' in complement_state:
                    complement_state = complement_state['state_dict']
                complement_state = {k.removeprefix('module.'): v for (k, v) in complement_state.items()}
                self.complementary_model.load_state_dict(complement_state, strict=True)
            self.complementary_model.requires_grad_(False)
            if complementary_checkpoint:
                if self.pretrained_report is None:
                    self.pretrained_report = {}
                self.pretrained_report['complementary_checkpoint'] = complementary_checkpoint
        if freeze_shared:
            self._freeze_transferred_path()

    def _freeze_transferred_path(self):
        shared_modules = (self.audio_proj, self.video_proj, self.patch_proj, self.question_proj, self.prompt_proj, self.av_context, self.temporal_proposal, self.temporal_grounding, self.fusion_linear, self.answer_head, self.spatial_grounding.attn_qst_query, self.spatial_grounding.qst_query_linear1, self.spatial_grounding.qst_query_linear2, self.spatial_grounding.qst_query_visual_norm, self.spatial_grounding.TokensAttn)
        for module in shared_modules:
            module.requires_grad_(False)

    @staticmethod
    def _motion(video: Tensor) -> Tensor:
        delta = (video[:, 1:] - video[:, :-1]).norm(p=2, dim=-1)
        return F.pad(delta, (1, 0))

    def _load_tspm_initialization(self, checkpoint: str):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'TSPM initialization checkpoint not found: {path}')
        state = torch.load(path, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.removeprefix('module.'): v for (k, v) in state.items()}
        prefix_map = {'input_a.': 'audio_proj.', 'input_v.': 'video_proj.', 'input_v_patch.': 'patch_proj.', 'input_qst.': 'question_proj.', 'input_qst_prompt.': 'prompt_proj.', 'AV_Attn.': 'av_context.', 'TemporalPerception.': 'temporal_proposal.', 'SpatioPerception.': 'spatial_grounding.', 'QstTempGrd_Module.': 'temporal_grounding.', 'av_fusion_fc.': 'fusion_linear.', 'answer_pred_fc.': 'answer_head.'}
        mapped = {}
        for (old_name, tensor) in state.items():
            for (old_prefix, new_prefix) in prefix_map.items():
                if old_name.startswith(old_prefix):
                    mapped[new_prefix + old_name[len(old_prefix):]] = tensor
                    break
        message = self.load_state_dict(mapped, strict=False)
        return {'checkpoint': str(path), 'loaded_tensors': len(mapped), 'missing_keys': list(message.missing_keys), 'unexpected_keys': list(message.unexpected_keys)}

    def forward(self, data: Dict[str, Tensor]):
        audio = self.audio_proj(data['audio'])
        video = self.video_proj(data['video'])
        patches = self.patch_proj(data['patch'])
        question = self.question_proj(data['quest'].squeeze(1))
        prompt = self.prompt_proj(data['prompt'].squeeze(1))
        audio_activity = audio.norm(p=2, dim=-1)
        visual_motion = self._motion(video)
        if not self.ablate_adaptive_position:
            audio = self.adaptive_position(audio, question, audio_activity, visual_motion)
            video = self.adaptive_position(video, question, audio_activity, visual_motion)
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
        if not self.ablate_semantic_context:
            semantic_tokens = torch.cat((question_for_reasoning.unsqueeze(1), audio_events, visual_events), dim=1)
            semantic = self.semantic_context(semantic_tokens)[:, 0]
            fused = fused + self.semantic_scale * self.semantic_project(semantic)
        answer_feature = torch.tanh(fused * question_for_reasoning)
        primary_logits = self.answer_head(answer_feature)
        if self.complementary_model is None:
            answer_logits = primary_logits
        else:
            self.complementary_model.eval()
            with torch.no_grad():
                complement_logits = self.complementary_model(data)['out']
            consensus = self.primary_mix * primary_logits.float().softmax(dim=-1) + (1.0 - self.primary_mix) * complement_logits.float().softmax(dim=-1)
            answer_logits = consensus.clamp_min(1e-08).log()
        output = {'out': answer_logits}
        if self.training:
            tacl = F.mse_loss(F.normalize(audio_events, dim=-1), F.normalize(video_events, dim=-1))
            sacl = (1.0 - F.cosine_similarity(F.normalize(question_for_reasoning, dim=-1), F.normalize(visual_events.mean(dim=1), dim=-1), dim=-1)).mean()
            output['loss_tacl'] = self.lambda_tacl * tacl
            output['loss_sacl'] = self.lambda_sacl * sacl
        return output
