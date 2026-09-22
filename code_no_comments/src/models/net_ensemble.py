import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.net import QA_AVAF
from src.models.net_tiger_adaptive import QA_TIGERAdaptive

class QASTTigerEnsemble(nn.Module):

    def __init__(self, qast_weight: str, tiger_weight: str, tiger_mix: float=0.65, d_model: int=512, video_dim: int=768, patch_dim: int=1024, audio_dim: int=128, encoder_type: str='ViT-L/14@336px'):
        super().__init__()
        self.qast = QA_AVAF(d_model=d_model, video_dim=video_dim, patch_dim=patch_dim, audio_dim=audio_dim, encoder_type=encoder_type, alpha=0.05, beta=0.01)
        self.tiger = QA_TIGERAdaptive(d_model=d_model, video_dim=video_dim, patch_dim=patch_dim, audio_dim=audio_dim, topK=7, num_experts=7, encoder_type=encoder_type)
        self.qast.load_state_dict(torch.load(qast_weight, map_location='cpu'), strict=False)
        self.tiger.load_state_dict(torch.load(tiger_weight, map_location='cpu'), strict=True)
        self.tiger_mix = float(tiger_mix)
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool=True):
        super().train(False)
        self.qast.eval()
        self.tiger.eval()
        return self

    def forward(self, batch):
        qast_logp = F.log_softmax(self.qast(batch)['out'], dim=-1)
        tiger_logp = F.log_softmax(self.tiger(batch)['out'], dim=-1)
        mixed_logp = (1.0 - self.tiger_mix) * qast_logp + self.tiger_mix * tiger_logp
        return {'out': mixed_logp}
