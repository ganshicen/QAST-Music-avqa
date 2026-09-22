import torch
import torch.nn as nn
from src.models.clip import load

class CLIP_TEncoder(nn.Module):

    def __init__(self, model_name: str):
        super(CLIP_TEncoder, self).__init__()
        model = load(model_name, q_aware_N=-1, device='cpu')[0]
        self.dtype = model.dtype
        self.transformer = model.transformer
        self.vocab_size = model.vocab_size
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        del model

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)[:x.shape[1]]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return (x[torch.arange(x.shape[0]), torch.argmax(text, dim=-1)] @ self.text_projection, x)
