from pathlib import Path
import numpy as np
import torch.nn as nn

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def get_tome_feat(model_size: str='base', patch_size: int=32, image_size: int=224, save_dir='./data/feats/visual_tome14'):
    model = timm.create_model(f'vit_{model_size}_patch{patch_size}_{image_size}', pretrained=True).to('cuda')
    model.head = Identity()
    model.global_pool = None
    tome.patch.timm(model, trace_source=True)
    model = model.to('cuda')
    if patch_size == 16 and image_size == 384:
        model.r = [25] * 23
    elif patch_size == 32 and image_size == 224:
        model.r = 2
    data = torch.randn(1, 3, image_size, image_size).to('cuda')
    output = model(data)
    print(output.shape)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for (i, batch) in enumerate(loader):
        reshaped_data = get_items(batch, 'cuda')
        print(f'process {i} / {len(loader)}')
        names = batch['name']
        B = len(names)
        all_exists = True
        for i in range(B):
            save_path = f'{save_dir}/{names[i]}.npy'
            if Path(save_path).exists():
                print(f'File {save_path} exists')
                continue
            all_exists = False
        if all_exists:
            continue
        data = reshaped_data['video']
        (B, T) = data.shape[:2]
        with torch.no_grad():
            output = model(data.reshape(B * T, *data.shape[2:]))
            output = output.reshape(B, T, *output.shape[1:])
        for i in range(B):
            save_path = f'{save_dir}/{names[i]}.npy'
            if Path(save_path).exists():
                print(f'File {save_path} exists')
                continue
            print(f'File {save_path} saved')
            np.save(save_path, output[i].cpu().numpy())
if __name__ == '__main__':
    import sys
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]
    sys.path.append(ROOT.as_posix())
    from box import Box
    from src.utils import set_logger, logging_config
    from src.trainutils import get_items
    from src.dataset import AVQA_dataset
    from configs.music_avqa.ours_vitl14 import config as conf_vitl14
    from configs.music_avqa.ours_vitb32 import config as conf_vitb32
    import torch
    import src.tome as tome
    import timm
    model_type = 'vitb32'
    if model_type == 'vitb32':
        conf_vitb32 = Box(conf_vitb32)
        conf_vitb32.debug = True
        conf_vitb32.weight = 'none.pt'
        conf_vitb32.data.video_feat = None
        conf_vitb32.data.patch_feat = None
        for mode in ['train', 'valid', 'test']:
            conf_vitb32.mode = mode
            set_logger(conf_vitb32)
            logging_config(conf_vitb32)
            vitb32_data = AVQA_dataset(conf_vitb32, conf_vitb32.mode)
            B = 4
            loader = torch.utils.data.DataLoader(vitb32_data, num_workers=16, batch_size=B, shuffle=False)
            get_tome_feat(model_size='base', patch_size=32, image_size=224, save_dir='./data/feats/visual_tome14_b32_r2')
    elif model_type == 'vitl14':
        cfg_vitl14 = Box(conf_vitl14)
        cfg_vitl14.debug = True
        cfg_vitl14.weight = 'none.pt'
        cfg_vitl14.data.video_feat = None
        cfg_vitl14.data.patch_feat = None
        for mode in ['train', 'valid', 'test']:
            cfg_vitl14.mode = mode
            set_logger(cfg_vitl14)
            logging_config(cfg_vitl14)
            vitl14_data = AVQA_dataset(cfg_vitl14, cfg_vitl14.mode)
            B = 4
            loader = torch.utils.data.DataLoader(vitl14_data, num_workers=16, batch_size=B, shuffle=False)
            get_tome_feat(model_size='large', patch_size=16, image_size=384, save_dir='./data/feats/visual_tome14')
