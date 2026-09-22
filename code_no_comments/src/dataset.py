import ast
import torch
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.models.clip import tokenize
from src.prompt_matcher import match_prompt
from src.models.vggish import wavfile_to_examples
qtype2idx = {'Audio': {'Counting': 0, 'Comparative': 1}, 'Visual': {'Counting': 2, 'Location': 3}, 'Audio-Visual': {'Existential': 4, 'Counting': 5, 'Location': 6, 'Comparative': 7, 'Temporal': 8}}
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

class AVQA_dataset(Dataset):

    def __init__(self, config: dict, mode: str, transform: transforms.Compose=None):
        self.mode = mode
        self.config = config
        self.type = config.type
        self.root = config.data.root
        self.audio_feat = (ROOT / self.root / config.data.audio_feat).as_posix() if config.data.audio_feat is not None else None
        self.video_feat = (ROOT / self.root / config.data.video_feat).as_posix() if config.data.video_feat is not None else None
        self.patch_feat = (ROOT / self.root / config.data.patch_feat).as_posix() if config.data.patch_feat is not None else None
        self.quest_feat = (ROOT / self.root / config.data.quest_feat).as_posix() if config.data.quest_feat is not None else None
        self.prompt_feat = (ROOT / self.root / config.data.prompt_feat).as_posix() if config.data.prompt_feat is not None else None
        self.tokenizer = tokenize
        self.bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        self.size = config.data.img_size
        self.sample_rate = config.data.frame_sample_rate
        annot = f'{self.mode}_annot'
        annot = eval(f'self.config.data.{annot}')
        annot = ROOT / self.root / annot
        with open(file=annot.as_posix(), mode='r', encoding='utf-8') as f:
            self.samples = self.question_process(json.load(f))
        ans_quelen = self.get_max_question_length()
        self.answer_to_ix = ans_quelen['ans2ix']
        self.max_que_len = ans_quelen['max_que_len']
        self.config.num_labels = len(self.answer_to_ix)
        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)
        self.video_list = video_list
        self.video_len = 60 * len(video_list)
        self.cache = {}
        self.transform = transform if transform is not None else transforms.Compose([transforms.Resize((self.size, self.size)), transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    def __len__(self):
        return len(self.samples)

    def load_samples(self, sample):
        labels = torch.tensor(data=[self.answer_to_ix[sample['anser']]], dtype=torch.long)
        ques_type = ast.literal_eval(sample['type'])
        qtype_label = torch.tensor([qtype2idx[ques_type[0]][ques_type[1]]], dtype=torch.long)
        if self.quest_feat is not None:
            quest_id = sample['question_id']
            quest = np.load(Path(self.quest_feat) / f'{int(quest_id)}.npy')
            prompt = np.load(Path(self.prompt_feat) / f'{int(quest_id)}.npy')
        else:
            question = sample['question_content']
            quest = self.tokenizer(question, truncate=True).squeeze()
            prompt = self.tokenizer(sample['qprompt'], truncate=True).squeeze()
        name = sample['video_id']
        if self.video_feat is not None:
            video = np.load(Path(self.video_feat) / f'{name}.npy')
            video = torch.from_numpy(video)[::self.sample_rate]
            if self.patch_feat is not None:
                patch = np.load(Path(self.patch_feat) / f'{name}.npy')
                patch = torch.from_numpy(patch)[::self.sample_rate]
            else:
                patch = None
        else:
            frame_dir = ROOT / self.root / self.config.data.frames_dir / name
            frame_path = sorted(list(frame_dir.glob('*.jpg')))[:60]
            frame_path = frame_path[::self.sample_rate]
            video = torch.stack([self.transform(Image.open(frame_path[i]).convert('RGB')) for i in range(len(frame_path))], dim=0)
            patch = None
        if self.audio_feat is not None:
            audio_path = Path(self.audio_feat) / f'{name}.npy'
            if not audio_path.exists():
                legacy_audio_path = Path(self.audio_feat) / f'{name}.mp4.npy'
                if legacy_audio_path.exists():
                    audio_path = legacy_audio_path
            audio = np.load(audio_path.as_posix())
            audio = torch.from_numpy(audio)[::self.sample_rate]
        else:
            audio_dir = ROOT / self.root / self.config.data.audios_dir
            audio_path = audio_dir / f'{name}.wav'
            audio = wavfile_to_examples(audio_path.as_posix(), num_secs=60)
            audio = torch.from_numpy(audio)[::self.sample_rate]
        data = {'quest': quest, 'prompt': prompt, 'type': ques_type, 'label': labels, 'qtype_label': qtype_label, 'video': video, 'audio': audio, 'name': name}
        if patch is not None:
            data.update({'patch': patch})
        return data

    def __getitem__(self, index):
        sample = self.samples[index]
        batch = self.load_samples(sample)
        batch['_index'] = index
        return batch

    def question_process(self, samples):
        for (index, sample) in enumerate(samples):
            question = sample['question_content'].lstrip().rstrip().split(' ')
            question[-1] = question[-1][:-1]
            prompt = match_prompt(sample['question_content'], sample['templ_values'])
            templ_value_index = 0
            for word_index in range(len(question)):
                if '<' in question[word_index]:
                    question[word_index] = ast.literal_eval(sample['templ_values'])[templ_value_index]
                    templ_value_index = templ_value_index + 1
            samples[index]['question_content'] = ' '.join(question)
            samples[index]['qprompt'] = prompt
        return samples

    def get_max_question_length(self):
        ans_quelen = ROOT / self.root / self.config.data.ans_quelen
        if ans_quelen.exists():
            with open(file=ans_quelen.as_posix(), mode='r') as f:
                ans_quelen = json.load(f)
        else:
            ans_quelen = {}
            ans2ix = {}
            answer_index = 0
            max_que_len = 0
            train_path = ROOT / self.root / self.config.data.train_annot
            valid_path = ROOT / self.root / self.config.data.valid_annot
            with open(file=train_path.as_posix(), mode='r') as f:
                samples = json.load(f)
            for sample in tqdm(samples):
                que_tokens = self.tokenizer(sample['question_content'].lstrip().rstrip()[:-1])
                que_len = len(torch.nonzero(que_tokens['input_ids']))
                if max_que_len < que_len:
                    max_que_len = que_len
                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1
            with open(file=valid_path.as_posix(), mode='r') as f:
                samples = json.load(f)
            for sample in samples:
                que_tokens = self.tokenizer(sample['question_content'].lstrip().rstrip()[:-1])
                que_len = len(torch.nonzero(que_tokens['input_ids']))
                if max_que_len < que_len:
                    max_que_len = que_len
                if ans2ix.get(sample['anser']) is None:
                    ans2ix[sample['anser']] = answer_index
                    answer_index += 1
            save_path = ROOT / self.root / self.config.data.ans_quelen
            with open(file=save_path.as_posix(), mode='w') as f:
                ans_quelen['ans2ix'] = ans2ix
                ans_quelen['max_que_len'] = max_que_len
                json.dump(obj=ans_quelen, fp=f)
        return ans_quelen
