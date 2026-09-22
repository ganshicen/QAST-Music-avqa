import copy
from collections import deque, defaultdict
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Deque, Dict, Tuple, Optional, Union
__all__ = ['PretrainedCfg', 'filter_pretrained_cfg', 'DefaultCfg']

@dataclass
class PretrainedCfg:
    url: Optional[Union[str, Tuple[str, str]]] = None
    file: Optional[str] = None
    state_dict: Optional[Dict[str, Any]] = None
    hf_hub_id: Optional[str] = None
    hf_hub_filename: Optional[str] = None
    source: Optional[str] = None
    architecture: Optional[str] = None
    tag: Optional[str] = None
    custom_load: bool = False
    input_size: Tuple[int, int, int] = (3, 224, 224)
    test_input_size: Optional[Tuple[int, int, int]] = None
    min_input_size: Optional[Tuple[int, int, int]] = None
    fixed_input_size: bool = False
    interpolation: str = 'bicubic'
    crop_pct: float = 0.875
    test_crop_pct: Optional[float] = None
    crop_mode: str = 'center'
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    num_classes: int = 1000
    label_offset: Optional[int] = None
    label_names: Optional[Tuple[str]] = None
    label_descriptions: Optional[Dict[str, str]] = None
    pool_size: Optional[Tuple[int, ...]] = None
    test_pool_size: Optional[Tuple[int, ...]] = None
    first_conv: Optional[str] = None
    classifier: Optional[str] = None
    license: Optional[str] = None
    description: Optional[str] = None
    origin_url: Optional[str] = None
    paper_name: Optional[str] = None
    paper_ids: Optional[Union[str, Tuple[str]]] = None
    notes: Optional[Tuple[str]] = None

    @property
    def has_weights(self):
        return self.url or self.file or self.hf_hub_id

    def to_dict(self, remove_source=False, remove_null=True):
        return filter_pretrained_cfg(asdict(self), remove_source=remove_source, remove_null=remove_null)

def filter_pretrained_cfg(cfg, remove_source=False, remove_null=True):
    filtered_cfg = {}
    keep_null = {'pool_size', 'first_conv', 'classifier'}
    for (k, v) in cfg.items():
        if remove_source and k in {'url', 'file', 'hf_hub_id', 'hf_hub_id', 'hf_hub_filename', 'source'}:
            continue
        if remove_null and v is None and (k not in keep_null):
            continue
        filtered_cfg[k] = v
    return filtered_cfg

@dataclass
class DefaultCfg:
    tags: Deque[str] = field(default_factory=deque)
    cfgs: Dict[str, PretrainedCfg] = field(default_factory=dict)
    is_pretrained: bool = False

    @property
    def default(self):
        return self.cfgs[self.tags[0]]

    @property
    def default_with_tag(self):
        tag = self.tags[0]
        return (tag, self.cfgs[tag])
