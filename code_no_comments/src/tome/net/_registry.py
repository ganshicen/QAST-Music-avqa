import fnmatch
import re
import sys
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple
from ._pretrained import PretrainedCfg, DefaultCfg
__all__ = ['split_model_name_tag', 'get_arch_name', 'register_model', 'generate_default_cfgs', 'list_models', 'list_pretrained', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules', 'get_pretrained_cfg_value', 'is_model_pretrained']
_module_to_models: Dict[str, Set[str]] = defaultdict(set)
_model_to_module: Dict[str, str] = {}
_model_entrypoints: Dict[str, Callable[..., Any]] = {}
_model_has_pretrained: Set[str] = set()
_model_default_cfgs: Dict[str, PretrainedCfg] = {}
_model_pretrained_cfgs: Dict[str, PretrainedCfg] = {}
_model_with_tags: Dict[str, List[str]] = defaultdict(list)
_module_to_deprecated_models: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
_deprecated_models: Dict[str, Optional[str]] = {}

def split_model_name_tag(model_name: str, no_tag: str='') -> Tuple[str, str]:
    (model_name, *tag_list) = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return (model_name, tag)

def get_arch_name(model_name: str) -> str:
    return split_model_name_tag(model_name)[0]

def generate_default_cfgs(cfgs: Dict[str, Union[Dict[str, Any], PretrainedCfg]]):
    out = defaultdict(DefaultCfg)
    default_set = set()
    for (k, v) in cfgs.items():
        if isinstance(v, dict):
            v = PretrainedCfg(**v)
        has_weights = v.has_weights
        (model, tag) = split_model_name_tag(k)
        is_default_set = model in default_set
        priority = has_weights and (not tag) or (tag.endswith('*') and (not is_default_set))
        tag = tag.strip('*')
        default_cfg = out[model]
        if priority:
            default_cfg.tags.appendleft(tag)
            default_set.add(model)
        elif has_weights and (not default_cfg.is_pretrained):
            default_cfg.tags.appendleft(tag)
        else:
            default_cfg.tags.append(tag)
        if has_weights:
            default_cfg.is_pretrained = True
        default_cfg.cfgs[tag] = v
    return out

def register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        default_cfg = mod.default_cfgs[model_name]
        if not isinstance(default_cfg, DefaultCfg):
            assert isinstance(default_cfg, dict)
            pretrained_cfg = PretrainedCfg(**default_cfg)
            default_cfg = DefaultCfg(tags=deque(['']), cfgs={'': pretrained_cfg})
        for (tag_idx, tag) in enumerate(default_cfg.tags):
            is_default = tag_idx == 0
            pretrained_cfg = default_cfg.cfgs[tag]
            model_name_tag = '.'.join([model_name, tag]) if tag else model_name
            replace_items = dict(architecture=model_name, tag=tag if tag else None)
            if pretrained_cfg.hf_hub_id and pretrained_cfg.hf_hub_id == 'timm/':
                replace_items['hf_hub_id'] = pretrained_cfg.hf_hub_id + model_name_tag
            pretrained_cfg = replace(pretrained_cfg, **replace_items)
            if is_default:
                _model_pretrained_cfgs[model_name] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    _model_has_pretrained.add(model_name)
            if tag:
                _model_pretrained_cfgs[model_name_tag] = pretrained_cfg
                if pretrained_cfg.has_weights:
                    _model_has_pretrained.add(model_name_tag)
                _model_with_tags[model_name].append(model_name_tag)
            else:
                _model_with_tags[model_name].append(model_name)
        _model_default_cfgs[model_name] = default_cfg
    return fn

def _deprecated_model_shim(deprecated_name: str, current_fn: Callable=None, current_tag: str=''):

    def _fn(pretrained=False, **kwargs):
        assert current_fn is not None, f'Model {deprecated_name} has been removed with no replacement.'
        current_name = '.'.join([current_fn.__name__, current_tag]) if current_tag else current_fn.__name__
        warnings.warn(f'Mapping deprecated model name {deprecated_name} to current {current_name}.', stacklevel=2)
        pretrained_cfg = kwargs.pop('pretrained_cfg', None)
        return current_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg or current_tag, **kwargs)
    return _fn

def register_model_deprecations(module_name: str, deprecation_map: Dict[str, Optional[str]]):
    mod = sys.modules[module_name]
    module_name_split = module_name.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''
    for (deprecated, current) in deprecation_map.items():
        if hasattr(mod, '__all__'):
            mod.__all__.append(deprecated)
        current_fn = None
        current_tag = ''
        if current:
            (current_name, current_tag) = split_model_name_tag(current)
            current_fn = getattr(mod, current_name)
        deprecated_entrypoint_fn = _deprecated_model_shim(deprecated, current_fn, current_tag)
        setattr(mod, deprecated, deprecated_entrypoint_fn)
        _model_entrypoints[deprecated] = deprecated_entrypoint_fn
        _model_to_module[deprecated] = module_name
        _module_to_models[module_name].add(deprecated)
        _deprecated_models[deprecated] = current
        _module_to_deprecated_models[module_name][deprecated] = current

def _natural_key(string_: str) -> List[Union[int, str]]:
    return [int(s) if s.isdigit() else s for s in re.split('(\\d+)', string_.lower())]

def _expand_filter(filter: str):
    (filter_base, filter_tag) = split_model_name_tag(filter)
    if not filter_tag:
        return ['.'.join([filter_base, '*']), filter]
    else:
        return [filter]

def list_models(filter: Union[str, List[str]]='', module: str='', pretrained: bool=False, exclude_filters: Union[str, List[str]]='', name_matches_cfg: bool=False, include_tags: Optional[bool]=None) -> List[str]:
    if filter:
        include_filters = filter if isinstance(filter, (tuple, list)) else [filter]
    else:
        include_filters = []
    if include_tags is None:
        include_tags = pretrained
    all_models: Set[str] = _module_to_models[module] if module else set(_model_entrypoints.keys())
    all_models = all_models - _deprecated_models.keys()
    if include_tags:
        models_with_tags: Set[str] = set()
        for m in all_models:
            models_with_tags.update(_model_with_tags[m])
        all_models = models_with_tags
        include_filters = [ef for f in include_filters for ef in _expand_filter(f)]
        exclude_filters = [ef for f in exclude_filters for ef in _expand_filter(f)]
    if include_filters:
        models: Set[str] = set()
        for f in include_filters:
            include_models = fnmatch.filter(all_models, f)
            if len(include_models):
                models = models.union(include_models)
    else:
        models = all_models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)
            if len(exclude_models):
                models = models.difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    if name_matches_cfg:
        models = set(_model_pretrained_cfgs).intersection(models)
    return sorted(models, key=_natural_key)

def list_pretrained(filter: Union[str, List[str]]='', exclude_filters: str='') -> List[str]:
    return list_models(filter=filter, pretrained=True, exclude_filters=exclude_filters, include_tags=True)

def get_deprecated_models(module: str='') -> Dict[str, str]:
    all_deprecated = _module_to_deprecated_models[module] if module else _deprecated_models
    return deepcopy(all_deprecated)

def is_model(model_name: str) -> bool:
    arch_name = get_arch_name(model_name)
    return arch_name in _model_entrypoints

def model_entrypoint(model_name: str, module_filter: Optional[str]=None) -> Callable[..., Any]:
    arch_name = get_arch_name(model_name)
    if module_filter and arch_name not in _module_to_models.get(module_filter, {}):
        raise RuntimeError(f'Model ({model_name} not found in module {module_filter}.')
    return _model_entrypoints[arch_name]

def list_modules() -> List[str]:
    modules = _module_to_models.keys()
    return sorted(modules)

def is_model_in_modules(model_name: str, module_names: Union[Tuple[str, ...], List[str], Set[str]]) -> bool:
    arch_name = get_arch_name(model_name)
    assert isinstance(module_names, (tuple, list, set))
    return any((arch_name in _module_to_models[n] for n in module_names))

def is_model_pretrained(model_name: str) -> bool:
    return model_name in _model_has_pretrained

def get_pretrained_cfg(model_name: str, allow_unregistered: bool=True) -> Optional[PretrainedCfg]:
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    (arch_name, tag) = split_model_name_tag(model_name)
    if arch_name in _model_default_cfgs:
        raise RuntimeError(f'Invalid pretrained tag ({tag}) for {arch_name}.')
    if allow_unregistered:
        return None
    raise RuntimeError(f'Model architecture ({arch_name}) has no pretrained cfg registered.')

def get_pretrained_cfg_value(model_name: str, cfg_key: str) -> Optional[Any]:
    cfg = get_pretrained_cfg(model_name, allow_unregistered=False)
    return getattr(cfg, cfg_key, None)
