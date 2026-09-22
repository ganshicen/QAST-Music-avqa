import dataclasses
import logging
import os
from copy import deepcopy
from typing import Optional, Dict, Callable, Any, Tuple
from torch import nn as nn
from torch.hub import load_state_dict_from_url
from timm.models._features import FeatureListNet, FeatureHookNet
from timm.models._features_fx import FeatureGraphNet
from timm.models._helpers import load_state_dict
from timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf
from timm.models._manipulate import adapt_input_conv
from timm.models._pretrained import PretrainedCfg
from timm.models._prune import adapt_model_from_file
from timm.models._registry import get_pretrained_cfg
_logger = logging.getLogger(__name__)
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False
_USE_OLD_CACHE = int(os.environ.get('TIMM_USE_OLD_CACHE', 0)) > 0
__all__ = ['set_pretrained_download_progress', 'set_pretrained_check_hash', 'load_custom_pretrained', 'load_pretrained', 'pretrained_cfg_for_features', 'resolve_pretrained_cfg', 'build_model_with_cfg']

def _resolve_pretrained_source(pretrained_cfg):
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    pretrained_sd = pretrained_cfg.get('state_dict', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    elif pretrained_sd:
        load_from = 'state_dict'
        pretrained_loc = pretrained_sd
        assert isinstance(pretrained_loc, dict)
    elif pretrained_file:
        load_from = 'file'
        pretrained_loc = pretrained_file
    else:
        old_cache_valid = False
        if _USE_OLD_CACHE:
            old_cache_valid = check_cached_file(pretrained_url) if pretrained_url else False
        if not old_cache_valid and hf_hub_id and has_hf_hub(necessary=True):
            load_from = 'hf-hub'
            pretrained_loc = hf_hub_id
        elif pretrained_url:
            load_from = 'url'
            pretrained_loc = pretrained_url
    if load_from == 'hf-hub' and pretrained_cfg.get('hf_hub_filename', None):
        pretrained_loc = (pretrained_loc, pretrained_cfg['hf_hub_filename'])
    return (load_from, pretrained_loc)

def set_pretrained_download_progress(enable=True):
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable

def set_pretrained_check_hash(enable=True):
    global _CHECK_HASH
    _CHECK_HASH = enable

def load_custom_pretrained(model: nn.Module, pretrained_cfg: Optional[Dict]=None, load_fn: Optional[Callable]=None):
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning('Invalid pretrained config, cannot load weights.')
        return
    (load_from, pretrained_loc) = _resolve_pretrained_source(pretrained_cfg)
    if not load_from:
        _logger.warning('No pretrained weights exist for this model. Using random initialization.')
        return
    if load_from == 'hf-hub':
        _logger.warning('Hugging Face hub not currently supported for custom load pretrained models.')
    elif load_from == 'url':
        pretrained_loc = download_cached_file(pretrained_loc, check_hash=_CHECK_HASH, progress=_DOWNLOAD_PROGRESS)
    if load_fn is not None:
        load_fn(model, pretrained_loc)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(pretrained_loc)
    else:
        _logger.warning('Valid function to load pretrained weights is not available, using random initialization.')

def load_pretrained(model: nn.Module, pretrained_cfg: Optional[Dict]=None, num_classes: int=1000, in_chans: int=3, filter_fn: Optional[Callable]=None, strict: bool=True):
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        raise RuntimeError('Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.')
    (load_from, pretrained_loc) = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'state_dict':
        _logger.info(f'Loading pretrained weights from state dict')
        state_dict = pretrained_loc
    elif load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(pretrained_loc, progress=_DOWNLOAD_PROGRESS, check_hash=_CHECK_HASH)
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict_from_url(pretrained_loc, map_location='cpu', progress=_DOWNLOAD_PROGRESS, check_hash=_CHECK_HASH)
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            state_dict = load_state_dict_from_hf(*pretrained_loc)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc)
    else:
        model_name = pretrained_cfg.get('architecture', 'this model')
        raise RuntimeError(f'No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.')
    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            state_dict = filter_fn(state_dict)
    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')
    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]
    model.load_state_dict(state_dict, strict=strict)

def pretrained_cfg_for_features(pretrained_cfg):
    pretrained_cfg = deepcopy(pretrained_cfg)
    to_remove = ('num_classes', 'classifier', 'global_pool')
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg

def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)

def _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        default_kwarg_names += ('img_size',)
    for n in default_kwarg_names:
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])
    _filter_kwargs(kwargs, names=kwargs_filter)

def resolve_pretrained_cfg(variant: str, pretrained_cfg=None, pretrained_cfg_overlay=None) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)
    if not pretrained_cfg:
        _logger.warning(f'No pretrained configuration specified for {model_with_tag} model. Using a default. Please add a config to the model pretrained_cfg registry or pass explicitly.')
        pretrained_cfg = PretrainedCfg()
    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)
    return pretrained_cfg

def build_model_with_cfg(model_cls: Callable, variant: str, pretrained: bool, pretrained_cfg: Optional[Dict]=None, pretrained_cfg_overlay: Optional[Dict]=None, model_cfg: Optional[Any]=None, feature_cfg: Optional[Dict]=None, pretrained_strict: bool=True, pretrained_filter_fn: Optional[Callable]=None, kwargs_filter: Optional[Tuple[str]]=None, **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=pretrained_cfg, pretrained_cfg_overlay=pretrained_cfg_overlay)
    pretrained_cfg = pretrained_cfg.to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter)
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pruned:
        model = adapt_model_from_file(model, variant)
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(model, pretrained_cfg=pretrained_cfg, num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3), filter_fn=pretrained_filter_fn, strict=pretrained_strict)
    if features:
        feature_cls = FeatureListNet
        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None:
            feature_cfg.setdefault('output_fmt', output_fmt)
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)
        model.default_cfg = model.pretrained_cfg
    return model
