from copy import deepcopy
from experiments.qast_ehr.configs.qast_ehr_a_seed713 import config as full_config
STRUCTURAL_SWITCHES = ('enable_evidence_router', 'enable_event_memory', 'enable_temporal_reader', 'enable_dynamic_fusion')
LOSS_WEIGHTS = ('lambda_tacl', 'lambda_sacl', 'lambda_event', 'lambda_evidence', 'lambda_diversity', 'lambda_position')

def common(output_dir: str) -> dict:
    config = deepcopy(full_config)
    config['output_dir'] = output_dir
    config['seed'] = 713
    config['epochs'] = 15
    config['weight'] = ''
    config['distill_logits'] = None
    model = config['hyper_params']['model']
    model['enable_scm'] = False
    model.update({switch: True for switch in STRUCTURAL_SWITCHES})
    return config

def structural(output_dir: str, disabled: str) -> dict:
    if disabled not in STRUCTURAL_SWITCHES:
        raise ValueError(f'unknown structural switch: {disabled}')
    config = common(output_dir)
    config['hyper_params']['model'][disabled] = False
    return config

def loss(output_dir: str, disabled: str) -> dict:
    if disabled not in LOSS_WEIGHTS:
        raise ValueError(f'unknown loss weight: {disabled}')
    config = common(output_dir)
    config['hyper_params']['model'][disabled] = 0.0
    return config
