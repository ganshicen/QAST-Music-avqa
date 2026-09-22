from copy import deepcopy
from experiments.qast_ehr.configs.qast_ehr_a_seed713 import config as candidate_a
SWITCHES = ('enable_cmg', 'enable_scm', 'enable_adaptive_position', 'enable_patch_grounder')

def build_ablation(output_dir: str, disabled: str) -> dict:
    if disabled not in SWITCHES:
        raise ValueError(f'disabled must be one of {SWITCHES}, got {disabled!r}')
    config = deepcopy(candidate_a)
    config['output_dir'] = output_dir
    config['seed'] = 713
    config['epochs'] = 15
    config['weight'] = ''
    config['distill_logits'] = None
    config['hyper_params']['model'].update({switch: switch != disabled for switch in SWITCHES})
    return config
