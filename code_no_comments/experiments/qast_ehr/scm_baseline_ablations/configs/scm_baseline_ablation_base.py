from copy import deepcopy
from experiments.qast_ehr.configs.qast_ehr_a_seed713 import config as full_config
TARGETS = ('enable_cmg', 'enable_adaptive_position', 'enable_patch_grounder')

def build_scm_baseline_ablation(output_dir: str, additionally_disabled: str) -> dict:
    if additionally_disabled not in TARGETS:
        raise ValueError(f'additionally_disabled must be one of {TARGETS}, got {additionally_disabled!r}')
    config = deepcopy(full_config)
    config['output_dir'] = output_dir
    config['seed'] = 713
    config['epochs'] = 15
    config['weight'] = ''
    config['distill_logits'] = None
    switches = {'enable_cmg': True, 'enable_scm': False, 'enable_adaptive_position': True, 'enable_patch_grounder': True}
    switches[additionally_disabled] = False
    config['hyper_params']['model'].update(switches)
    return config
