from copy import deepcopy
from experiments.qast_ehr.ablations.configs.without_scm_seed713 import config as seed713
config = deepcopy(seed713)
config['seed'] = 123
config['output_dir'] = 'D:/model/qast_ehr_ablation_without_scm_seed123'
