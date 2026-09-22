from experiments.qast_ehr.configs.qast_ehr_base import build_config
config = build_config('D:/model/qast_ehr_b_seed713')
config['hyper_params']['model']['lambda_event'] = 0.05
config['hyper_params']['model']['lambda_evidence'] = 0.09
config['hyper_params']['optim']['lr'] = 0.00015
