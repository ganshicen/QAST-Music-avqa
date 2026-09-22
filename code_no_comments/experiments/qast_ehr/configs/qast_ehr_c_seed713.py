from experiments.qast_ehr.configs.qast_ehr_base import build_config
config = build_config('D:/model/qast_ehr_c_seed713')
config['hyper_params']['model']['num_temporal_queries'] = 12
config['hyper_params']['model']['event_scales'] = (3, 9, 21)
config['hyper_params']['model']['position_max_strength'] = 0.11
config['hyper_params']['model']['lambda_evidence'] = 0.08
