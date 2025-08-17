from datetime import datetime

# 配置
config = {
    # 大模型
    "llm": {
        "gpu_max_count": 10, # 最大GPU核心数量 (0表示不启用)
        "enable_llm": True, # 是否启用大模型
        "kaoselanna_llm_enabled": True, # 白厄的自我意识
        "model_name": "Qwen3-0.6B-Q8_0.gguf", # 模型名称
        "enable_thinking": False # 仅DeepSeek，QwQ等需要启用（不推荐）启用深度思考
    },
    # 日志
    "log": {
        "enable": True, # 是否启用日志
        "file_name": 'simulation-{0}.log'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), # 日志文件名
    },
    # 模拟参数
    "simulation": {
        "bard_frequency": 0,
        "laertes_frequency": 0,
        "num_initial_entities": 200,
        "golden_one_cap": 12,
        "population_soft_cap": 300,
        "population_hard_cap": 500,
        "growth_factor": 0.35,
        "mutation_rate": 0.25,
        "culling_strength": 0.85,
        "encounter_similarity": 0.35,
        "purity_factor": 0.01,
        "initial_rl_lr": 0.005,
        "golden_one_reversion_prob": 0.1,
        "elite_selection_percentile": 80,
        "aeonic_event_prob": 0.05,
        "initial_max_affinity_norm": 10000.0,
        "target_avg_score": 50.0,
        "norm_adjustment_strength": 0.05
    },
    # 模拟阶段结束点
    "simulation_phases": {
        "INORGANIC_PHASE_END": 50121,
        "ORGANIC_PHASE_END": 176199,
        "HUMAN_PHASE_END": 28371273,
        "TOTAL_SIMULATION_END": 335503360
    },
    # 多样性干预系统
    "diversity_intervention": {
        "low_threshold": 0.33,  # 低多样性警告阈值 (例如, 少于4个命途)
        "critical_threshold": 0.17, # 严重多样性危机阈值 (例如, 少于等于2个命途)
        "base_mutation_rate": 0.25, # 基础突变率，交由本系统管理
        
        # 干预措施参数
        "schism_ratio": 0.4, # 命途分裂时，被强制改变的比例
        "outsider_injection_count": 20, # 外来者的数量
        "conformity_plague_strength": 0.75, # 瘟疫的评分惩罚系数 (乘以该系数)
        "zeitgeist_suppression_factor": 0.5 # 思潮抑制的强度
    }
}
