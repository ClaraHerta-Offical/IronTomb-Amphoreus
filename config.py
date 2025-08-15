from datetime import datetime

# 配置
config = {
    "generations": 500, # 轮回次数
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
    }
}