# cpu_llm_interface.py
from llama_cpp import Llama
import os
from config import config

class CpuLlmInterface:
    def __init__(self, model_folder="models"):
        """
        初始化并加载GGUF模型。
        """
        if not config['llm']['enable_llm']:
            print("\033[93m大模型功能已禁用。请在配置文件中启用。\033[0m")
            self.llm = None
            return

        model_name = config['llm']['model_name']
        model_path = os.path.join(model_folder, model_name)

        if not os.path.exists(model_path):
            print(f"\n\033[91m错误：找不到模型文件！\033[0m")
            print(f"请确保你已经下载了 '{model_name}'")
            print(f"并将其放置在项目的 '{model_folder}' 文件夹中。")
            return

        try:
            # 初始化Llama实例
            # n_ctx: 模型的上下文长度，即能处理多少token
            # n_gpu_layers: 设为0，强制模型完全在CPU上运行
            # verbose: 设为False，避免在控制台打印过多日志
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Qwen3的上下文窗口较大
                n_gpu_layers=config['llm']['gpu_max_count'],  # GPU核心数量
                verbose=False,
                enable_thinking=config['llm']['enable_thinking']
            )
            print("\033[92m模型加载成功！翁法罗斯拥有了新的低语者。\033[0m")
        except Exception as e:
            print(f"\n\033[91m模型加载时发生致命错误: {e}\033[0m")
            print("这可能是由于文件损坏或库安装不正确。")

    def generate_response(self, prompt, max_tokens=120):
        """
        使用加载的模型生成响应。
        """
        if self.llm is None:
            return "错误：模型未能成功加载，无法生成响应。"

        messages = [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用Llama.create_chat_completion来处理聊天格式
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7, 
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        return output['choices'][0]['message']['content'].strip()