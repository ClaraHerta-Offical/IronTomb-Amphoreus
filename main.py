# main.py

import os
import sys
import traceback
import colorama
from simulation import AeonEvolution
import logging
from config import config
from datetime import datetime

logging.basicConfig(
    filename=config['log']['file_name'],
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s'
)

class LoggerWriter:
    def __init__(self, level, stream):
        self.level = level
        self.stream = stream
        self._buffer = ''
    def write(self, message):
        if message != '\n':
            self._buffer += message
        if '\n' in message:
            self.flush()
        self.stream.write('[{0}][{1}] {2}\r\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.level.__name__, self._buffer))  # 同时输出到控制台
    def flush(self):
        if self._buffer:
            self.level(self._buffer)
            self._buffer = ''
        self.stream.flush()

def run_simple():
    """ 以简单的控制台模式运行模拟 """
    colorama.init()  # 初始化 colorama 以在 Windows 上启用 ANSI 颜色代码
    sim = None
    try:
        sim = AeonEvolution(
            # --- LLM 控制 ---
            bard_frequency=0,         
            laertes_frequency=0,      
            kaoselanna_llm_enabled=config['llm']['kaoselanna_llm_enabled'],

            # --- 其他模拟参数 ---
            num_initial_entities=200, 
            golden_one_cap=12, 
            population_soft_cap=300,
            population_hard_cap=500, 
            growth_factor=0.35, 
            mutation_rate=0.25,
            culling_strength=0.85, 
            encounter_similarity=0.35, 
            purity_factor=0.01,
            initial_rl_lr=0.005, 
            golden_one_reversion_prob=0.1,
            elite_selection_percentile=80, 
            aeonic_event_prob=0.05,
            initial_max_affinity_norm=10000.0, 
            target_avg_score=50.0,
            norm_adjustment_strength=0.05
        )
        print("=== 翁法罗斯 v10.3 (Dev) 启动 ===")
        sim.start(num_generations=config['generations'])

    except KeyboardInterrupt:
        print("\n\n模拟被用户中断。正在退出...")
        # sys.exit(0) 

    except Exception:
        # 在发生其他异常时打印完整的追溯信息
        traceback.print_exc()

    finally:
        # 确保在模拟结束或中断时保存模型
        if sim and hasattr(sim, 'policy_saver'):
            print("\n正在尝试保存策略模型...")
            sim.policy_saver.save_policy_models()
        
        print("\n模拟已结束。按任意键退出。")
        if os.name == 'nt':
            os.system('pause')
        sys.exit(0) 


if __name__ == "__main__":
    if config['log']['enable']:
        sys.stdout = LoggerWriter(logging.info, sys.stdout)
        sys.stderr = LoggerWriter(logging.error, sys.stderr)
    run_simple()