# main.py

import os
import sys
import traceback
import colorama
from simulation import AeonEvolution
import logging
from config import config
from datetime import datetime
import argparse #

# 创建一个 logger 实例
logger = logging.getLogger("OmphalosLogger")
logger.setLevel(logging.INFO)

if config['log']['enable']:
    file_handler = logging.FileHandler(config['log']['file_name'], mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(message)s') 
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        # logging 模块会自己处理 flush，这里 pass 即可
        pass

if config['log']['enable']:
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

def run_simple():
    """ 以简单的控制台模式运行模拟 """
    colorama.init()
    
    sim = None
    try:
        sim = AeonEvolution(
            # --- LLM ---
            bard_frequency=config['simulation']['bard_frequency'],         
            laertes_frequency=config['simulation']['laertes_frequency'],      
            kaoselanna_llm_enabled=config['llm']['kaoselanna_llm_enabled'],

            # --- 其他模拟参数 ---
            num_initial_entities=config['simulation']['num_initial_entities'], 
            golden_one_cap=config['simulation']['golden_one_cap'], 
            population_soft_cap=config['simulation']['population_soft_cap'],
            population_hard_cap=config['simulation']['population_hard_cap'], 
            growth_factor=config['simulation']['growth_factor'], 
            mutation_rate=config['simulation']['mutation_rate'],
            culling_strength=config['simulation']['culling_strength'], 
            encounter_similarity=config['simulation']['encounter_similarity'], 
            purity_factor=config['simulation']['purity_factor'],
            initial_rl_lr=config['simulation']['initial_rl_lr'], 
            golden_one_reversion_prob=config['simulation']['golden_one_reversion_prob'],
            elite_selection_percentile=config['simulation']['elite_selection_percentile'], 
            aeonic_event_prob=config['simulation']['aeonic_event_prob'],
            initial_max_affinity_norm=config['simulation']['initial_max_affinity_norm'], 
            target_avg_score=config['simulation']['target_avg_score'],
            norm_adjustment_strength=config['simulation']['norm_adjustment_strength']
        )
        print("=== 翁法罗斯 v10.4 (Dev) 启动 ===")
        sim.start(
            num_generations=config['simulation_phases']['TOTAL_SIMULATION_END']
        )

    except KeyboardInterrupt:
        print("\n\n模拟被用户中断。正在退出...")
        

    except Exception:
        traceback.print_exc()

    finally:
        if sim and hasattr(sim, 'policy_saver'):
            print("\n正在尝试保存策略模型...")
            sim.policy_saver.save_policy_models()
        
        print("\n模拟已结束。按任意键退出。")
        if os.name == 'nt':
            os.system('pause')
        sys.exit(0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翁法罗斯")
    parser.add_argument('--disable-llm', action='store_true', 
                        help='彻底禁用LLM参与演算，无需安装llama_cpp。')
    args = parser.parse_args()

    # 根据参数更配
    if args.disable_llm:
        config['llm']['enable_llm'] = False
        print("\033[93m命令行参数 --disable-llm 已启用，LLM功能已彻底禁用。\033[0m")

    run_simple()
