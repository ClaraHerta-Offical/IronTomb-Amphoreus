# main.py

import os
import sys
import traceback
import colorama
from simulation import AeonEvolution
import logging
from config import config
from datetime import datetime
import argparse 
from pyclog import ClogFileHandler, constants # 导入 pyclog
import re # 导入 re 模块

colorama.init(autoreset=True) # 确保在sys.stdout重定向前初始化colorama

# 创建一个 logger 实例
logger = logging.getLogger("OmphalosLogger")
logger.setLevel(logging.INFO)

if config['log']['enable']:
    # 使用 ClogFileHandler 替换 FileHandler
    clog_handler = ClogFileHandler(config['log']['file_name'], compression_code=constants.COMPRESSION_GZIP)
    file_formatter = logging.Formatter('%(message)s')
    clog_handler.setFormatter(file_formatter)
    logger.addHandler(clog_handler)

# 终端输出仍然使用 StreamHandler
class CustomConsoleFormatter(logging.Formatter):
    def format(self, record):
        # 获取完整的格式化日志行
        log_line = super().format(record)
        
        # 查找消息部分的起始位置
        # 格式字符串是 '[%(asctime)s][%(levelname)s] %(message)s'
        # 我们需要找到 ']' 之后的第一个空格，作为消息的起始
        match = re.match(r'\[.*?\]\[.*?\]\s*', log_line)
        if match:
            prefix_length = match.end()
        else:
            prefix_length = 0 # 如果没有匹配到，则不缩进

        # 将日志行按换行符分割
        lines = log_line.splitlines()
        
        # 对除第一行以外的每一行添加缩进
        if len(lines) > 1:
            indented_lines = [lines[0]]
            for line in lines[1:]:
                indented_lines.append(' ' * prefix_length + line)
            return '\n'.join(indented_lines)
        else:
            return log_line

# 终端输出仍然使用 StreamHandler
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = CustomConsoleFormatter('[%(asctime)s][%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 移除 LoggerWriter 和 sys.stdout/sys.stderr 重定向
# if config['log']['enable']:
#     sys.stdout = LoggerWriter(logger.info)
#     sys.stderr = LoggerWriter(logger.error)

def run_simple(load_save_path=None):
    """ 以简单的控制台模式运行模拟 """
    
    sim = None
    try:
        sim = AeonEvolution(
            # --- LLM ---
            bard_frequency=config['simulation']['bard_frequency'],         
            laertes_frequency=config['simulation']['laertes_frequency'],      
            kaoselanna_llm_enabled=config['llm']['kaoselanna_llm_enabled'],

            fast_forward=args.fast_forward if 'args' in locals() and hasattr(args, 'fast_forward') else False,
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
        if load_save_path:
            sim.load_simulation_state(load_save_path)

        logger.info("=== 翁法罗斯 v10.4 (Dev) 启动 ===")
        sim.start(
            num_generations=config['simulation_phases']['TOTAL_SIMULATION_END']
        )

    except KeyboardInterrupt:
        logger.info("\n\n模拟被用户中断。正在退出...")
        

    except Exception:
        traceback.print_exc()

    finally:
        if sim and hasattr(sim, 'policy_saver'):
            logger.info("\n正在尝试保存策略模型...")
            sim.policy_saver.save_policy_models()
            logger.info("\n正在尝试保存clog...")
            logging.shutdown()
        
        print("\n模拟已结束。按任意键退出。")
        if os.name == 'nt':
            os.system('pause')
        sys.exit(0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翁法罗斯")
    parser.add_argument('--disable-llm', action='store_true', 
                        help='彻底禁用LLM参与演算，无需安装llama_cpp。')
    parser.add_argument('--load-save', type=str, default=None,
                        help='从指定的存档文件加载并开始模拟。')
    parser.add_argument('--fast-forward', action='store_true',
                        help='快速演化模式，仅显示进度条和周期性报告。')
    global args; args = parser.parse_args() # 定义为全局变量以便run_simple访问
 
    # 根据参数更配
    if args.disable_llm:
        config['llm']['enable_llm'] = False
        logger.info("\033[93m命令行参数 --disable-llm 已启用，LLM功能已彻底禁用。\033[0m")
    
    if args.fast_forward:
        # 找到控制台处理器并抑制其输出，以实现快速模式
        found_handler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                logger.info("\033[93m快速演化模式已启用。控制台输出将受到抑制。\033[0m")
                handler.setLevel(logging.CRITICAL + 1) # 将等级设为极高，使其忽略所有常规日志
                found_handler = True
        if not found_handler:
             logger.warning("警告：无法找到控制台日志处理器，--fast-forward可能无法完全生效。")

    run_simple()
