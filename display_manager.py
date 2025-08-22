import sys
import time

import logging 
logger = logging.getLogger("OmphalosLogger") 

class DisplayManager:
    def __init__(self):
        self._last_update_time = 0

    def update_and_display_progress(self, phase: str, current_value: float, max_value: float):
        # 每隔 0.1 秒才真正更新一次屏幕，避免I/O过于频繁
        current_time = time.time()
        progress = (current_value / max_value) if max_value > 0 else 0
        if current_time - self._last_update_time < 0.1 and progress < 0.99:
             return
        self._last_update_time = current_time

        bar_length = 50
        
        progress = min(max(progress, 0), 1.0) 
        
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        phase_map = {
            'inorganic': '阶段一：无机推演', 
            'organic': '阶段二：有机测算', 
            'normal': '阶段三：凡人时代', 
            'cycle': '阶段四：永劫轮回'
        }
        phase_text = phase_map.get(phase, phase)
        
        message = f"演算进度: |{bar}| {progress:.1%} ({phase_text})"
        
        sys.stdout.write(f"\r{message}   ")
        sys.stdout.flush()

    def display_interruption_animation(self):
        spinner = ['/', '-', '\\', '|']
        message = "\033[91m警告：侦测异常反应... 原有进程已中断... 正在启动备用协议...\033[0m"
        logger.info(f"\n{message}")
        for i in range(20):
            spin_char = spinner[i % len(spinner)]
            sys.stdout.write(f"\r载入中... {spin_char}")
            sys.stdout.flush()
            time.sleep(0.1)
        logger.info("\n")
