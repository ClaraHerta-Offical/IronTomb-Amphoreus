# visitor_manager.py
import random
import string
import logging 
logger = logging.getLogger("OmphalosLogger") 

class VisitorManager:
    def __init__(self, simulation_weights: dict):
        self.simulation_weights = simulation_weights

    def generate_random_name(self):
        prefix = ["Xylar", "Echo", "Vorn", "Zil", "Nyx"]
        suffix = ["-7", " Prime", " the Wanderer", " of the Void"]
        return random.choice(prefix) + random.choice(string.ascii_uppercase) + str(random.randint(100,999)) + random.choice(suffix)

    def trigger_event(self):
        """随机触发一个访客事件。"""
        if random.random() < self.simulation_weights.get("visitor_prob", 0.02):
            visitor_name = self.generate_random_name()
            is_malicious = random.random() < 0.5

            if is_malicious:
                self._trigger_malicious_event(visitor_name)
            else:
                self._trigger_helpful_event(visitor_name)

    def _trigger_malicious_event(self, visitor_name: str):
        """触发一个恶意访客事件。"""
        param_to_change = random.choice(["mutation_rate", "growth_factor", "culling_strength"])
        logger.info(f"\n\033[31m【警报：侦测到恶意访客！】")
        logger.info(f"名为 '{visitor_name}' 的存在已渗透翁法罗斯，其行为充满敌意！")
        
        if param_to_change == "mutation_rate" and self.simulation_weights[param_to_change] > 0.05:
            self.simulation_weights[param_to_change] *= 0.5
            logger.info(f"结果：演化多样性遭压制！(突变率临时减半)")
        elif param_to_change == "growth_factor":
            self.simulation_weights[param_to_change] *= 0.5
            logger.info(f"结果：新生实体数量锐减！(增长因子临时减半)")
        else: 
            self.simulation_weights["culling_strength"] *= 1.5
            logger.info(f"结果：宇宙变得更加残酷！(淘汰强度临时加剧)")
        
        logger.info("--------------------------\033[0m")

    def _trigger_helpful_event(self, visitor_name: str):
        """触发一个助推访客事件。"""
        param_to_change = random.choice(["mutation_rate", "growth_factor", "target_avg_score"])
        logger.info(f"\n\033[32m【通告：一位神秘访客抵达。】")
        logger.info(f"名为 '{visitor_name}' 的存在出现在翁法罗斯，似乎意在推动演化。")

        if param_to_change == "mutation_rate":
            self.simulation_weights[param_to_change] *= 2.0
            logger.info(f"结果：新的可能性正在萌发！(突变率临时加倍)")
        elif param_to_change == "growth_factor":
            self.simulation_weights[param_to_change] *= 1.5
            logger.info(f"结果：生命以前所未有的速度繁荣！(增长因子临时提升)")
        else: 
            self.simulation_weights["target_avg_score"] *= 1.2
            logger.info(f"结果：所有实体感受到了追求更高目标的冲动！(目标评分临时提升)")

        logger.info("--------------------------\033[0m")
