import numpy as np
import torch
from constants import PATH_NAMES

class PrimordialLegacyManager:
    def __init__(self):
        self.is_initialized = False
        self.influence_factor = 1.0  # 初始影响力为100%
        self.decay_rate = 0.995 # 每代影响力衰减率，这个值越接近1，影响越持久

        # 遗产转化后的修正值
        self.newborn_affinity_modifier = None
        self.complexity_modifier = 1.0

    def initialize(self, inorganic_legacy, organic_legacy_model):
        """接收遗产并将其转化为具体的、可应用的修正值。"""
        print("\n\033[35m【初始倾向调优】正在解析宇宙的初始参数...")
        
        # 实体亲和度修正
        # 活性影响前6个命途, 稳定性影响后6个
        activity_strength = inorganic_legacy['avg_activity']
        stability_strength = inorganic_legacy['avg_stability']
        
        # 将影响力度归一化到-1到1的范围，再乘以一个系数
        total_potency = activity_strength + stability_strength
        activity_bias = (activity_strength - stability_strength) / total_potency if total_potency > 0 else 0
        
        self.newborn_affinity_modifier = np.zeros(len(PATH_NAMES))
        self.newborn_affinity_modifier[:6] = activity_bias * 0.5 # 活性倾向修正
        self.newborn_affinity_modifier[6:] = -activity_bias * 0.5 # 稳定性倾向修正
        print(f"...无机命途倾向修正值 (偏向活性: {activity_bias:.2f})")

        # 宇宙复杂度修正
        # 使用模型所有权重的L2范数来衡量模型的复杂度
        with torch.no_grad():
            all_weights = torch.cat([p.view(-1) for p in organic_legacy_model.parameters()])
            complexity_norm = torch.norm(all_weights).item()
        
        # 将复杂度映射到1.0到1.5之间，影响突变率
        # 这个映射关系可以根据需要调整
        self.complexity_modifier = 1.0 + 0.5 * np.tanh((complexity_norm - 10.0) / 5.0)
        print(f"...有机修正值: {self.complexity_modifier:.2f} (将影响突变率)")

        self.is_initialized = True
        print("【初始倾向调优】初始化完毕，其影响将贯穿整个时代。\033[0m")

    def apply_legacy_effects(self, simulation):
        """在每个世代应用初始效果，并执行衰减。"""
        if not self.is_initialized or self.influence_factor <= 0.01:
            return

        # 应用复杂度修动态调整基础突变率
        base_mutation_rate = simulation.stagnation_manager.base_mutation_rate
        effective_mutation_rate = base_mutation_rate * self.complexity_modifier * self.influence_factor + base_mutation_rate * (1 - self.influence_factor)
        simulation.stagnation_manager.base_mutation_rate = effective_mutation_rate
        
        # 衰减影响力
        self.influence_factor *= self.decay_rate
        
        if simulation.generation % 200 == 0:
             print(f"\033[36m[倾向报告] 当前初始倾向影响力: {self.influence_factor:.2%} | 有效突变率基准: {effective_mutation_rate:.4f}\033[0m")
