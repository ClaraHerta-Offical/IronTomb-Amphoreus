# diversity_manager.py
import numpy as np
import random
from collections import Counter
from constants import PATH_NAMES, TITAN_NAMES

import logging 
logger = logging.getLogger("OmphalosLogger") 

class DiversityInterventionManager:
    def __init__(self, config, population_manager, parliament_manager):
        self.config = config['diversity_intervention']
        self.population_manager = population_manager
        self.parliament_manager = parliament_manager
        self.active_intervention = None
        self.intervention_duration = 0
        
        # --- 映射字典 ---
        self.intervention_display_names = {
            'path_schism': '命途分裂',
            'outsider_injection': '天外来客',
            'conformity_plague': '服从瘟疫',
            'zeitgeist_suppression': '思潮抑制',
            'minority_subsidy': '异端扶持'
        }

    def _calculate_diversity_index(self, population: list):
        """计算当前的多样性指数。"""
        if not population:
            return 0.0, []
        
        path_counts = Counter(p.dominant_path_idx for p in population)
        num_present_paths = len(path_counts)
        diversity_index = num_present_paths / len(PATH_NAMES)
        
        return diversity_index, path_counts

    def assess_and_intervene(self, population: list, generation: int):
        """每代评估一次，并在必要时触发干预。"""
        # 如果当前有干预正在进行，则倒计时
        if self.intervention_duration > 0:
            self.intervention_duration -= 1
            if self.intervention_duration == 0:
                display_name = self.intervention_display_names.get(self.active_intervention, self.active_intervention)
                logger.info(f"\033[36m【干预结束】 翁法罗斯的秩序正在从 ‘{display_name}’ 事件中恢复。\033[0m")
                
                # 关键的重置逻辑
                if self.active_intervention == 'zeitgeist_suppression':
                    self.parliament_manager.suppression_factor = 1.0
                    
                self.active_intervention = None
            return # 在干预期间，不进行新的评估

        diversity_index, path_counts = self._calculate_diversity_index(population)
        
        # 根据配置文件中的阈值决定干预等级
        if diversity_index <= self.config['critical_threshold']:
            self._trigger_critical_intervention(population, path_counts)
        elif diversity_index <= self.config['low_threshold']:
            self._trigger_low_intervention(population, path_counts)
        
        # 更新StagnationManager中的突变率
        self.population_manager.mutation_rate = self._get_adaptive_mutation_rate(diversity_index)

    def _get_adaptive_mutation_rate(self, diversity_index):
        """根据多样性动态调整突变率，可以替代StagnationManager中的旧逻辑。"""
        base_rate = self.config['base_mutation_rate']
        if diversity_index < 0.15:
            return base_rate * 3.0  # 极低多样性，大幅提高突变
        elif diversity_index < 0.3:
            return base_rate * 2.0  # 低多样性，提高突变
        else:
            return base_rate # 恢复正常

    def _trigger_low_intervention(self, population: list, path_counts: Counter):
        """当多样性过低时，采取温和的干预措施。"""
        intervention_choice = random.choice(['zeitgeist_suppression', 'minority_subsidy'])
        self.active_intervention = intervention_choice
        self.intervention_duration = 15 # 干预持续15代

        if intervention_choice == 'zeitgeist_suppression':
            logger.info("\n\033[33m【多样性干预：思潮抑制】 翁法罗斯的主流思想受到质疑！\033[0m")
            self.parliament_manager.suppression_factor = self.config['zeitgeist_suppression_factor']
        
        elif intervention_choice == 'minority_subsidy':
            logger.info("\n\033[33m【多样性干预：异端扶持】 翁法罗斯开始保护少数派思想，为罕见的命途提供生存空间。\033[0m")
            self._subsidize_minorities(population, path_counts)

    def _trigger_critical_intervention(self, population: list, path_counts: Counter):
        """当多样性处于崩溃边缘时，采取激烈的休克疗法。"""
        intervention_choice = random.choice(['path_schism', 'outsider_injection', 'conformity_plague'])
        self.active_intervention = intervention_choice
        self.intervention_duration = 1 # 干预持续1代，立即生效

        dominant_path_idx = path_counts.most_common(1)[0][0]

        if intervention_choice == 'path_schism':
            logger.info(f"\n\033[91m【多样性危机：命途分裂】 主流命途 '{PATH_NAMES[dominant_path_idx]}' 内部发生分裂，被迫重新选择立场！\033[0m")
            self._execute_path_schism(population, dominant_path_idx)

        elif intervention_choice == 'outsider_injection':
            logger.info(f"\n\033[91m【多样性危机：天外来客】 翁法罗斯的死寂吸引了未知的外来者，带来了全新的外来变量！\033[0m")
            self._inject_outsiders(population, path_counts)

        elif intervention_choice == 'conformity_plague':
            logger.info(f"\n\033[91m【多样性危机：服从瘟疫】 一场针对主流思想的瘟疫爆发，所有 '{PATH_NAMES[dominant_path_idx]}' 的追随者都感到了莫名的虚弱！\033[0m")
            self._apply_conformity_plague(population, dominant_path_idx)
    

    def _subsidize_minorities(self, population, path_counts):
        """为少数派或不存在的命途的实体提供评分加成。"""
        minority_paths = {i for i, name in enumerate(PATH_NAMES) if i not in path_counts}
        for entity in population:
            if entity.dominant_path_idx in minority_paths:
                entity.score *= 1.25 # 给予25%的评分加成

    def _execute_path_schism(self, population, dominant_path_idx):
        """强制将部分主流命途的实体突变为其他命途。"""
        followers = [p for p in population if p.dominant_path_idx == dominant_path_idx]
        num_to_convert = int(len(followers) * self.config['schism_ratio'])
        converts = random.sample(followers, num_to_convert)

        opposites = {"毁灭": "存护", "存护": "毁灭", "巡猎": "繁育", "繁育": "巡猎"}
        target_path_name = opposites.get(PATH_NAMES[dominant_path_idx], random.choice(PATH_NAMES))
        target_path_idx = PATH_NAMES.index(target_path_name)
        
        # 找到与目标命途关联最强的泰坦
        titan_influence = self.population_manager.titan_to_path_model.titan_to_path_matrix[:, target_path_idx]
        most_influential_titan_idx = np.argmax(titan_influence)

        for entity in converts:
            # 大幅提升与新命途相关的泰坦亲和度
            entity.titan_affinities[most_influential_titan_idx] *= 3.0
            entity.titan_affinities += np.random.normal(0, 5, len(TITAN_NAMES)) # 增加混乱
            self.population_manager.recalculate_and_normalize_entity(entity, None, None)

    def _inject_outsiders(self, population, path_counts):
        """创造一批全新的、持有稀有命途的实体。"""
        missing_paths = [i for i, name in enumerate(PATH_NAMES) if i not in path_counts]
        if not missing_paths:
            missing_paths = [path_counts.most_common()[-1][0]] # 如果都有，就选最少的

        for i in range(self.config['outsider_injection_count']):
            target_path_idx = random.choice(missing_paths)
            
            # 创建一个倾向于该命途的亲和度蓝图
            blueprint = np.zeros(len(TITAN_NAMES))
            titan_influence = self.population_manager.titan_to_path_model.titan_to_path_matrix[:, target_path_idx]
            strongest_titan_indices = np.argsort(titan_influence)[-3:] # 找到影响最大的3个泰坦
            blueprint[strongest_titan_indices] = 10.0
            
            name = self.population_manager.generate_unique_name()
            # 导入 Pathstrider 类
            from entities import Pathstrider
            new_affinities = blueprint + np.random.normal(0, 2, len(TITAN_NAMES))
            entity = Pathstrider(name, new_affinities, titan_to_path_model=self.population_manager.titan_to_path_model)
            
            self.population_manager.recalculate_and_normalize_entity(entity, None, None)
            population.append(entity)
            self.population_manager.name_to_entity_map[name] = entity

    def _apply_conformity_plague(self, population, dominant_path_idx):
        """对主流命途的实体施加一个临时的评分惩罚。"""
        for entity in population:
            if entity.dominant_path_idx == dominant_path_idx:
                entity.score *= self.config['conformity_plague_strength']
