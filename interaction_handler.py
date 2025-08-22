import numpy as np
import random
import math

from constants import PATH_NAMES, TITAN_NAMES
from population_manager import PopulationManager
from entities import Pathstrider
from models import TitanToPathModel

import logging 
logger = logging.getLogger("OmphalosLogger") 

class InteractionHandler:
    def __init__(self, PATH_RELATIONSHIP_MATRIX: np.ndarray, PATH_INTERACTION_MATRIX: np.ndarray, 
                 encounter_similarity: float, culling_strength: float, 
                 population_soft_cap: int, titan_to_path_model: TitanToPathModel,
                 population_manager: PopulationManager):
        
        self.PATH_RELATIONSHIP_MATRIX = PATH_RELATIONSHIP_MATRIX
        self.PATH_INTERACTION_MATRIX = PATH_INTERACTION_MATRIX
        self.encounter_similarity = encounter_similarity
        self.culling_strength = culling_strength
        self.population_soft_cap = population_soft_cap
        self.titan_to_path_model = titan_to_path_model
        self.population_manager = population_manager
        self._bins = {}
        self._bin_size = 10 # 每个桶10分

    def update_bins(self, population: list):
        """在每代开始时调用，更新分桶。"""
        self._bins.clear()
        for entity in population:
            bin_key = int(entity.score // self._bin_size)
            if bin_key not in self._bins:
                self._bins[bin_key] = []
            self._bins[bin_key].append(entity)

    def select_opponents(self, population: list, reincarnator: Pathstrider):
        if len(population) < 2: return None, None
        
        entity1 = np.random.choice(population)
        
        # --- 选择逻辑 --- 8/23/0.49
        bin_key = int(entity1.score // self._bin_size)
        
        potential_opponents = []
        # 从实体自己的桶、前一个桶、后一个桶里找对手
        for i in range(bin_key - 1, bin_key + 2):
            if i in self._bins:
                potential_opponents.extend(self._bins[i])
        
        # 移除自己
        potential_opponents = [p for p in potential_opponents if p is not entity1]

        if not potential_opponents:
            # Fallback: 如果邻近桶是空的，随机选一个
            fallback_pool = [p for p in population if p != entity1]
            if not fallback_pool: return entity1, None 
            return entity1, np.random.choice(fallback_pool)

        return entity1, np.random.choice(potential_opponents)

    def entity_interaction(self, strider1: Pathstrider, strider2: Pathstrider, 
                           population: list, reincarnator: Pathstrider, 
                           global_path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray,
                           zeitgeist_multiplier: float): 
        participants = [strider1, strider2]
        golden_one = next((p for p in participants if p.trait == "GoldenOne" and p.titan_aspect), None)
        titan_boss = next((p for p in participants if p.is_titan_boss), None)

        if golden_one and titan_boss and golden_one.titan_aspect == titan_boss.titan_aspect:
            return self._handle_titan_battle(golden_one, titan_boss)

        is_reincarnator_involved = reincarnator and reincarnator in {strider1, strider2}
        
        if is_reincarnator_involved and strider1.dominant_path_idx == strider2.dominant_path_idx:
            other_entity = strider2 if reincarnator is strider1 else strider1
            if other_entity.trait != "GoldenOne":
                logger.info(f"\033[91m卡厄斯兰那在其命途 '{PATH_NAMES[reincarnator.dominant_path_idx]}' 上对敌人 {other_entity.name} 发动了无条件的毁灭！\033[0m")
                return other_entity
        
        participants = {strider1, strider2}
        if reincarnator in participants:
            other_entity = next((p for p in participants if p is not reincarnator), None)
            if other_entity:
                try:
                    neg_world_idx = TITAN_NAMES.index("负世")
                    if other_entity.trait == "GoldenOne": reincarnator.titan_affinities[neg_world_idx] *= 1.2
                    else: reincarnator.titan_affinities[neg_world_idx] *= 1.02
                    self.population_manager.recalculate_and_normalize_entity(reincarnator, global_path_distribution, cosmic_zeitgeist)
                except (ValueError, IndexError): pass
        
        if strider1.trait == "GoldenOne" or strider2.trait == "GoldenOne": return None
        
        stronger, weaker = (strider1, strider2) if strider1.score > strider2.score else (strider2, strider1)
        if np.random.random() < math.tanh((stronger.score - weaker.score) * self.culling_strength):
            return weaker
        
        dom_path1, dom_path2 = strider1.dominant_path_idx, strider2.dominant_path_idx
        interaction_type = self.PATH_RELATIONSHIP_MATRIX[dom_path1, dom_path2]
        
        if interaction_type == "CLASH": return self._handle_clash(strider1, strider2, global_path_distribution, cosmic_zeitgeist)
        elif interaction_type == "SYNERGY": self._handle_synergy(strider1, strider2, global_path_distribution, cosmic_zeitgeist)
        elif interaction_type == "REPULSION": self._handle_repulsion(strider1, strider2, global_path_distribution, cosmic_zeitgeist)
        elif interaction_type == "MENTORSHIP": self._handle_mentorship_or_assimilation(strider1, strider2, global_path_distribution, cosmic_zeitgeist)
        
        return None

    def _handle_clash(self, strider1: Pathstrider, strider2: Pathstrider, 
                      global_path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        if not (np.isfinite(strider1.score) and np.isfinite(strider2.score)): return None
        winner, loser = (strider1, strider2) if strider1.score * (1 + np.random.rand()*0.2) > strider2.score else (strider2, strider1)
        modifier = self.PATH_INTERACTION_MATRIX[winner.dominant_path_idx, loser.dominant_path_idx]
        influence = loser.titan_affinities * 0.1 * modifier
        winner.titan_affinities += influence
        loser.titan_affinities -= influence * 0.5
        self.population_manager.recalculate_and_normalize_entity(winner, global_path_distribution, cosmic_zeitgeist)
        self.population_manager.recalculate_and_normalize_entity(loser, global_path_distribution, cosmic_zeitgeist)
        if np.random.rand() < 0.1 * (winner.score / (loser.score + 1e-6) - 1): return loser
        return None

    def _handle_synergy(self, strider1: Pathstrider, strider2: Pathstrider, 
                        global_path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        strider1.titan_affinities += strider2.titan_affinities * 0.02
        strider2.titan_affinities += strider1.titan_affinities * 0.02
        self.population_manager.recalculate_and_normalize_entity(strider1, global_path_distribution, cosmic_zeitgeist)
        self.population_manager.recalculate_and_normalize_entity(strider2, global_path_distribution, cosmic_zeitgeist)
    
    def _handle_repulsion(self, strider1: Pathstrider, strider2: Pathstrider, 
                          global_path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        for s in [strider1, strider2]:
            dom_idx = np.argmax(s.titan_affinities)
            damage = s.titan_affinities[dom_idx] * 0.05
            s.titan_affinities[dom_idx] -= damage
            s.titan_affinities += np.random.rand(len(TITAN_NAMES)) * damage * 0.1
            self.population_manager.recalculate_and_normalize_entity(s, global_path_distribution, cosmic_zeitgeist)

    def _handle_mentorship_or_assimilation(self, strider1: Pathstrider, strider2: Pathstrider, 
                                           global_path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        stronger, weaker = (strider1, strider2) if strider1.score > strider2.score else (strider2, strider1)
        if not (np.isfinite(stronger.score) and np.isfinite(weaker.score) and weaker.score > 0): return
        score_ratio = stronger.score / (weaker.score + 1e-6)
        influence_strength = 0.05 * math.tanh(score_ratio - 1) 
        direction_vector = stronger.titan_affinities - weaker.titan_affinities
        weaker.titan_affinities += direction_vector * influence_strength
        weaker_influence_on_stronger = (weaker.titan_affinities - stronger.titan_affinities) * 0.005
        stronger.titan_affinities += weaker_influence_on_stronger
        self.population_manager.recalculate_and_normalize_entity(stronger, global_path_distribution, cosmic_zeitgeist)
        self.population_manager.recalculate_and_normalize_entity(weaker, global_path_distribution, cosmic_zeitgeist)

    def _handle_titan_battle(self, golden_one: Pathstrider, titan_boss: Pathstrider):
        """处理金裔对其对应的泰坦的战斗。"""
        logger.info(f"\n\033[35m【逐火之旅】黄金裔 {golden_one.name} ({golden_one.titan_aspect}) 挑战泰坦 {titan_boss.name}！\033[0m")
        
        damage = golden_one.score * (0.2 + random.random() * 0.3)
        titan_boss.hp -= damage
        
        logger.info(f"{golden_one.name} 对泰坦造成了 {damage:.2f} 点伤害！泰坦剩余HP: {titan_boss.hp:.2f}")

        if titan_boss.hp <= 0:
            logger.info(f"\033[92m泰坦 {titan_boss.name} 已被击败！黄金裔 {golden_one.name} 继承了其火种，成为半神！\033[0m")
            golden_one.data_modification_unlocked = True
            golden_one.score *= 1.5 
            return titan_boss 
        
        if random.random() < 0.05: 
            logger.info(f"\033[91m泰坦发起了反击！{golden_one.name} 灰飞烟灭！\033[0m")
            return golden_one
            
        return None
