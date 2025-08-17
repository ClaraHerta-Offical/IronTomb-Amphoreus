import numpy as np
import random
from collections import Counter

from constants import GREEK_ROOTS, TITAN_NAMES, PATH_NAMES
from entities import Pathstrider
from models import TitanToPathModel

class PopulationManager:
    def __init__(self, existing_names: set, name_to_entity_map: dict, 
                 base_titan_affinities: np.ndarray, mutation_rate: float, 
                 purity_factor: float, golden_one_cap: int, 
                 golden_one_reversion_prob: float, titan_to_path_model: TitanToPathModel,
                 max_affinity_norm: float, min_norm: float, max_norm: float):
        
        self.existing_names = existing_names
        self.name_to_entity_map = name_to_entity_map
        self.base_titan_affinities = base_titan_affinities
        self.mutation_rate = mutation_rate
        self.purity_factor = purity_factor
        self.golden_one_cap = golden_one_cap
        self.golden_one_reversion_prob = golden_one_reversion_prob
        self.titan_to_path_model = titan_to_path_model
        self.max_affinity_norm = max_affinity_norm
        self.min_norm = min_norm
        self.max_norm = max_norm
        self.cosmic_tide_vector = np.zeros(len(TITAN_NAMES)) 

    def generate_unique_name(self):
        while True:
            r1, r2 = random.sample(GREEK_ROOTS, 2)
            num = random.randint(1000, 9999)
            name = f"{r1}{r2}-{num}"
            if name not in self.existing_names:
                self.existing_names.add(name)
                return name

    def create_initial_population(self, num_entities: int, population: list, 
                                  cosmic_zeitgeist: np.ndarray):
        for _ in range(num_entities):
            name = self.generate_unique_name()
            entity = Pathstrider(name, self.base_titan_affinities + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape), titan_to_path_model=self.titan_to_path_model)
            self.recalculate_and_normalize_entity(entity, None, None) 
            population.append(entity)
            self.name_to_entity_map[name] = entity
        
    def recalculate_and_normalize_entity(self, entity: Pathstrider, path_distribution: np.ndarray, cosmic_zeitgeist: np.ndarray):
        entity.internal_purification(self.purity_factor)
        entity.titan_affinities += self.cosmic_tide_vector
        entity.titan_affinities = entity.titan_affinities.clip(min=0)
        self.normalize_affinities(entity)

    def normalize_affinities(self, entity_or_blueprint):
        target = entity_or_blueprint.titan_affinities if isinstance(entity_or_blueprint, Pathstrider) else entity_or_blueprint
        norm = np.linalg.norm(target)
        
        if not np.isfinite(norm) or norm == 0 or norm > self.max_affinity_norm:
            normalized_target = (target / (norm + 1e-9)) * self.max_affinity_norm
            if isinstance(entity_or_blueprint, Pathstrider):
                entity_or_blueprint.titan_affinities = normalized_target
            else: 
                entity_or_blueprint[:] = normalized_target
        return None 

    def get_global_path_distribution(self, population: list):
        if not population:
            return np.ones(len(PATH_NAMES)) / len(PATH_NAMES)
        path_counts = np.zeros(len(PATH_NAMES))
        for p in population:
            path_counts[p.dominant_path_idx] += 1
        distribution = path_counts / len(population)
        return distribution

    def update_golden_ones(self, population: list):
        """
        更新黄金裔：将上一代泰坦黄金裔清除，并选举12名新的天选黄金裔。
        """
        for p in population:
            if p.trait == "GoldenOne":
                p.trait = "Mortal"
                p.titan_aspect = None
                p.data_modification_unlocked = False

        # 选举新的黄金裔
        if not population: return
        candidates = [p for p in population if not p.is_titan_boss and p.trait != "Reincarnator"]
        candidates.sort(key=lambda p: p.score, reverse=True)

        num_titan_golden_ones = min(len(candidates), 12)
        titan_aspect_names = list(TITAN_NAMES) # 复制列表分配
        random.shuffle(titan_aspect_names)

        for i in range(num_titan_golden_ones):
            entity = candidates[i]
            entity.trait = "GoldenOne"
            entity.golden_one_tenure = 0
            entity.titan_aspect = titan_aspect_names.pop()

        # 分配普通黄金裔（如果上限更高）
        num_normal_golden_ones = min(len(candidates) - 12, self.golden_one_cap - 12)
        if num_normal_golden_ones > 0:
            for i in range(num_normal_golden_ones):
                entity = candidates[12 + i]
                entity.trait = "GoldenOne"
                entity.golden_one_tenure = 0

    def check_and_replenish_population(self, population: list, population_soft_cap: int, 
                                       aeonic_cycle_mode: bool, reincarnator: Pathstrider, 
                                       cosmic_zeitgeist: np.ndarray):
        if len(population) < population_soft_cap * 0.8:
            num_to_add = int(population_soft_cap - len(population))
            if num_to_add <= 0: return
            
            if aeonic_cycle_mode: 
                blueprint = self.base_titan_affinities * (1.5 * random.uniform(0.1, 1.5))
            else:
                golden_ones = [p for p in population if p.trait == "GoldenOne"]
                if not golden_ones: return
                template_entity = random.choice(golden_ones)
                blueprint = template_entity.titan_affinities
                
            self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist)

    def replenish_population_by_growth(self, population: list, num_to_add: int, cosmic_zeitgeist: np.ndarray):
        """Replenish based on the current blueprint from the guide network."""
        blueprint = self.base_titan_affinities
        self._add_new_entities(population, num_to_add, blueprint, cosmic_zeitgeist)

    def _add_new_entities(self, population: list, num_to_add: int, blueprint: np.ndarray, 
                          cosmic_zeitgeist: np.ndarray, legacy_modifier: np.ndarray = None): # 新增参数
        dist_newborns = self.get_global_path_distribution(population)
        for _ in range(num_to_add):
            name = self.generate_unique_name()
            new_affinities = blueprint + np.random.normal(0, self.mutation_rate, self.base_titan_affinities.shape)
            entity = Pathstrider(name, new_affinities, titan_to_path_model=self.titan_to_path_model)
            
            # 应用修正
            if legacy_modifier is not None:
                # 将命途修正值转换回对泰坦亲和度的影响
                # 这是通过反馈矩阵 (path_to_titan) 来实现的，虽然不完美...Sakaye不想改了
                titan_modifier = np.dot(legacy_modifier, self.titan_to_path_model.path_to_titan_feedback_matrix)
                entity.titan_affinities += titan_modifier * 50.0 # 50.0 是一个可调的魔法数字

            self.recalculate_and_normalize_entity(entity, dist_newborns, cosmic_zeitgeist)
            population.append(entity)
            self.name_to_entity_map[name] = entity
