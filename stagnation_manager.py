import numpy as np
import random
import math
import torch.optim as optim

from constants import PATH_NAMES, TITAN_NAMES, LONG_TERM_STAGNATION_THRESHOLD, BAIE_STAGNATION_THRESHOLD
from entities import Pathstrider
from population_manager import PopulationManager

class StagnationManager:
    def __init__(self, population_manager: PopulationManager, 
                 guide_optimizer: optim.Adam, 
                 initial_rl_lr: float,
                 base_mutation_rate: float):
        
        self.population_manager = population_manager
        self.guide_optimizer = guide_optimizer
        self.initial_rl_lr = initial_rl_lr
        self.base_mutation_rate = base_mutation_rate
        self.mutation_rate = base_mutation_rate 

        self.long_term_stagnation_counter = 0
        self.score_history = [] 
        self.baie_stagnation_counter = 0

        self.cosmic_tide_vector = np.zeros(len(TITAN_NAMES))
        self.tide_duration = 0
        self.tide_counter = 0

    def check_stagnation_and_intervene(self, population: list, reincarnator: Pathstrider, last_avg_score: float, last_baie_score: float, cosmic_zeitgeist: np.ndarray):
        if np.isfinite(last_avg_score):
            self.score_history.append(last_avg_score)
            if len(self.score_history) > LONG_TERM_STAGNATION_THRESHOLD:
                self.score_history.pop(0)
            
            if len(self.score_history) == LONG_TERM_STAGNATION_THRESHOLD:
                window = np.array(self.score_history)
                if (np.max(window) - np.min(window)) < (np.max(window) * 0.01):
                    self.long_term_stagnation_counter += 1
                else:
                    self.long_term_stagnation_counter = 0 
            
            if self.long_term_stagnation_counter >= 10: 
                self._trigger_conceptual_awakening(population)
                self.long_term_stagnation_counter = 0 
                self.score_history = [] 
        
        if reincarnator:
            if reincarnator.score < last_baie_score * 1.01:
                self.baie_stagnation_counter += 1
            else:
                self.baie_stagnation_counter = 0 

            if self.baie_stagnation_counter >= BAIE_STAGNATION_THRESHOLD:
                self._trigger_revelation_boost(reincarnator, cosmic_zeitgeist)
                self.baie_stagnation_counter = 0 

    def _trigger_conceptual_awakening(self, population: list):
        print(f"\n\033[91m【全局停滞】 来古士: 翁法罗斯的演化已停滞 {LONG_TERM_STAGNATION_THRESHOLD} 世代！必须引入新的变量！\033[0m")
        if not population: return
        
        total_affinities = np.sum([p.titan_affinities for p in population], axis=0)
        weakest_titan_idx = np.argmin(total_affinities)
        weakest_titan_name = TITAN_NAMES[weakest_titan_idx]

        self.tide_duration = 50 
        self.tide_counter = self.tide_duration
        tide_strength = random.uniform(0.8, 1.5) 
        self.cosmic_tide_vector[:] = 0
        self.cosmic_tide_vector[weakest_titan_idx] = tide_strength
        
        print(f"\n\033[35m来古士引入了新变量: 在未来 {self.tide_duration} 世代，'{weakest_titan_name}' 的影响被史诗级增强了。\033[0m")
        print("\033[93m...学习策略已调整: 提高探索性，寻求突破。\033[0m")
        for g in self.guide_optimizer.param_groups:
            g['lr'] *= 2.0 

    def _trigger_revelation_boost(self, reincarnator: Pathstrider, cosmic_zeitgeist: np.ndarray):
        if not reincarnator: return
        print(f"\n\033[96m事件: 卡厄斯兰那 ({reincarnator.name}) 的成长已停滞 {BAIE_STAGNATION_THRESHOLD} 世代，命运的齿轮开始转动...\033[0m")
        
        non_zero_affinities = np.where(reincarnator.titan_affinities > 0, reincarnator.titan_affinities, np.inf)
        weakest_link_idx = np.argmin(non_zero_affinities)
        
        reincarnator.titan_affinities[weakest_link_idx] *= 2.5
        reincarnator.titan_affinities[weakest_link_idx] += 1.0 
        
        global_dist = self.population_manager.get_global_path_distribution([]) 
        self.population_manager.recalculate_and_normalize_entity(reincarnator, None, cosmic_zeitgeist)

    def adjust_mutation_rate(self, population: list):
        if not population: 
            self.mutation_rate = self.base_mutation_rate
            return 0
            
        num_dominant_paths = len(set(p.dominant_path_idx for p in population))
        diversity_metric = num_dominant_paths / len(PATH_NAMES)
        
        if diversity_metric < 0.25:
            self.mutation_rate = self.base_mutation_rate * 2.5
            print(f"\033[33m可观测样本过低({diversity_metric:.2f})！异常变量已临时提高至 {self.mutation_rate:.3f}\033[0m")
        else:
            self.mutation_rate = self.base_mutation_rate
            
        self.population_manager.mutation_rate = self.mutation_rate
        return diversity_metric

    def update_cosmic_tide(self):
        self.tide_counter -= 1
        if self.tide_counter <= 0:
            self.tide_duration = random.randint(5, 15)
            self.tide_counter = self.tide_duration
            tide_strength = random.uniform(0.1, 0.5)
            num_favored_titans = random.randint(1, 3)
            favored_indices = np.random.choice(len(TITAN_NAMES), num_favored_titans, replace=False)
            
            self.cosmic_tide_vector[:] = 0 
            self.cosmic_tide_vector[favored_indices] = tide_strength
            
            favored_names = ", ".join([TITAN_NAMES[i] for i in favored_indices])
            print(f"\n\033[35m来古士引入了新变量: 在未来 {self.tide_duration} 世代，'{favored_names}' 的影响增强了。\033[0m")
        else: 
            self.cosmic_tide_vector *= 0.95
        
        # Update the manager
        self.population_manager.cosmic_tide_vector = self.cosmic_tide_vector