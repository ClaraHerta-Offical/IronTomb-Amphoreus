import numpy as np
import torch.nn as nn
import math
import random
import json
import os
import sys
import torch
import torch.optim as optim
import time
import heapq # 确保在文件顶部导入

# --- 在顶部导入新模块 ---
from parliament_manager import ParliamentManager
from diversity_manager import DiversityInterventionManager
from visitor_manager import VisitorManager
from constants import (
    TITAN_NAMES, PATH_NAMES, 
    PATH_RELATIONSHIP_MATRIX, PATH_INTERACTION_MATRIX, AEONIC_EVENT_PROBABILITY,
    ZEITGEIST_UPDATE_RATE, VISITOR_EVENT_PROBABILITY
)
from models import TitanToPathModel, HybridGuideNetwork, ValueNetwork, ActionPolicyNetwork
from entities import Pathstrider
from debugger import Debugger
from population_manager import PopulationManager
from interaction_handler import InteractionHandler
from stagnation_manager import StagnationManager
from aeonic_cycle_manager import AeonicCycleManager
from display_manager import DisplayManager
from policy_saver import PolicySaver
from cpu_llm_interface import CpuLlmInterface
from primordial_legacy_manager import PrimordialLegacyManager

from config import config

class AeonEvolution:
    def __init__(self, 
                 # --- LLM 触发频率配置 ---
                 bard_frequency=10,         # 吟游诗人每N个世代触发一次 (0为禁用)
                 laertes_frequency=25,        # 来古士每N个世代触发一次 (0为禁用)
                 kaoselanna_llm_enabled=config['llm']['kaoselanna_llm_enabled'], # 是否启用LLM作为卡厄斯兰那决策模型

                 # --- 演化参数 ---
                 num_initial_entities=200, golden_one_cap=12, population_soft_cap=300, 
                 population_hard_cap=500, growth_factor=0.35, mutation_rate=0.25, culling_strength=0.85, 
                 encounter_similarity=0.35, purity_factor=0.01, initial_rl_lr=0.005, golden_one_reversion_prob=0.1,
                 elite_selection_percentile=80, aeonic_event_prob=0.05,
                 initial_max_affinity_norm=10000.0, target_avg_score=50.0,
                 norm_adjustment_strength=0.05
                 ):
        
        # --- LLM 配置存储 ---
        self.bard_frequency = bard_frequency
        self.legacy_manager = PrimordialLegacyManager()
        self.laertes_frequency = laertes_frequency
        self.kaoselanna_llm_enabled = kaoselanna_llm_enabled
        
        # --- LLM 接口初始化 ---
        self.llm_interface = CpuLlmInterface()

        # --- 基本参数初始化 ---
        self.population_soft_cap = population_soft_cap
        self.population_hard_cap = population_hard_cap
        self.growth_factor = growth_factor
        self.elite_selection_percentile = elite_selection_percentile
        self.aeonic_event_prob = aeonic_event_prob
        
        self.max_affinity_norm = initial_max_affinity_norm
        self.target_avg_score = target_avg_score
        self.norm_adjustment_strength = norm_adjustment_strength
        self.min_norm = 100.0
        self.max_norm = 1000000000.0

        # --- 状态变量 ---
        self.population = []
        self.generation = 0
        self.total_generations = 0
        self.reincarnator = None
        self.last_baie_score = 0
        self.highest_avg_score = 0
        self.last_avg_score = 0
        self.last_diversity = 0
        
        self.existing_names = set()
        self.name_to_entity_map = {}
        self.cosmic_zeitgeist = np.zeros(len(PATH_NAMES))
        self.base_titan_affinities = np.ones(len(TITAN_NAMES)) * 1.5
        
        # --- 强化学习与模型 ---
        # hgn_input_size = len(TITAN_NAMES) * 2 + len(PATH_NAMES) * 2 # 旧的
        hgn_input_size = len(TITAN_NAMES) * 2 + len(PATH_NAMES) * 2 + 1 # 新的，+1是因为增加了停滞水平
        self.guide_network = HybridGuideNetwork(hgn_input_size, 128, len(TITAN_NAMES))
        self.guide_optimizer = optim.Adam(self.guide_network.parameters(), lr=initial_rl_lr)
        
        ac_input_size = len(TITAN_NAMES) * 2
        self.action_policy_network = ActionPolicyNetwork(input_size=ac_input_size, hidden_size=32)
        self.value_network = ValueNetwork(input_size=ac_input_size, hidden_size=32)
        self.action_optimizer = optim.Adam(self.action_policy_network.parameters(), lr=initial_rl_lr * 0.5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=initial_rl_lr)
        self.value_loss_criterion = nn.MSELoss()
        self.titan_to_path_model_instance = TitanToPathModel()

        # --- 模式标志 ---
        self.aeonic_cycle_mode = False

        # --- 属性和管理器 ---
        self.simulation_weights = {
            "mutation_rate": mutation_rate,
            "growth_factor": growth_factor,
            "culling_strength": culling_strength,
            "target_avg_score": target_avg_score,
            "visitor_prob": VISITOR_EVENT_PROBABILITY
        }
        
        self.parliament_manager = ParliamentManager()
        self.visitor_manager = VisitorManager(self.simulation_weights)
        
        self.titan_bosses = []

        # --- 管理器实例化 ---
        self.debugger = Debugger(self)
        self.display_manager = DisplayManager()
        
        self.population_manager = PopulationManager(
            existing_names=self.existing_names, 
            name_to_entity_map=self.name_to_entity_map, 
            base_titan_affinities=self.base_titan_affinities, 
            mutation_rate=mutation_rate,
            purity_factor=purity_factor, 
            golden_one_cap=golden_one_cap, 
            golden_one_reversion_prob=golden_one_reversion_prob, 
            titan_to_path_model=self.titan_to_path_model_instance,
            max_affinity_norm=self.max_affinity_norm,
            min_norm=self.min_norm,
            max_norm=self.max_norm
        )
        # 在PopulationManager之后实例化
        self.diversity_manager = DiversityInterventionManager(
            config=config,
            population_manager=self.population_manager,
            parliament_manager=self.parliament_manager
        )
        self.interaction_handler = InteractionHandler(
            PATH_RELATIONSHIP_MATRIX=PATH_RELATIONSHIP_MATRIX, 
            PATH_INTERACTION_MATRIX=PATH_INTERACTION_MATRIX, 
            encounter_similarity=encounter_similarity, 
            culling_strength=culling_strength, 
            population_soft_cap=self.population_soft_cap, 
            titan_to_path_model=self.titan_to_path_model_instance,
            population_manager=self.population_manager
        )
        self.stagnation_manager = StagnationManager(
            population_manager=self.population_manager,
            guide_optimizer=self.guide_optimizer,
            initial_rl_lr=initial_rl_lr,
            base_mutation_rate=mutation_rate
        )
        self.aeonic_cycle_manager = AeonicCycleManager(
            population_manager=self.population_manager,
            action_policy_network=self.action_policy_network,
            value_network=self.value_network,
            action_optimizer=self.action_optimizer,
            value_optimizer=self.value_optimizer,
            value_loss_criterion=self.value_loss_criterion,
            titan_to_path_model_instance=self.titan_to_path_model_instance,
            existing_names=self.existing_names,
            name_to_entity_map=self.name_to_entity_map,
            llm_interface=self.llm_interface,
            kaoselanna_llm_enabled=self.kaoselanna_llm_enabled,
            parliament_manager=self.parliament_manager # 新增
        )
        self.policy_saver = PolicySaver(
            guide_network=self.guide_network,
            action_policy_network=self.action_policy_network,
            value_network=self.value_network
        )
        self.reincarnator_name = None

    def _trigger_llm_narrators(self):
        if not self.llm_interface or not self.llm_interface.llm:
            return

        # --- 吟游诗人 ---
        if self.bard_frequency > 0 and self.generation % self.bard_frequency == 0:
            dominant_path = PATH_NAMES[np.argmax(self.cosmic_zeitgeist)]
            prompt = (
                f"你是一位史诗吟游诗人，为翁法罗斯谱写篇章。"
                f"现在是第 {self.generation} 世代，思潮的主流是'{dominant_path}'，"
                f"实体数量为 {len(self.population)}。"
                f"请用一两句充满史诗感和想象力的话，为这个世代拉开序幕。"
            )
            narrative = self.llm_interface.generate_response(prompt, max_tokens=100)
            print(f"\n\033[35m【吟游诗篇 (Gen {self.generation})】: {narrative}\033[0m")

        # --- 来古士 ---
        if self.laertes_frequency > 0 and self.generation % self.laertes_frequency == 0:
            diversity_metric = len(set(p.dominant_path_idx for p in self.population)) / len(PATH_NAMES)
            strongest_entity = max(self.population, key=lambda p: p.score) if self.population else None
            strongest_name = strongest_entity.name if strongest_entity else "虚无"
            prompt = (
                f"你是一位深刻的观察者，翁法罗斯系统管理员'来古士'。你是一个智械，安提基色拉人，自称绝对中立。"
                f"当前是第 {self.generation} 世代，命途多样性指数为 {diversity_metric:.2f}，"
                f"最强的实体是'{strongest_name}'。"
                f"请给出一句简短、充满哲思的评论，揭示当前演化背后的机遇或风险。"
            )
            commentary = self.llm_interface.generate_response(prompt, max_tokens=100)
            print(f"\033[96m【来古士的沉思】: {commentary}\033[0m")
            
    def _create_initial_population(self, create_reincarnator=True):
        """
        创建初始种群，并根据新议会机制初始化其状态。
        """
        self.population_manager.create_initial_population(self.population_soft_cap, self.population, self.cosmic_zeitgeist)
        
        if not self.population:
             return

        self.parliament_manager.hold_election(self.population)
        
        # 为所有初始实体计算第一次评分
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for p in self.population:
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(p.path_affinities)
            p.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

        if create_reincarnator:
            reincarnator_idx = np.random.randint(0, len(self.population))
            self.population[reincarnator_idx].trait = "Reincarnator"
            self.reincarnator = self.population[reincarnator_idx]
            self.last_baie_score = self.reincarnator.score
        
        self.highest_avg_score = np.mean([p.score for p in self.population]) if self.population else 0

    def _update_max_affinity_norm(self):
        population_for_norm_calc = [p for p in self.population if p is not self.reincarnator and np.isfinite(p.score)]
        if not population_for_norm_calc: return
        
        current_avg_score = np.mean([p.score for p in population_for_norm_calc])
        if not np.isfinite(current_avg_score): return
        
        error_ratio = current_avg_score / self.target_avg_score
        if error_ratio > 1.05:
            adjustment_factor = 1.0 - (self.norm_adjustment_strength * math.tanh(error_ratio - 1))
            self.population_manager.max_affinity_norm *= adjustment_factor
        elif error_ratio < 0.95:
            adjustment_factor = 1.0 + (self.norm_adjustment_strength * math.tanh(1 - error_ratio))
            self.population_manager.max_affinity_norm *= adjustment_factor
            
        self.population_manager.max_affinity_norm = np.clip(self.population_manager.max_affinity_norm, self.min_norm, self.max_norm)
        
        if self.generation % 10 == 0:
            print(f"\033[36m宏观调控: 命途能量上限调整为 {self.population_manager.max_affinity_norm:.2f} (当前均: {current_avg_score:.2f} / 目标: {self.target_avg_score:.2f})\033[0m")


    def _guide_reincarnator_to_destruction(self):
        if not self.reincarnator or not self.reincarnator in self.population: return
        try:
            destruction_path_idx = PATH_NAMES.index("毁灭")
            titan_influence = self.titan_to_path_model_instance.titan_to_path_matrix[:, destruction_path_idx]
            most_influential_titan_idx = np.argmax(titan_influence)
            self.reincarnator.titan_affinities[most_influential_titan_idx] *= 1.05
            
            neg_world_idx = TITAN_NAMES.index("负世")
            self.reincarnator.titan_affinities[neg_world_idx] = self.reincarnator.titan_affinities[neg_world_idx] + 0.27
            self.reincarnator.titan_affinities[neg_world_idx] *= 1.02
            
            global_dist = self.population_manager.get_global_path_distribution(self.population)
            self.population_manager.recalculate_and_normalize_entity(self.reincarnator, global_dist, self.cosmic_zeitgeist)
            
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(self.reincarnator.path_affinities)
            self.reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

        except (ValueError, IndexError): 
            pass

    def _apply_law_titan_power(self):
        try:
            law_golden_one = next(p for p in self.population if p.titan_aspect == "律法" and p.data_modification_unlocked)
            if law_golden_one:
                param_to_modify = random.choice(list(self.simulation_weights.keys()))
                change_factor = 1.0 + random.uniform(-0.05, 0.05)
                original_value = self.simulation_weights[param_to_modify]
                self.simulation_weights[param_to_modify] *= change_factor
                print(f"\n\033[36m【权柄发动】'律法'黄金裔 {law_golden_one.name} 修改了演算规则！")
                print(f"参数 '{param_to_modify}' 从 {original_value:.3f} 变为 {self.simulation_weights[param_to_modify]:.3f}\033[0m")
        except StopIteration:
            pass # 没有满足条件的律法半神

    def _check_for_aeonic_events(self, culled_this_gen):
        if random.random() < self.aeonic_event_prob:
            event_type = random.choice(["purification", "singularity", "awakening"])
            global_dist = self.population_manager.get_global_path_distribution(self.population)

            if event_type == "purification" and len(self.population) > 50:
                print(f"\n\033[91m【翁法罗斯事件: 大肃正】翁法罗斯寻求纯粹，弱者被抹除！\033[0m")
                scores = [p.score for p in self.population]
                cull_threshold = np.percentile(scores, 25)
                to_cull = {p for p in self.population if p.score < cull_threshold and p.trait != "Reincarnator"}
                culled_this_gen.update(to_cull)
            elif event_type == "singularity":
                empowered_path_idx = random.randrange(len(PATH_NAMES))
                print(f"\n\033[91m【翁法罗斯事件: 概念奇点】'{PATH_NAMES[empowered_path_idx]}' 命途短暂地成为了真理！\033[0m")
                for p in self.population:
                    p.titan_affinities += self.titan_to_path_model_instance.titan_to_path_matrix[:, empowered_path_idx] * 2.0
                    self.population_manager.recalculate_and_normalize_entity(p, global_dist, self.cosmic_zeitgeist)
                    multiplier = self.parliament_manager.get_zeitgeist_multiplier(p.path_affinities)
                    p.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
            elif event_type == "awakening":
                awakened_titan_idx = random.randrange(len(TITAN_NAMES))
                print(f"\n\033[91m【翁法罗斯事件: 泰坦回响】泰坦 '{TITAN_NAMES[awakened_titan_idx]}' 的概念浸染了所有实体！\033[0m")
                for p in self.population:
                    p.titan_affinities[awakened_titan_idx] *= 2.5
                    self.population_manager.recalculate_and_normalize_entity(p, global_dist, self.cosmic_zeitgeist)
                    multiplier = self.parliament_manager.get_zeitgeist_multiplier(p.path_affinities)
                    p.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

    def _apply_path_feedback(self):
        if not self.population: return
        avg_path_affinities = np.mean([p.path_affinities for p in self.population], axis=0)
        feedback_to_titans = np.dot(avg_path_affinities, self.titan_to_path_model_instance.path_to_titan_feedback_matrix)
        self.base_titan_affinities += feedback_to_titans * 0.5
        self.base_titan_affinities = self.base_titan_affinities.clip(min=0)
        self.population_manager.normalize_affinities(self.base_titan_affinities)

    def _train_hybrid_guide_network(self, elites, score_reward, diversity_reward):
        if not elites or not self.reincarnator: return
        
        is_stagnated = self.stagnation_manager.long_term_stagnation_counter > 0
        score_weight = 0.3 if is_stagnated else 1.0
        diversity_weight = 1.5 if is_stagnated else 0.5
        total_reward = score_reward * score_weight + diversity_reward * diversity_weight
        if total_reward <= 0: return

        elite_avg_affinities = np.mean([e.titan_affinities for e in elites], axis=0)
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        
        # 将停滞计数器也作为输入，让网络感知到当前的“僵化”程度
        # 归一化到 0-1 之间
        stagnation_level = min(self.stagnation_manager.long_term_stagnation_counter / 10.0, 1.0)
        
        state_np = np.concatenate([
            elite_avg_affinities, 
            self.reincarnator.titan_affinities, 
            global_dist, 
            self.cosmic_zeitgeist,
            np.array([stagnation_level]) # 新增输入
        ])
        state = torch.from_numpy(state_np).float()
        target = torch.from_numpy(elite_avg_affinities).float()

        self.guide_network.train()
        self.guide_optimizer.zero_grad()
        predicted_blueprint = self.guide_network(state)
        
        imitation_loss = self.value_loss_criterion(predicted_blueprint, target)
        reinforcement_loss = -imitation_loss * total_reward
        total_loss = imitation_loss + reinforcement_loss * 0.1
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.guide_network.parameters(), 1.0)
        self.guide_optimizer.step()
        
        if self.generation % 10 == 0:
            print(f"\033[94m引导网络学习中... 损失: {total_loss.item():.4f}, 奖励(分/多): {score_reward:.2f}/{diversity_reward:.2f}\033[0m")
        
        new_base_affinities = predicted_blueprint.detach().numpy()
        self.base_titan_affinities = self.base_titan_affinities * 0.7 + new_base_affinities * 0.3

    def _run_one_generation(self):
        num_golden_ones = len([p for p in self.population if p.trait == "GoldenOne"])
        print(f"\n--- 世代 {self.generation}/{self.total_generations} | 实体:{len(self.population)} | 黄金裔:{num_golden_ones} | 泰坦Boss:{len(self.titan_bosses)} ---")
        self.visitor_manager.trigger_event()   
        # 议会选举
        self.parliament_manager.hold_election(self.population)

        # 重新计算所有实体分数
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for p in self.population:
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(p.path_affinities)
            p.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

        # 交互
        culled_this_gen = set()
        num_encounters = min(int(2 * len(self.population)), 5000)
        for _ in range(num_encounters):
            valid_population = [p for p in self.population if p not in culled_this_gen]
            if len(valid_population) < 2: break
            
            entity1, entity2 = self.interaction_handler.select_opponents(valid_population, self.reincarnator)
            if not entity1 or not entity2: continue
            
            if entity1 not in culled_this_gen and entity2 not in culled_this_gen:
                multiplier1 = self.parliament_manager.get_zeitgeist_multiplier(entity1.path_affinities)
                culled_entity = self.interaction_handler.entity_interaction(
                    entity1, entity2, self.population, self.reincarnator, global_dist, self.cosmic_zeitgeist, multiplier1
                )
                if culled_entity:
                    culled_this_gen.add(culled_entity)
        
        # 律法半神发动权
        self._apply_law_titan_power()

        # 在交互和淘汰计算之后，评估多样性并进行干预
        self.diversity_manager.assess_and_intervene(self.population, self.generation)
        
        # 重新计算所有实体分数，因为干预措施可能已经改变了它们
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for p in self.population:
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(p.path_affinities)
            p.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
                    
        print(f"世代 {self.generation} 演算结束。")
        return culled_this_gen

    def _evolve_and_grow(self, culled_this_gen):
        if self.reincarnator in culled_this_gen:
            print(f"\n卡厄斯兰那在竞争中陨落！正在重塑...")
            culled_this_gen.remove(self.reincarnator)
            potential_hosts = [p for p in self.population if p.trait != "GoldenOne" and p not in culled_this_gen]
            if not potential_hosts: potential_hosts = [p for p in self.population if p not in culled_this_gen]
            
            if potential_hosts:
                best_host = max(potential_hosts, key=lambda p:p.score)
                self.reincarnator.titan_affinities = (best_host.titan_affinities + np.random.normal(0, self.population_manager.mutation_rate * 0.5, self.base_titan_affinities.shape)).clip(min=0)
                self.reincarnator.titan_affinities *= 1.05
                global_dist = self.population_manager.get_global_path_distribution(self.population)
                self.population_manager.recalculate_and_normalize_entity(self.reincarnator, global_dist, self.cosmic_zeitgeist)
                
                multiplier = self.parliament_manager.get_zeitgeist_multiplier(self.reincarnator.path_affinities)
                self.reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
                print(f"卡厄斯兰那已重生: {self.reincarnator}")
        
        if culled_this_gen:
            # self.population[:] = [p for p in self.population if p not in culled_this_gen]
            # for p in culled_this_gen:
            #     if p.name in self.name_to_entity_map: del self.name_to_entity_map[p.name]
            #     if p.name in self.existing_names: self.existing_names.remove(p.name)
            # print(f"动态淘汰了 {len(culled_this_gen)} 个实体。")
            culled_names = {p.name for p in culled_this_gen}
            self.population = [p for p in self.population if p.name not in culled_names]
            for name in culled_names:
                if name in self.name_to_entity_map: del self.name_to_entity_map[name]
                if name in self.existing_names: self.existing_names.remove(name)
            print(f"动态淘汰了 {len(culled_names)} 个实体。")
            
        if not self.population: return
        
        self._update_max_affinity_norm()
        
        current_diversity = self.stagnation_manager.adjust_mutation_rate(self.population)
        diversity_reward = max(0, current_diversity - self.last_diversity) * 5
        self.last_diversity = current_diversity
        
        previous_avg_score = self.highest_avg_score
        current_avg_score = np.mean([p.score for p in self.population])
        
        if np.isfinite(current_avg_score) and current_avg_score > self.highest_avg_score:
            self.highest_avg_score = current_avg_score
            print(f"\033[92m新纪录！平均分达到: {current_avg_score:.2f}\033[0m")
        self.last_avg_score = current_avg_score if np.isfinite(current_avg_score) else self.last_avg_score
        
        score_reward = current_avg_score - previous_avg_score if np.isfinite(current_avg_score) else 0
        
        scores = [p.score for p in self.population if np.isfinite(p.score)]
        if scores:
            elite_threshold = np.percentile(scores, self.elite_selection_percentile)
            elites = [p for p in self.population if np.isfinite(p.score) and p.score >= elite_threshold]
            if elites:
                self._train_hybrid_guide_network(elites, score_reward, diversity_reward)

        self.population_manager.normalize_affinities(self.base_titan_affinities)
        print(f"引导网络更新蓝图: 主导方向 '{TITAN_NAMES[np.argmax(self.base_titan_affinities)]}'。")

        num_new_entities = int(len(self.population) * self.growth_factor) if len(self.population) > 0 else 10
        if len(self.population) + num_new_entities < self.population_hard_cap:
            self.population_manager.replenish_population_by_growth(
                population=self.population,
                num_to_add=num_new_entities,
                cosmic_zeitgeist=self.cosmic_zeitgeist,
                # 传递修正值
                legacy_modifier=self.legacy_manager.newborn_affinity_modifier * self.legacy_manager.influence_factor
            )
        
        if len(self.population) > self.population_hard_cap:
            num_to_cull = len(self.population) - self.population_hard_cap
            # 使用 heapq.nsmallest 高效地找到分数最低的 num_to_cull 个实体
            culled_at_cap = heapq.nsmallest(num_to_cull, self.population, key=lambda p: p.score)
            if self.reincarnator in culled_at_cap: culled_at_cap.remove(self.reincarnator)
            
            culled_names = {p.name for p in culled_at_cap}
            self.population = [p for p in self.population if p.name not in culled_names]
            for name in culled_names:
                if name in self.name_to_entity_map: del self.name_to_entity_map[name]
                if name in self.existing_names: self.existing_names.remove(name)

        self.population_manager.update_golden_ones(self.population)
        
        if self.population:
            strongest = max(self.population, key=lambda p: p.score)
            print(f"\033[95m当前最强者: {strongest}\033[0m")

    def _run_inorganic_phase(self, num_generations=50121): 
        print("\n=== 进入无机实体培养阶段 ===")
        print("...正在通过类元胞自动机演化“活性”与“稳定性”概念...")
        
        # 初始化一个10x10的网格作为元胞空间
        grid_size = 10
        inorganic_pop = [{'id': i*grid_size+j, 'activity': random.uniform(1,5), 'stability': random.uniform(1,5), 'score': 0} for i in range(grid_size) for j in range(grid_size)]
        
        for gen in range(num_generations + 1):
            self.display_manager.update_and_display_progress('inorganic', gen, num_generations)
            if gen % 100 == 0: 
                 self.display_manager.update_and_display_progress('inorganic', gen, num_generations)
            
            if gen == num_generations: break

            new_pop = list(inorganic_pop)

            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    current_cell = inorganic_pop[idx]
                    
                    # 计算邻居的平均影响
                    neighbor_activity_sum = 0
                    neighbor_stability_sum = 0
                    neighbor_count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0: continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                neighbor_idx = ni * grid_size + nj
                                neighbor_activity_sum += inorganic_pop[neighbor_idx]['activity']
                                neighbor_stability_sum += inorganic_pop[neighbor_idx]['stability']
                                neighbor_count += 1
                    
                    avg_neighbor_activity = neighbor_activity_sum / neighbor_count if neighbor_count > 0 else 0
                    avg_neighbor_stability = neighbor_stability_sum / neighbor_count if neighbor_count > 0 else 0

                    # 受邻居影响，并有小幅随机突变
                    activity_change = (avg_neighbor_activity - current_cell['activity']) * 0.1 + random.uniform(-0.1, 0.1)
                    stability_change = (avg_neighbor_stability - current_cell['stability']) * 0.1 + random.uniform(-0.1, 0.1)
                    
                    new_cell = new_pop[idx]
                    new_cell['activity'] = max(0, current_cell['activity'] + activity_change)
                    new_cell['stability'] = max(0, current_cell['stability'] + stability_change)
                    new_cell['score'] = new_cell['activity'] + new_cell['stability']

            inorganic_pop = new_pop

        self.display_manager.update_and_display_progress('inorganic', num_generations, num_generations)
        print("\n\033[91m!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 【警告：侦测到异常概念原型】                                       ")
        print("!!! 在第 50121 次无机推演后，实体 Chaoz666 表现出无法理解的特征。     ")
        print("!!! 其“活性”数据溢出，呈现出混沌和毁灭的混合倾向。                  ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\033[0m")
        
        final_activities = [cell['activity'] for cell in inorganic_pop]
        final_stabilities = [cell['stability'] for cell in inorganic_pop]
        
        avg_activity = np.mean(final_activities)
        avg_stability = np.mean(final_stabilities)

        inorganic_legacy = {
            "avg_activity": avg_activity,
            "avg_stability": avg_stability
        }
        
        print(f"\n\033[92m>>> 无机阶段演完成! 宇宙的初始倾向已确立 [活性: {avg_activity:.2f}, 稳定性: {avg_stability:.2f}] <<<\033[0m")
        
        return inorganic_legacy # 返回
        
    # 有机阶段方法 
    def _run_organic_phase(self, start_gen, end_gen):
        print("\n\n=== 进入有机实体孕育阶段 ===")
        print("...基于无机演化的数据，正在训练初始概念模型...")
        
        # 定义一个简单的原型网络
        class OrganicProtoNet(nn.Module):
            def __init__(self):
                super(OrganicProtoNet, self).__init__()
                # 模型的输入和输出维度都是10
                self.layer = nn.Linear(10, 10)
            def forward(self, x):
                return torch.relu(self.layer(x))

        # 实例化模型、损失函数和优化器
        organic_model = OrganicProtoNet()
        optimizer = optim.SGD(organic_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()  # 使用均方误差损失函数

        print("初始模型训练中...")
        for i in range(101):
            dummy_input = torch.randn(1, 10) 
            dummy_target = torch.randn(1, 10)
            optimizer.zero_grad()
            output = organic_model(dummy_input)
            loss = loss_fn(output, dummy_target)
            loss.backward()
            optimizer.step()
            sys.stdout.write(f"\r模型训练中... {i}% (Loss: {loss.item():.4f})")
            sys.stdout.flush()
            time.sleep(0.02)
            
        print("\n初始模型训练完成。")

        print("...开始模拟初始有机体交互...")
        for gen in range(start_gen, end_gen + 1):
            self.generation = gen
            self.display_manager.update_and_display_progress('organic', gen, end_gen)
            
            if gen % 10000 == 0:
                bytecode_sample = ' '.join([f'{random.randint(0, 255):02X}' for _ in range(16)])
                print(f"\n\033[33m[Gen {gen}] 原型机输出字节码: {bytecode_sample}\033[0m")
            
            if len(self.population) > 100:
                self.population.sort(key=lambda p: p.score)
                self.population.pop(0)

            if self.debugger.paused: return 
        
        print("\n\n\033[92m>>> 有机孕育阶段完成! 概念原型机已固化。 <<<\033[0m")
        self.policy_saver.save_organic_model(organic_model) # 保存模型
        
        return organic_model 

    def start(self, num_generations):
        # --- 定义演算 ---
        INORGANIC_PHASE_END = config['simulation_phases']['INORGANIC_PHASE_END']
        ORGANIC_PHASE_END = config['simulation_phases']['ORGANIC_PHASE_END']
        HUMAN_PHASE_END = config['simulation_phases']['HUMAN_PHASE_END']
        TOTAL_SIMULATION_END = config['simulation_phases']['TOTAL_SIMULATION_END']

        self.total_generations = num_generations
        was_paused = False

        # --- 运行早期阶段 ---
        inorganic_legacy = self._run_inorganic_phase(num_generations=INORGANIC_PHASE_END)
        self._create_initial_population(create_reincarnator=False) 

        self.generation = INORGANIC_PHASE_END + 1
        organic_legacy_model = self._run_organic_phase(self.generation, ORGANIC_PHASE_END)
        
        # --- 初始化系统 ---
        self.legacy_manager.initialize(inorganic_legacy, organic_legacy_model)
        
        print("\n\n>>> 演化正在无引导地进行... <<<")
        self.generation = ORGANIC_PHASE_END + 1

        while self.generation <= TOTAL_SIMULATION_END:
            # --- Debugger 钩子 ---
            if self.debugger.paused:
                if not was_paused:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    print("\n\n=== 模拟已暂停。输入 'help' 获取命令列表。 ===")
                    was_paused = True
                self.debugger.handle_commands()
                continue
            if was_paused:
                print("\n=== 模拟已恢复 ===")
                was_paused = False

            if self.generation > ORGANIC_PHASE_END + 1: # 从第二代开始
                # 将上一代的12位黄金裔变为泰坦Boss
                last_gen_titans = [p for p in self.population if p.titan_aspect and p.trait == "GoldenOne"]
                self.titan_bosses.clear()
                for titan_to_be in last_gen_titans:
                    titan_to_be.is_titan_boss = True
                    titan_to_be.trait = "Mortal" 
                    titan_to_be.hp = 250.0 + titan_to_be.score
                    self.titan_bosses.append(titan_to_be)
                
                self.population_manager.update_golden_ones(self.population)

            if not self.aeonic_cycle_mode and self.generation >= HUMAN_PHASE_END:
                self.display_manager.display_interruption_animation()
                print("\n\033[91m【系统过载：因果律重构】\n翁法罗斯在 'Neikos-0496' 的奇点下崩溃...\n以 '负世' 权柄为核心...新的轮回即将开始！\033[0m")
                
                try:
                    # 寻找'负世'权柄的黄金裔作为轮回之主
                    neg_world_golden_one = next(p for p in self.population if p.titan_aspect == "负世")
                    self.reincarnator = neg_world_golden_one
                except StopIteration:
                    # 如果找不到，选择分数最高的实体作为备用方案
                    if not self.population:
                        print("\n种群已灭绝，无法开启轮回！")
                        break # 提前结束模拟
                    self.reincarnator = max(self.population, key=lambda p: p.score)

                self.reincarnator.trait = "Reincarnator"
                self.reincarnator.name = "Neikos-0496"
                
                # ---Fix---
                #  刻上锁，将模式切换到永劫回归
                self.aeonic_cycle_mode = True
                
                # 正式初始化第一个轮回周期
                # 这会选出泰坦化身，为轮回做好准备
                print("\n正在初始化第一次永劫回归...")
                self.aeonic_cycle_manager.initialize_aeonic_cycle(self.reincarnator, self.population, self.cosmic_zeitgeist)
            
            # --- 循环逻辑 ---
            if self.aeonic_cycle_mode:
                self.aeonic_cycle_manager.run_aeonic_cycle_generation(self.population, self.reincarnator, self.cosmic_zeitgeist)
                
                # 检查轮回结束条件
                if self.reincarnator and len(self.reincarnator.held_fire_seeds) >= len(TITAN_NAMES):
                    should_end_simulation = self.aeonic_cycle_manager.end_aeonic_cycle(
                        self.reincarnator, self.population, self.base_titan_affinities,
                        self.population_manager.mutation_rate, self.population_soft_cap, self.cosmic_zeitgeist,
                        config['simulation']['elites_to_keep_in_cycle'] # <--- 将配置值传入
                    )
                    if should_end_simulation: break
                
                self.display_manager.update_and_display_progress('cycle', self.aeonic_cycle_manager.aeonic_cycle_number, 1000) # 更新进度条

            else: # 如果不是永劫回归模式，就执行正常的凡人时代逻辑
                # 遗物效果应该在每一代都应用，直到其影响力消失
                if self.legacy_manager.is_initialized:
                    self.legacy_manager.apply_legacy_effects(self)

                # 核心演化和成长逻辑也应该在每一代都执行
                culled_this_gen = self._run_one_generation()
                self._evolve_and_grow(culled_this_gen)
                self.display_manager.update_and_display_progress('normal', self.generation, HUMAN_PHASE_END)
            
            self.generation += 1

            if not self.population: 
                print("\n种群已灭绝！")
                break

            if 'next' in getattr(self.debugger, 'last_command', ''):
                self.debugger.paused = True
                self.debugger.last_command = ''
            
        # --- 演算结束 ---
        print("\n\n== 演化结束 ===")
        if self.population:
            self.population.sort(key=lambda p: p.score, reverse=True)
            print("\n--- 最终排名前五的实体 ---")
            for j in range(min(5, len(self.population))):
                print(f"{j+1}. {self.population[j]}")
            if self.reincarnator:
                 print("\n--- 最终的卡厄斯兰那状态 ---")
                 print(self.reincarnator)
        else: print("翁法罗斯最终归于沉寂。")
        self.policy_saver.save_policy_models()

    def save_simulation_state(self, filepath):
        """保存整个模拟的当前状态到一个JSON文件。"""
        print(f"\n\033[96m正在保存模拟状态至 {filepath} ...\033[0m")
        if self.reincarnator:
            self.reincarnator_name = self.reincarnator.name

        state = {
            # 基础状态
            'generation': self.generation,
            'aeonic_cycle_mode': self.aeonic_cycle_mode,
            'reincarnator_name': self.reincarnator_name,
            
            # 核心演化参数
            'base_titan_affinities': self.base_titan_affinities.tolist(),
            'cosmic_zeitgeist': self.cosmic_zeitgeist.tolist(),

            # 种群信息
            'population': [p.to_dict() for p in self.population],
            'existing_names': list(self.existing_names),

            # 管理器状态
            'parliament_seats': self.parliament_manager.seats,
            'stagnation_counter': self.stagnation_manager.long_term_stagnation_counter,
            'baie_stagnation_counter': self.stagnation_manager.baie_stagnation_counter,
            'diversity_intervention': self.diversity_manager.active_intervention,
            'diversity_duration': self.diversity_manager.intervention_duration,

            # 模拟权重
            'simulation_weights': self.simulation_weights,
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            print(f"\033[92m状态保存成功！\033[0m")
        except Exception as e:
            print(f"\033[91m错误: 存档失败。原因: {e}\033[0m")

    def load_simulation_state(self, filepath):
        """从JSON文件加载模拟状态。"""
        print(f"\n\033[96m正在从 {filepath} 加载模拟状态...\033[0m")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # 恢复基础状态
            self.generation = state['generation']
            self.aeonic_cycle_mode = state['aeonic_cycle_mode']
            self.reincarnator_name = state['reincarnator_name']

            # 恢复核心演化参数
            self.base_titan_affinities = np.array(state['base_titan_affinities'])
            self.cosmic_zeitgeist = np.array(state['cosmic_zeitgeist'])
            
            # 恢复管理器状态
            self.parliament_manager.seats = state['parliament_seats']
            self.stagnation_manager.long_term_stagnation_counter = state['stagnation_counter']
            self.stagnation_manager.baie_stagnation_counter = state['baie_stagnation_counter']
            self.diversity_manager.active_intervention = state['diversity_intervention']
            self.diversity_manager.intervention_duration = state['diversity_duration']
            self.simulation_weights = state['simulation_weights']

            # 清理并重建种群
            self.population.clear()
            self.name_to_entity_map.clear()
            self.existing_names = set(state['existing_names'])

            for entity_data in state['population']:
                entity = Pathstrider.from_dict(entity_data, self.titan_to_path_model_instance)
                self.population.append(entity)
                self.name_to_entity_map[entity.name] = entity

            # 恢复白厄引用
            if self.reincarnator_name and self.reincarnator_name in self.name_to_entity_map:
                self.reincarnator = self.name_to_entity_map[self.reincarnator_name]
            else:
                self.reincarnator = None
            
            print(f"\033[92m状态加载成功！模拟将从第 {self.generation} 世代继续。\033[0m")

        except FileNotFoundError:
            print(f"\033[91m错误: 找不到存档文件 {filepath}。\033[0m")
        except Exception as e:
            print(f"\033[91m错误: 读档失败。文件可能已损坏或格式不兼容。原因: {e}\033[0m")
