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

# --- 本地模块导入 ---
from constants import (
    TITAN_NAMES, PATH_NAMES, 
    PATH_RELATIONSHIP_MATRIX, PATH_INTERACTION_MATRIX, AEONIC_EVENT_PROBABILITY,
    ZEITGEIST_UPDATE_RATE
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
        hgn_input_size = len(TITAN_NAMES) * 2 + len(PATH_NAMES) * 2 
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
            kaoselanna_llm_enabled=self.kaoselanna_llm_enabled
        )
        self.policy_saver = PolicySaver(
            guide_network=self.guide_network,
            action_policy_network=self.action_policy_network,
            value_network=self.value_network
        )

    def _trigger_llm_narrators(self):
        if not self.llm_interface or not self.llm_interface.llm:
            return

        # --- 吟游诗人  ---
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

        # --- 来古士 (Laertes) ---
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
            
    def _create_initial_population(self):
        self.population_manager.create_initial_population(self.population_soft_cap, self.population, self.cosmic_zeitgeist)
        
        self._update_cosmic_zeitgeist()
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for p in self.population:
            p.recalculate_concepts(global_dist, self.cosmic_zeitgeist)

        if not self.population: return

        reincarnator_idx = np.random.randint(0, len(self.population))
        self.population[reincarnator_idx].trait = "Reincarnator"
        self.reincarnator = self.population[reincarnator_idx]
        self.highest_avg_score = np.mean([p.score for p in self.population]) if self.population else 0
        self.last_baie_score = self.reincarnator.score

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

    def _update_cosmic_zeitgeist(self):
        if not self.population: return
        total_weighted_votes = np.zeros(len(PATH_NAMES))
        total_weight = 0
        for entity in self.population:
            vote_proposal = entity.generate_vote_proposal()
            vote_weight = entity.get_vote_weight() * entity.score
            total_weighted_votes += vote_proposal * vote_weight
            total_weight += vote_weight
        
        if total_weight == 0: return
        current_vote_result = total_weighted_votes / total_weight
        self.cosmic_zeitgeist = (self.cosmic_zeitgeist * (1 - ZEITGEIST_UPDATE_RATE) + current_vote_result * ZEITGEIST_UPDATE_RATE)
        
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for p in self.population:
            p.recalculate_concepts(global_dist, self.cosmic_zeitgeist)
            
        if self.generation % 10 == 0:
            dominant_zeitgeist_idx = np.argmax(self.cosmic_zeitgeist)
            dominant_zeitgeist_name = PATH_NAMES[dominant_zeitgeist_idx]
            print(f"\033[32m思潮更新: 当前时代的主流是 '{dominant_zeitgeist_name}' (权重: {self.cosmic_zeitgeist[dominant_zeitgeist_idx]:.3f})\033[0m")

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
        except (ValueError, IndexError): 
            pass

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
            elif event_type == "awakening":
                awakened_titan_idx = random.randrange(len(TITAN_NAMES))
                print(f"\n\033[91m【翁法罗斯事件: 泰坦回响】泰坦 '{TITAN_NAMES[awakened_titan_idx]}' 的概念浸染了所有实体！\033[0m")
                for p in self.population:
                    p.titan_affinities[awakened_titan_idx] *= 2.5
                    self.population_manager.recalculate_and_normalize_entity(p, global_dist, self.cosmic_zeitgeist)

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
        
        state_np = np.concatenate([elite_avg_affinities, self.reincarnator.titan_affinities, global_dist, self.cosmic_zeitgeist])
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
        print(f"\n--- 世代 {self.generation}/{self.total_generations} | 实体数目:{len(self.population)} | 黄金裔:{num_golden_ones} ---")
        
        self._trigger_llm_narrators()
        
        self.stagnation_manager.update_cosmic_tide()
        self._update_cosmic_zeitgeist() 
        self._apply_path_feedback()
        self._guide_reincarnator_to_destruction() 
        
        culled_this_gen = set()
        self._check_for_aeonic_events(culled_this_gen)
        
        self.stagnation_manager.check_stagnation_and_intervene(
            population=self.population,
            reincarnator=self.reincarnator, 
            last_avg_score=self.last_avg_score,
            last_baie_score=self.last_baie_score,
            cosmic_zeitgeist=self.cosmic_zeitgeist
        )
        self.last_baie_score = self.reincarnator.score if self.reincarnator else 0
        
        num_encounters = min(int(2 * len(self.population)), 5000)
        global_dist = self.population_manager.get_global_path_distribution(self.population)
        for _ in range(num_encounters):
            valid_population = [p for p in self.population if p not in culled_this_gen]
            if len(valid_population) < 2: break
            
            entity1, entity2 = self.interaction_handler.select_opponents(valid_population, self.reincarnator)
            if not entity1 or not entity2: continue
            
            if entity1 not in culled_this_gen and entity2 not in culled_this_gen:
                culled_entity = self.interaction_handler.entity_interaction(
                    entity1, entity2, self.population, self.reincarnator, global_dist, self.cosmic_zeitgeist
                )
                if culled_entity and culled_entity.trait != "Reincarnator":
                    culled_this_gen.add(culled_entity)
                    
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
                self.population_manager.recalculate_and_normalize_entity(self.reincarnator, self.population_manager.get_global_path_distribution(self.population), self.cosmic_zeitgeist)
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
                cosmic_zeitgeist=self.cosmic_zeitgeist
            )
        
        if len(self.population) > self.population_hard_cap:
            num_to_cull = len(self.population) - self.population_hard_cap
            self.population.sort(key=lambda p: p.score)
            culled_at_cap = self.population[:num_to_cull]
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

    def _run_inorganic_phase(self, num_generations=50, activity_threshold=15.0):
        print("\n=== 进入无机实体培养阶段 ===")
        print("...正在演化纯粹的“活性”与“稳定性”概念...")
        inorganic_pop = [{'id': i, 'activity': random.uniform(1,5), 'stability': random.uniform(1,5)} for i in range(100)]
        for gen in range(num_generations + 1):
            self.display_manager.update_and_display_progress('inorganic', gen, num_generations)
            time.sleep(0.05)
            if gen == num_generations: break
            for p in inorganic_pop: p['score'] = p['activity'] + p['stability']
            inorganic_pop.sort(key=lambda x: x['score'], reverse=True)
            survivors = inorganic_pop[:50]
            if survivors and survivors[0]['activity'] > activity_threshold:
                best_prototype = survivors[0]
                self.display_manager.update_and_display_progress('inorganic', num_generations, num_generations)
                print(f"\n\033[92m>>> 原型验证完成! 发现高活性实体 (ID: {best_prototype['id']})! <<<\n>>> 数据: 活性={best_prototype['activity']:.2f}, 稳定性={best_prototype['stability']:.2f}\n>>> 基于原型，已将目标平均分调整为: {self.target_avg_score:.2f} <<<\033[0m")
                return
            new_pop = list(survivors)
            for _ in range(50):
                parent = random.choice(survivors) if survivors else {'activity': random.uniform(1,5), 'stability': random.uniform(1,5)}
                new_entity = {'id': len(new_pop) + gen*100, 'activity': parent['activity'] + random.uniform(-0.5, 0.8), 'stability': parent['stability'] + random.uniform(-0.5, 0.5)}
                new_pop.append(new_entity)
            inorganic_pop = new_pop
        print("\n\033[93m警告: 未能在无机阶段发现高活性原型，将使用默认参数开始演算。\033[0m")
        
    def start(self, num_generations=80):
        self._run_inorganic_phase()
        print("\n\n>>> 正式演算开始... <<<")
        self._create_initial_population()
        self.total_generations = num_generations
        
        KAOSELANNA_PHASE_END_SCORE = 18.45
        was_paused = False

        while self.generation < num_generations:
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

            # --- PTL ---
            if not self.aeonic_cycle_mode and self.reincarnator and self.reincarnator.score > KAOSELANNA_PHASE_END_SCORE:
                self.aeonic_cycle_manager.trigger_aeonic_cycle_countdown(self.reincarnator)
                self.display_manager.display_interruption_animation()
                if self.aeonic_cycle_manager.initialize_aeonic_cycle(self.reincarnator, self.population, self.cosmic_zeitgeist):
                    self.aeonic_cycle_mode = True
                else:
                    self.aeonic_cycle_mode = False

            # --- ML ---
            if self.aeonic_cycle_mode:
                self.aeonic_cycle_manager.run_aeonic_cycle_generation(self.population, self.reincarnator, self.cosmic_zeitgeist)
                if self.reincarnator and len(self.reincarnator.held_fire_seeds) >= len(TITAN_NAMES):
                    should_end_simulation = self.aeonic_cycle_manager.end_aeonic_cycle(
                        self.reincarnator, self.population, self.base_titan_affinities,
                        self.population_manager.mutation_rate, self.population_soft_cap, self.cosmic_zeitgeist
                    )
                    if should_end_simulation: break
                
                self.display_manager.update_and_display_progress('cycle', self.aeonic_cycle_manager.aeonic_cycle_number, 10)
            else:
                self.generation += 1
                culled_this_gen = self._run_one_generation()
                self._evolve_and_grow(culled_this_gen)
                current_kaos_score = self.reincarnator.score if self.reincarnator else 0
                self.display_manager.update_and_display_progress('normal', current_kaos_score, KAOSELANNA_PHASE_END_SCORE)
            
            if not self.population: 
                print("\n种群已灭绝！")
                break

            if 'next' in getattr(self.debugger, 'last_command', ''):
                self.debugger.paused = True
                self.debugger.last_command = ''
            
        # --- EoS ---
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