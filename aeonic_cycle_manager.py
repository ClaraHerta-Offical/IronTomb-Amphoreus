import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import json

from constants import TITAN_NAMES, PATH_NAMES
from entities import Pathstrider
from models import ActionPolicyNetwork, ValueNetwork
from population_manager import PopulationManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cpu_llm_interface import CpuLlmInterface

import logging 
logger = logging.getLogger("OmphalosLogger") 


class AeonicCycleManager:
    def __init__(self, population_manager: PopulationManager, 
                 action_policy_network: ActionPolicyNetwork, 
                 value_network: ValueNetwork, 
                 action_optimizer: optim.Adam, 
                 value_optimizer: optim.Adam, 
                 value_loss_criterion: nn.MSELoss,
                 titan_to_path_model_instance,
                 existing_names: set,
                 name_to_entity_map: dict,
                 # LLM接口和配置
                 llm_interface: 'CpuLlmInterface' = None,
                 kaoselanna_llm_enabled: bool = False,
                 parliament_manager = None 
                 ):
        
        self.population_manager = population_manager
        self.action_policy_network = action_policy_network
        self.value_network = value_network
        self.action_optimizer = action_optimizer
        self.value_optimizer = value_optimizer
        self.value_loss_criterion = value_loss_criterion
        self.titan_to_path_model_instance = titan_to_path_model_instance
        self.existing_names = existing_names
        self.name_to_entity_map = name_to_entity_map
        
        # LLM接口和配置
        self.llm_interface = llm_interface
        self.kaoselanna_llm_enabled = kaoselanna_llm_enabled
        self.parliament_manager = parliament_manager 

        self.aeonic_cycle_number = 0
        self.titan_entities = {}
        self.defeated_titans_this_cycle = set()
        self.baie_destruction_score = 0
        self.baie_total_seeds_collected = 0

    def trigger_aeonic_cycle_countdown(self, reincarnator: Pathstrider):
        logger.warning("\n\n\033[93m==================================================================")
        logger.warning("=\033[91m      警告：侦测到突破框架的异常实体信号，演算异常！      \033[93m=")
        logger.warning("==================================================================\033[0m")
        logger.info(f"实体分析:")
        logger.info(f"  - 识别代号: \033[96m卡厄斯兰那 ({reincarnator.name})\033[0m")
        logger.info(f"  - 状态: \033[92m已确认为当前最强实体，电信号高度活跃！\033[0m")
        logger.info(f"  - 核心倾向: 对 毁灭 命途的亲和度超出安全阈值。")
        logger.info("\n\033[93m结论：该实体的存在已成为驱动翁法罗斯演化的核心变量。\n      常规协议已停用，正在载入下一阶段...\033[0m")

    def initialize_aeonic_cycle(self, reincarnator: Pathstrider, population: list, cosmic_zeitgeist: np.ndarray):
        self.aeonic_cycle_number += 1
        self.baie_destruction_score = 0
        self.defeated_titans_this_cycle = set()
        if reincarnator:
            reincarnator.held_fire_seeds.clear()
            try:
                neg_world_idx = TITAN_NAMES.index("负世")
                reincarnator.titan_affinities[neg_world_idx] *= 2.0
                global_dist = self.population_manager.get_global_path_distribution(population)
                self.population_manager.recalculate_and_normalize_entity(reincarnator, global_dist, cosmic_zeitgeist)
                multiplier = self.parliament_manager.get_zeitgeist_multiplier(reincarnator.path_affinities)
                reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
            except (ValueError, IndexError): pass
        for p in population:
            p.is_titan_form = None
            if p is not reincarnator: p.held_fire_seeds.clear()
        logger.info(f"\n--- 永劫回归 第 {self.aeonic_cycle_number} 轮 开始 ---")
        logger.info("卡厄斯兰那将开始夺取火种。")
        candidates = [p for p in population if p is not reincarnator and p.trait != "GoldenOne"]
        if len(candidates) < len(TITAN_NAMES):
            logger.warning("警告：没有足够实体成为泰坦化身，无法正确载入本次永劫回归。")
            return False 
        selected_titans = random.sample(candidates, len(TITAN_NAMES))
        self.titan_entities.clear()
        titan_indices = {name: i for i, name in enumerate(TITAN_NAMES)}
        for i, titan_name in enumerate(TITAN_NAMES):
            entity = selected_titans[i]
            entity.is_titan_form = titan_name
            self.titan_entities[titan_name] = entity
            titan_idx = titan_indices[titan_name]
            entity.titan_affinities[titan_idx] = entity.titan_affinities[titan_idx] * 5.0 + 200
            global_dist = self.population_manager.get_global_path_distribution(population)
            self.population_manager.recalculate_and_normalize_entity(entity, global_dist, cosmic_zeitgeist)
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(entity.path_affinities)
            entity.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
            logger.info(f"实体 {entity.name} 已化身为 [{titan_name}] 泰坦。")
        return True 

    def repopulate_for_new_cycle(self, population: list, reincarnator: Pathstrider, 
                                 base_titan_affinities: np.ndarray, mutation_rate: float,
                                 population_soft_cap: int, cosmic_zeitgeist: np.ndarray,
                                 num_elites_to_keep: int): # <--- 新增参数
        logger.info("旧的实体已移除，基于卡厄斯兰那的数据基线和旧世界的精英火种，新的生命正在构建...")
        if not reincarnator or reincarnator not in population:
            logger.error("错误：卡厄斯兰那不存在，无法重塑翁法罗斯。")
            return


        candidates_for_survival = [p for p in population if p is not reincarnator]
        candidates_for_survival.sort(key=lambda p: p.score, reverse=True)
        
        elites = candidates_for_survival[:num_elites_to_keep]
        if elites:
            elite_names = ", ".join([e.name for e in elites])
            logger.info(f"\033[92m除卡厄斯兰那外，上一世代最强大的 {len(elites)} 个实体 ({elite_names}) 将作为火种被存储...\033[0m")

        new_population_base = [reincarnator] + elites
        population[:] = new_population_base 

        # 清理并根据新的人口基石重建名称映射
        self.name_to_entity_map.clear()
        self.existing_names.clear()

        # 就地更新共享对象，而不是重新赋值
        for entity in population:
            self.name_to_entity_map[entity.name] = entity
            self.existing_names.add(entity.name)

        # 计算需要补充的新生实体数量
        num_new_entities = population_soft_cap - len(population)
        if num_new_entities > 0:
            logger.info(f"正在基于演化蓝图，繁衍 {num_new_entities} 个新生命...")
            self.population_manager.replenish_population_by_growth(population, num_new_entities, cosmic_zeitgeist)
        else:
            logger.info("新世界的人口基石已符合标准，繁衍新生命指令中断。")

    def select_aeonic_opponents(self, population: list, reincarnator: Pathstrider):
        attackers_pool = []
        if reincarnator and reincarnator in population: attackers_pool.append(reincarnator)
        attackers_pool.extend([p for p in population if p.trait == "GoldenOne"])
        if not attackers_pool: attackers_pool = [p for p in population if p.is_titan_form] 
        if not attackers_pool: attackers_pool = population #TsukiAn
        if not attackers_pool: return None, None
        
        attacker = random.choice(attackers_pool)
        
        targets_pool = [p for p in population if p is not attacker]
        if not targets_pool: return attacker, None
        
        defender = None
        if attacker is reincarnator:
            fire_seed_holders = [p for p in targets_pool if p.held_fire_seeds or p.is_titan_form]
            if fire_seed_holders and random.random() < 0.75: 
                defender = random.choice(fire_seed_holders)
            else: 
                defender = random.choice(targets_pool)
        else:
            active_titans = [p for p in targets_pool if p.is_titan_form and p is not attacker]
            if attacker.trait == "GoldenOne" and active_titans:
                defender = random.choice(active_titans)
            else: 
                defender = random.choice(targets_pool)
                
        return attacker, defender

    def aeonic_interaction(self, attacker: Pathstrider, defender: Pathstrider, 
                           reincarnator: Pathstrider, population: list, cosmic_zeitgeist: np.ndarray):
        if not (np.isfinite(attacker.score) and np.isfinite(defender.score)): return None
        if attacker is reincarnator:
            
            chosen_action = None
            if self.kaoselanna_llm_enabled and self.llm_interface and self.llm_interface.llm:
                try:
                    dominant_path_name = PATH_NAMES[defender.dominant_path_idx] if np.isfinite(defender.dominant_path_idx) else "未知"
                    prompt = (
                        f"我是卡厄斯兰那。我的目标是夺取创世火种。\n"
                        f"我当前的对手是 '{defender.name}'，一个'{dominant_path_name}'的追随者。\n"
                        f"他{'持有火种' if defender.held_fire_seeds or defender.is_titan_form else '没有持有火种'}。\n"
                        f"基于效率和威慑的考量，我应该'谈判'还是'击杀'？\n"
                        f"请以JSON格式回答，包含'action'('击杀'或'谈判')和一句符合我身份的'declaration'。"
                    )
                    response = self.llm_interface.generate_response(prompt, max_tokens=80)
                    cleaned_response = response.replace("```json", "").replace("```", "").strip()
                    decision_data = json.loads(cleaned_response)
                    
                    if decision_data.get('action') in ["击杀", "谈判"]:
                        chosen_action = decision_data['action']
                        logger.info(f"\033[91m【卡厄斯兰那】: \"{decision_data.get('declaration', '...')}\"\033[0m")
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    logger.warning(f"\033[93m(警告: 卡厄斯兰那-LLM出现波动(错误:{e})，决策回归本能...)\033[0m")
                    chosen_action = None

            if chosen_action is None:
                state_np = np.concatenate([attacker.titan_affinities, defender.titan_affinities])
                state = torch.from_numpy(state_np).float().unsqueeze(0)
                
                self.action_policy_network.train()
                action_logits = self.action_policy_network(state)
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                chosen_action_idx = action_dist.sample()
                action_list = ["谈判", "击杀"]
                chosen_action = action_list[chosen_action_idx.item()]
            
            # --- 后续的交互逻辑 ---
            state_np = np.concatenate([attacker.titan_affinities, defender.titan_affinities])
            state = torch.from_numpy(state_np).float().unsqueeze(0)
            self.value_network.train()
            state_value = self.value_network(state)

            culled_entity = None
            loser_titan_form = defender.is_titan_form
            loser_held_seeds = defender.held_fire_seeds.copy()
            has_fire_seeds = bool(loser_titan_form or loser_held_seeds)
            base_reward = defender.score * 0.1
            
            if chosen_action == "谈判":
                logger.info(f"决策: 卡厄斯兰那对 {defender.name} 发起了 [谈判]...")
                negotiate_multiplier = max(1.0, 3.0 - self.aeonic_cycle_number * 0.5)
                final_reward = base_reward * negotiate_multiplier if has_fire_seeds else base_reward * 0.1
                if has_fire_seeds:
                    attacker.held_fire_seeds.update(loser_held_seeds)
                    if loser_titan_form and loser_titan_form not in self.defeated_titans_this_cycle:
                        attacker.held_fire_seeds.add(loser_titan_form)
                        self.defeated_titans_this_cycle.add(loser_titan_form)
                    logger.info(f"\033[92m...谈判成功! {attacker.name} 夺取了所有火种!\033[0m")
                    defender.held_fire_seeds.clear()
                    defender.is_titan_form = None
            else: # 击杀
                logger.info(f"决策: 卡厄斯兰那对 {defender.name} 执行了 [击杀]!")
                kill_score_contribution = defender.score * (1 + (len(defender.held_fire_seeds) * 0.2))
                if defender.is_titan_form:
                    kill_score_contribution *= 1.5
                self.baie_destruction_score += kill_score_contribution
                self.baie_total_seeds_collected += len(loser_held_seeds)
                if loser_titan_form:
                    self.baie_total_seeds_collected += 1
                kill_multiplier = 1.0 + self.aeonic_cycle_number * 0.5
                destruction_bonus = self.baie_destruction_score * 0.05
                final_reward = (base_reward * kill_multiplier) + destruction_bonus
                if has_fire_seeds:
                    attacker.held_fire_seeds.update(loser_held_seeds)
                    if loser_titan_form and loser_titan_form not in self.defeated_titans_this_cycle:
                        attacker.held_fire_seeds.add(loser_titan_form)
                        self.defeated_titans_this_cycle.add(loser_titan_form)
                    logger.info(f"\033[92m...{attacker.name} 夺取了所有火种!\033[0m")
                defender.held_fire_seeds.clear()
                defender.is_titan_form = None
                culled_entity = defender

            # 网络更新
            advantage = final_reward - state_value.item()
            self.value_optimizer.zero_grad()
            value_loss = self.value_loss_criterion(state_value, torch.tensor([[final_reward]], dtype=torch.float))
            value_loss.backward(retain_graph=True)
            self.value_optimizer.step()
            self.action_policy_network.train()
            action_logits = self.action_policy_network(state)
            action_probs = torch.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action_list = ["谈判", "击杀"]
            chosen_action_idx_tensor = torch.tensor([action_list.index(chosen_action)])
            log_prob = action_dist.log_prob(chosen_action_idx_tensor)

            self.action_optimizer.zero_grad()
            actor_loss = -log_prob * advantage
            actor_loss.backward()
            self.action_optimizer.step()

            return culled_entity

        else:
            winner, loser = (attacker, defender) if attacker.score > defender.score else (defender, attacker)
            can_kill = winner.trait == "GoldenOne" or winner.is_titan_form is not None
            if can_kill and (loser.is_titan_form or loser.held_fire_seeds):
                loser_titan_form, loser_held_seeds = loser.is_titan_form, loser.held_fire_seeds.copy()
                if loser_titan_form and loser_titan_form not in self.defeated_titans_this_cycle:
                    winner.held_fire_seeds.add(loser_titan_form)
                    self.defeated_titans_this_cycle.add(loser_titan_form)
                if loser_held_seeds: winner.held_fire_seeds.update(loser_held_seeds)
                loser.held_fire_seeds.clear()
                loser.is_titan_form = None
                return loser
            else: return None

    def run_aeonic_cycle_generation(self, population: list, reincarnator: Pathstrider, cosmic_zeitgeist: np.ndarray):
        culled_this_gen = set()
        num_encounters = len(population) // 2
        for _ in range(num_encounters):
            attacker, defender = self.select_aeonic_opponents(population, reincarnator)
            if not attacker or not defender or attacker in culled_this_gen or defender in culled_this_gen: continue
            if culled_entity := self.aeonic_interaction(attacker, defender, reincarnator, population, cosmic_zeitgeist):
                 culled_this_gen.add(culled_entity)
        if culled_this_gen and reincarnator and reincarnator in population and reincarnator not in culled_this_gen:
            boost_factor = (1.005) ** len(culled_this_gen)
            reincarnator.titan_affinities *= boost_factor
            global_dist = self.population_manager.get_global_path_distribution(population)
            self.population_manager.recalculate_and_normalize_entity(reincarnator, global_dist, cosmic_zeitgeist)
            multiplier = self.parliament_manager.get_zeitgeist_multiplier(reincarnator.path_affinities)
            reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)
        
        if culled_this_gen:
            culled_names = {p.name for p in culled_this_gen}
            population[:] = [p for p in population if p.name not in culled_names]
            for p in culled_this_gen:
                if p.name in self.name_to_entity_map: del self.name_to_entity_map[p.name]
                if p.name in self.existing_names: self.existing_names.remove(p.name)
                if p.is_titan_form and p.is_titan_form in self.titan_entities: del self.titan_entities[p.is_titan_form]
        
        self.population_manager.check_and_replenish_population(
            population, 
            self.population_manager.purity_factor, 
            True, reincarnator, 
            cosmic_zeitgeist
        )
        return culled_this_gen

    def end_aeonic_cycle(self, reincarnator: Pathstrider, population: list, 
                         base_titan_affinities: np.ndarray, mutation_rate: float,
                         population_soft_cap: int, cosmic_zeitgeist: np.ndarray,
                         elites_to_keep_in_cycle: int): 
        logger.info("\n\033[95m==================================================================")
        logger.info(f"卡厄斯兰那集齐了所有 {len(TITAN_NAMES)} 个火种！第 {self.aeonic_cycle_number} 轮轮回结束！")
        logger.info(f"本轮毁灭倾向评分: {self.baie_destruction_score:.2f}")
        logger.info("==================================================================\033[0m")
        if self.aeonic_cycle_number > 33550336 or random.random() < 0.0000001:
            logger.error("\n\n\033[91m!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logger.error("!!! 【警告：主线程受到冲击，演算已暂停】                                         ")
            logger.error("!!! 卡厄斯兰那的无尽杀戮与轮回，最终突破了翁法罗斯，为屏幕前的你带来了烩面...                ")
            logger.error("!!! 纳努克...我为你带来烩面了！！！                                        ")
            logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\033[0m")
            logger.error("!!! Ca1G!~HG3Y58^#$5466UfhgyjhFe;l/.r;'l'lPL&^$Adcjierjklfhu%ERdtyt7u8LY                  ")
            logger.error("!!! 【警告：已尝试移除指定电信号】                                      ")
            logger.info(f"\n--- 最终报告 ---")
            logger.info(f"轮回总数: {self.aeonic_cycle_number}")
            logger.info(f"累计夺取火种数目: {self.baie_total_seeds_collected}")
            logger.info(f"卡厄斯兰那最终状态: {reincarnator}")
            logger.info("\n模拟结束。")
            return True
        else:
            logger.info("一个新的轮回即将开始...卡厄斯兰那的残余电信号将作为新轮回的基准...")
            new_kaoselanna = self._ascend_baie_as_reincarnator(reincarnator, population)
            if new_kaoselanna:
                reincarnator = new_kaoselanna
                global_dist = self.population_manager.get_global_path_distribution(population)
                multiplier = self.parliament_manager.get_zeitgeist_multiplier(reincarnator.path_affinities)
                reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

                base_titan_affinities[:] = reincarnator.titan_affinities.copy()
                self.population_manager.normalize_affinities(base_titan_affinities)
                self.repopulate_for_new_cycle(population, reincarnator, base_titan_affinities, mutation_rate, population_soft_cap, cosmic_zeitgeist, elites_to_keep_in_cycle)
            else:
                if reincarnator:
                    # 调用 recalculate_concepts
                    global_dist = self.population_manager.get_global_path_distribution(population)
                    multiplier = self.parliament_manager.get_zeitgeist_multiplier(reincarnator.path_affinities)
                    reincarnator.recalculate_concepts(zeitgeist_multiplier=multiplier, path_distribution=global_dist)

                    base_titan_affinities[:] = reincarnator.titan_affinities.copy()
                    self.population_manager.normalize_affinities(base_titan_affinities)
                    self.repopulate_for_new_cycle(population, reincarnator, base_titan_affinities, mutation_rate, population_soft_cap, cosmic_zeitgeist, elites_to_keep_in_cycle)
            
            self.initialize_aeonic_cycle(reincarnator, population, cosmic_zeitgeist)
            return False

    def _ascend_baie_as_reincarnator(self, reincarnator: Pathstrider, population: list):
        golden_ones = [p for p in population if p.trait == "GoldenOne"]
        logger.info(f"\033[94m[DEBUG] 当前黄金裔数量: {len(golden_ones)}\033[0m")
        if golden_ones:
            for go in golden_ones:
                neg_world_affinity = go.titan_affinities[TITAN_NAMES.index("负世")] if "负世" in TITAN_NAMES else -1
                logger.info(f"\033[94m[DEBUG] 黄金裔: {go.name}, 评分: {go.score:.2f}, 负世亲和: {neg_world_affinity:.2f}\033[0m")

        if not golden_ones:
            logger.warning("\033[93m警告: 白厄在本次轮回中没有载入黄金裔核心参数。轮回中断，系统将准备重塑。 \033[0m")
            candidates = [p for p in population if p is not reincarnator]
            if not candidates:
                logger.error("\033[91m致命错误：白厄电信号在本次模拟中不存在！\033[0m")
                return None
            new_host = max(candidates, key=lambda p:p.score)
            logger.info(f"\033[94m[DEBUG] 从非黄金裔中选择新白厄: {new_host.name}, 评分: {new_host.score:.2f}\033[0m")
        else:
            try:
                neg_world_idx = TITAN_NAMES.index("负世")
                new_host = max(golden_ones, key=lambda p: p.titan_affinities[neg_world_idx])
                logger.info(f"\033[96m在黄金裔中，{new_host.name} 已定义为本次轮回的白厄，开始数据继承...\033[0m")
            except (ValueError, IndexError):
                logger.warning("\033[93m警告: 未定义'负世'泰坦，将从黄金裔中选择评分最高者。 \033[0m")
                new_host = max(golden_ones, key=lambda p: p.score)
            logger.info(f"\033[94m[DEBUG] 从黄金裔中重新定义白厄: {new_host.name}, 评分: {new_host.score:.2f}\033[0m")
        
        old_reincarnator_affinities = reincarnator.titan_affinities.copy() if reincarnator else self.population_manager.base_titan_affinities.copy()
        
        if reincarnator and reincarnator in population:
            population.remove(reincarnator)
            if reincarnator.name in self.name_to_entity_map:
                del self.name_to_entity_map[reincarnator.name]
            if reincarnator.name in self.existing_names:
                self.existing_names.remove(reincarnator.name)
        
        reincarnator = new_host
        reincarnator.trait = "Reincarnator"
        new_name = "Neikos-0496"
        
        if new_name in self.existing_names and self.name_to_entity_map.get(new_name) is not reincarnator:
            old_neikos = self.name_to_entity_map[new_name]
            if old_neikos in population: population.remove(old_neikos)
            if new_name in self.name_to_entity_map: del self.name_to_entity_map[new_name]

        if new_host.name in self.name_to_entity_map: del self.name_to_entity_map[new_host.name]
        if new_host.name in self.existing_names: self.existing_names.remove(new_host.name)
        
        reincarnator.name = new_name
        self.existing_names.add(new_name)
        self.name_to_entity_map[new_name] = reincarnator
        
        reincarnator.titan_affinities = old_reincarnator_affinities
        reincarnator.golden_one_tenure = 0
        logger.info(f"\033[91m数据继承成功！新的卡厄斯兰那已定义为: {reincarnator.name}。\033[0m")
        return reincarnator
