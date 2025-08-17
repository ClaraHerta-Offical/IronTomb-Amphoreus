import numpy as np
from constants import PATH_NAMES, PATH_RELATIONSHIP_MATRIX, path_idx, VOTE_WEIGHTS, TITAN_NAMES
import math
from models import TitanToPathModel

class Pathstrider:
    def __init__(self, name, titan_affinities, entity_trait="Mortal", titan_to_path_model=None):
        self.name = name
        self._titan_to_path_model = titan_to_path_model if titan_to_path_model is not None else TitanToPathModel()
        self.titan_affinities = np.array(titan_affinities).clip(min=0)
        self.trait = entity_trait
        self.golden_one_tenure = 0
        self.heroic_tendency = 0
        self._path_affinities = None
        
        self.score = 0
        
        self.hp = 100.0  # 仅对泰坦有意义
        self.is_titan_boss = False
        self.titan_aspect = None  
        self.data_modification_unlocked = False

        self.activity = 0
        self.stability = 0
        
        if self.trait == "Reincarnator" or self.trait == "GoldenOne":
            self.is_titan_form = None
        else:
            self.is_titan_form = None  
        self.held_fire_seeds = set() 


    @property
    def path_affinities(self):
        if self._path_affinities is None:
            self._path_affinities = self._titan_to_path_model.get_path_affinities(self.titan_affinities)
        return self._path_affinities
    
    @property
    def dominant_path_idx(self):
        return np.argmax(self.path_affinities)
    
    @property
    def purity(self):
        path_affs = self.path_affinities
        if np.sum(path_affs) == 0: return 0
        return np.max(path_affs) / np.sum(path_affs)

    def recalculate_concepts(self, zeitgeist_multiplier=1.0, path_distribution=None):
        """
        根据新的议会和评分机制重新计算。
        - zeitgeist_multiplier: 由 ParliamentManager 计算出的全局乘数
        - path_distribution: 全局的命途饱和度分布
        """
        if not np.all(np.isfinite(self.titan_affinities)):
            self.titan_affinities = np.ones(len(self.titan_affinities))

        self._path_affinities = None 
        
        self.activity = np.linalg.norm(self.path_affinities[:6]) * 10
        self.stability = np.linalg.norm(self.path_affinities[6:]) * 10
        
        base_potential = (self.activity + self.stability) * (1 + self.purity)

        # 饱和度惩罚
        saturation_modifier = 1.0
        if path_distribution is not None:
            dominance_penalty_factor = path_distribution[self.dominant_path_idx]
            saturation_modifier = 1.0 / (1.0 + 2 * dominance_penalty_factor)

        # 评分公式：基础潜力 * 饱和度修正 * 思潮乘数
        self.score = base_potential * saturation_modifier * zeitgeist_multiplier
        
        # 如果是泰坦，评分获得加成
        if self.is_titan_boss:
            self.score *= 5.0

        self.heroic_tendency = self.activity

    def generate_vote_proposal(self):
        """
        生成该实体的选票，即它所期望的理想思潮。
        """
        vote_vector = np.zeros(len(PATH_NAMES))
        my_dom_idx = self.dominant_path_idx
        
        for i in range(len(PATH_NAMES)):
            if i == my_dom_idx:
                vote_vector[i] = 1.0
                continue

            relationship = PATH_RELATIONSHIP_MATRIX[my_dom_idx, i]
            
            if relationship == "SYNERGY":
                vote_vector[i] = 0.5  
            elif relationship == "MENTORSHIP":
                vote_vector[i] = 0.2  
            elif relationship == "REPULSION":
                vote_vector[i] = -0.7 
            elif relationship == "CLASH":
                vote_vector[i] = -1.0 
        
        e_x = np.exp(vote_vector - np.max(vote_vector))
        return e_x / e_x.sum(axis=0)

    def get_vote_weight(self):
        """
        获取该实体投票权重
        """
        return VOTE_WEIGHTS.get(self.trait, 1.0)


    def internal_purification(self, purity_factor):
        if purity_factor <= 0 or len(self.titan_affinities) < 2: return
        dominant_idx = np.argmax(self.titan_affinities)
        dominant_value = self.titan_affinities[dominant_idx]
        reduction_amount = dominant_value * purity_factor
        mask = np.ones(len(self.titan_affinities), dtype=bool)
        mask[dominant_idx] = False
        self.titan_affinities[mask] -= reduction_amount
        self.titan_affinities = self.titan_affinities.clip(min=0)
        self._path_affinities = None

    def __repr__(self):
        dominant_path_name = PATH_NAMES[self.dominant_path_idx]
        
        tags = []
        if self.is_titan_boss:
            tags.append(f"泰坦真身: {self.titan_aspect} (HP: {self.hp:.0f})")
        elif self.trait == "Reincarnator":
            tags.append("卡厄斯兰那" if self.name != "Neikos-0496" else "白厄")
        elif self.trait == "GoldenOne":
            if self.titan_aspect:
                status = "权限已解锁" if self.data_modification_unlocked else "待解锁"
                tags.append(f"黄金裔({self.titan_aspect}, {status})")
            else:
                tags.append(f"黄金裔(任期:{self.golden_one_tenure})")

        if self.is_titan_form: 
            tags.append(f"泰坦: {self.is_titan_form}")
        
        if self.held_fire_seeds:
            tags.append(f"火种({len(self.held_fire_seeds)})")

        tag_str = ", ".join(tags) if tags else f"'{dominant_path_name}'的追随者"

        return f"[{self.name}] <{tag_str}>(评分:{self.score:.2f}|纯:{self.purity:.2f})"

    def to_dict(self):
        """将实体状态转换为可序列化的字典。"""
        return {
            'name': self.name,
            'titan_affinities': self.titan_affinities.tolist(), # NumPy 数组转为列表
            'trait': self.trait,
            'golden_one_tenure': self.golden_one_tenure,
            'score': self.score,
            'hp': self.hp,
            'is_titan_boss': self.is_titan_boss,
            'titan_aspect': self.titan_aspect,
            'data_modification_unlocked': self.data_modification_unlocked,
            'is_titan_form': self.is_titan_form,
            'held_fire_seeds': list(self.held_fire_seeds) # set 转为列表
        }

    @classmethod
    def from_dict(cls, data, titan_to_path_model):
        """从字典和模型实例中重建实体对象。"""
        entity = cls(
            name=data['name'],
            titan_affinities=np.array(data['titan_affinities']),
            entity_trait=data.get('trait', 'Mortal'),
            titan_to_path_model=titan_to_path_model
        )
        entity.golden_one_tenure = data.get('golden_one_tenure', 0)
        entity.score = data.get('score', 0)
        entity.hp = data.get('hp', 100.0)
        entity.is_titan_boss = data.get('is_titan_boss', False)
        entity.titan_aspect = data.get('titan_aspect', None)
        entity.data_modification_unlocked = data.get('data_modification_unlocked', False)
        entity.is_titan_form = data.get('is_titan_form', None)
        entity.held_fire_seeds = set(data.get('held_fire_seeds', []))
        
        # 重新计算派生属性
        entity.recalculate_concepts()

        return entity
