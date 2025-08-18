import numpy as np
import torch
import torch.nn as nn
from constants import TITAN_NAMES, PATH_NAMES, path_idx

class TitanToPathModel:
    def __init__(self):
        self.titan_to_path_matrix = np.zeros((len(TITAN_NAMES), len(PATH_NAMES)))
        self.path_to_titan_feedback_matrix = np.zeros((len(PATH_NAMES), len(TITAN_NAMES)))
        titan_idx = {name: i for i, name in enumerate(TITAN_NAMES)}
        path_idx_local = {name: i for i, name in enumerate(PATH_NAMES)} 
        self.titan_to_path_matrix[titan_idx["纷争"], path_idx_local["毁灭"]] = 0.9
        self.titan_to_path_matrix[titan_idx["负世"], path_idx_local["存护"]] = 0.1
        self.titan_to_path_matrix[titan_idx["负世"], path_idx_local["毁灭"]] = 1.0
        self.titan_to_path_matrix[titan_idx["理性"], path_idx_local["智识"]] = 0.9
        self.titan_to_path_matrix[titan_idx["浪漫"], path_idx_local["同谐"]] = 0.8
        self.titan_to_path_matrix[titan_idx["死亡"], path_idx_local["虚无"]] = 0.8
        self.titan_to_path_matrix[titan_idx["岁月"], path_idx_local["记忆"]] = 0.8
        self.titan_to_path_matrix[titan_idx["门径"], path_idx_local["均衡"]] = 0.8
        self.titan_to_path_matrix[titan_idx["天空"], path_idx_local["巡猎"]] = 0.6
        self.titan_to_path_matrix[titan_idx["大地"], path_idx_local["丰饶"]] = 0.6
        self.titan_to_path_matrix[titan_idx["大地"], path_idx_local["存护"]] = 0.4
        self.titan_to_path_matrix[titan_idx["海洋"], path_idx_local["繁育"]] = 0.7
        self.titan_to_path_matrix[titan_idx["诡计"], path_idx_local["欢愉"]] = 0.7
        self.titan_to_path_matrix[titan_idx["诡计"], path_idx_local["虚无"]] = 0.3
        self.titan_to_path_matrix[titan_idx["律法"], path_idx_local["均衡"]] = 0.6
        self.titan_to_path_matrix[titan_idx["律法"], path_idx_local["智识"]] = 0.3
        self.titan_to_path_matrix[titan_idx["纷争"], path_idx_local["同谐"]] = -0.5
        self.titan_to_path_matrix[titan_idx["死亡"], path_idx_local["丰饶"]] = -0.6
        self.titan_to_path_matrix[titan_idx["理性"], path_idx_local["欢愉"]] = -0.4
        self.titan_to_path_matrix[titan_idx["岁月"], path_idx_local["繁育"]] = -0.3
        self.titan_to_path_matrix[titan_idx["天空"], path_idx_local["存护"]] = -0.2
        self.titan_to_path_matrix[titan_idx["负世"], path_idx_local["巡猎"]] = -0.3
        self.path_to_titan_feedback_matrix[path_idx_local["毁灭"], titan_idx["纷争"]] = 0.1
        self.path_to_titan_feedback_matrix[path_idx_local["存护"], titan_idx["负世"]] = 0.1
        self.path_to_titan_feedback_matrix[path_idx_local["智识"], titan_idx["理性"]] = 0.1
        self.path_to_titan_feedback_matrix[path_idx_local["巡猎"], titan_idx["天空"]] = 0.1

    def get_path_affinities(self, titan_affinities):
        raw_affinities = np.dot(titan_affinities, self.titan_to_path_matrix)
        non_negative_affinities = np.maximum(0, raw_affinities)
        total_affinity = np.sum(non_negative_affinities)
        if total_affinity > 0:
            return non_negative_affinities / total_affinity
        else:
            return non_negative_affinities

class HybridGuideNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HybridGuideNetwork, self).__init__()
        # 增加网络深度和复杂度
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # 增加Dropout层，防止过拟合，强制网络探索
            nn.Linear(hidden_size, hidden_size * 2), # 增加神经元数量
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state, exploration_noise=0.1):
        final_output = self.network(state)
        # 探索噪声，有助于跳出局部最优
        if self.training:
            noise = torch.randn_like(final_output) * exploration_noise
            final_output += noise
        # 输出非负
        return torch.relu(final_output)

class ValueNetwork(nn.Module):
    """
    价值网络
    """
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, 1) 

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class ActionPolicyNetwork(nn.Module):
    """
    行动策略网络
    在给定状态下，决定要采取的具体行动（谈判或击杀）
    """
    def __init__(self, input_size, hidden_size, output_size=2):
        super(ActionPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, output_size) # // 输出: [谈判logit, 击杀logit]

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)
