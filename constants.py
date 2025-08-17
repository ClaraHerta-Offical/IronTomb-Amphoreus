import numpy as np

GREEK_ROOTS = [
    "Aion", "Aether", "Ananke", "Chaos", "Chronos", "Erebus", "Gaea", "Hemera",
    "Hypnos", "Nesoi", "Nyx", "Ourea", "Physis", "Pontus", "Tartarus", "Thalassa",
    "Thanatos", "Uranus", "Helios", "Eos", "Selene", "Leto", "Asteria", "Astraeus",
    "Pallas", "Perses", "Krios", "Koios", "Iapetus", "Hyperion", "Okeanos", "Tethys",
    "Theia", "Themis", "Mnemosyne", "Phoebe", "Rhea", "Metis", "Typhon", "Echidna","Lans",
    "Neikos" 
]

TITAN_NAMES = [
    "天空", "大地", "海洋", "浪漫", "负世", "理性",
    "诡计", "纷争", "死亡", "岁月", "律法", "门径"
]

PATH_NAMES = [
    "毁灭", "巡猎", "智识", "同谐", "欢愉", "繁育",
    "存护", "丰饶", "虚无", "记忆", "均衡", "纯美"
]

# 索引字典
path_idx = {name: i for i, name in enumerate(PATH_NAMES)}

# 交互矩阵和关系矩阵定义了命途间的固定关系
PATH_INTERACTION_MATRIX = np.ones((len(PATH_NAMES), len(PATH_NAMES)))
PATH_INTERACTION_MATRIX[path_idx["毁灭"], path_idx["存护"]] = 0.7
PATH_INTERACTION_MATRIX[path_idx["存护"], path_idx["毁灭"]] = 0.7
PATH_INTERACTION_MATRIX[path_idx["巡猎"], path_idx["繁育"]] = 0.8
PATH_INTERACTION_MATRIX[path_idx["繁育"], path_idx["巡猎"]] = 0.8
PATH_INTERACTION_MATRIX[path_idx["欢愉"], path_idx["记忆"]] = 0.9
PATH_INTERACTION_MATRIX[path_idx["记忆"], path_idx["欢愉"]] = 0.9
PATH_INTERACTION_MATRIX[path_idx["智识"], path_idx["记忆"]] = 1.2
PATH_INTERACTION_MATRIX[path_idx["记忆"], path_idx["智识"]] = 1.2
PATH_INTERACTION_MATRIX[path_idx["同谐"], path_idx["欢愉"]] = 1.2
PATH_INTERACTION_MATRIX[path_idx["欢愉"], path_idx["同谐"]] = 1.2

PATH_RELATIONSHIP_MATRIX = np.full((len(PATH_NAMES), len(PATH_NAMES)), "MENTORSHIP", dtype=object)
for i in range(len(PATH_NAMES)):
    PATH_RELATIONSHIP_MATRIX[i, i] = "SYNERGY"
clash_pairs = [("毁灭", "存护"), ("巡猎", "繁育"), ("毁灭", "丰饶"), ("虚无", "丰饶")]
for p1, p2 in clash_pairs:
    PATH_RELATIONSHIP_MATRIX[path_idx[p1], path_idx[p2]] = "CLASH"
    PATH_RELATIONSHIP_MATRIX[path_idx[p2], path_idx[p1]] = "CLASH"
synergy_pairs = [("同谐", "欢愉"), ("智识", "记忆")]
for p1, p2 in synergy_pairs:
    PATH_RELATIONSHIP_MATRIX[path_idx[p1], path_idx[p2]] = "SYNERGY"
    PATH_RELATIONSHIP_MATRIX[path_idx[p2], path_idx[p1]] = "SYNERGY"
for p_name in PATH_NAMES:
    if p_name not in ["毁灭", "虚无"]:
        PATH_RELATIONSHIP_MATRIX[path_idx["同谐"], path_idx[p_name]] = "SYNERGY"
        PATH_RELATIONSHIP_MATRIX[path_idx[p_name], path_idx["同谐"]] = "SYNERGY"
repulsion_pairs = [("纯美", "毁灭"), ("纯美", "繁育")]
for p1, p2 in repulsion_pairs:
    PATH_RELATIONSHIP_MATRIX[path_idx[p1], path_idx[p2]] = "REPULSION"
    PATH_RELATIONSHIP_MATRIX[path_idx[p2], path_idx[p1]] = "REPULSION"

# 翁法罗斯事件基础概率
AEONIC_EVENT_PROBABILITY = 0.04

# 访客事件基础概率
VISITOR_EVENT_PROBABILITY = 0.02

# 投票权重配置
VOTE_WEIGHTS = {
    "Mortal": 1.0,         # 普通实体的基础投票权重
    "GoldenOne": 5.0,      # 黄金裔
    "Reincarnator": 8.0    # 白厄
}

# 思潮更新的学习率
ZEITGEIST_UPDATE_RATE = 0.05 

# 长周期停滞检测的代数阈值
LONG_TERM_STAGNATION_THRESHOLD = 1024

# 白厄专属停滞检测的代数阈值
# 想吃小白，喜欢。
# 蓝的要珍惜，红的别浪费！
BAIE_STAGNATION_THRESHOLD = 128
