# parliament_manager.py
import numpy as np
from constants import PATH_NAMES

class ParliamentManager:
    def __init__(self):
        self.seats = {name: 0 for name in PATH_NAMES}
        self.total_seats = 100

    def hold_election(self, population: list):
        """根据所有实体的投票，分配议会席位。"""
        if not population:
            return

        total_votes = np.zeros(len(PATH_NAMES))
        for entity in population:
            vote_proposal = entity.generate_vote_proposal()
            vote_weight = entity.get_vote_weight() * entity.score
            total_votes += vote_proposal * vote_weight

        if np.sum(total_votes) == 0:
            # 如果没有有效投票，平均分配
            seats_per_path = self.total_seats // len(PATH_NAMES)
            for name in PATH_NAMES:
                self.seats[name] = seats_per_path
            return

        # 根据票数比例分配席位
        vote_distribution = total_votes / np.sum(total_votes)
        allocated_seats = (vote_distribution * self.total_seats).astype(int)

        # 处理取整问题
        remainder = self.total_seats - np.sum(allocated_seats)
        if remainder > 0:
            top_indices = np.argsort(vote_distribution)[-remainder:]
            allocated_seats[top_indices] += 1
        
        for i, name in enumerate(PATH_NAMES):
            self.seats[name] = allocated_seats[i]
        
        print("\n\033[34m--- 公民议会换届 ---")
        dominant_party = max(self.seats, key=self.seats.get)
        print(f"本世代议会已组成，命途主导 '{dominant_party}' (席位: {self.seats[dominant_party]}/{self.total_seats})")
        print("--------------------------\033[0m")

    def get_zeitgeist_multiplier(self, entity_path_affinities: np.ndarray) -> float:
        """根据实体命途与当前议会格局，计算思潮乘数。"""
        if np.sum(list(self.seats.values())) == 0:
            return 1.0
        seat_vector = np.array([self.seats[name] for name in PATH_NAMES])
        normalized_seat_vector = seat_vector / self.total_seats
        alignment_score = np.dot(entity_path_affinities, normalized_seat_vector) - np.dot(1 - entity_path_affinities, normalized_seat_vector)
        multiplier = 1.15 + 0.35 * np.tanh(alignment_score * 3)
        return max(0.8, min(1.5, multiplier))
