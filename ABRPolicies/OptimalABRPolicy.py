import logging

import numpy as np

from ABRPolicies.ComplexABRPolicy import MPC
from ABRPolicies.ThroughputEstimator import StepEstimator

LOGGING_LEVEL = logging.INFO
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


class Optimal(MPC):

    def __init__(self, abr_name, max_quality_change,
                 lookahead=5, deterministic=True):
        """
        Derivative of MPC which simulates the future with the actual bandwidth and chooses the actions which maximise the QoE
        :param abr_name:
        :param max_quality_change:
        :param lookahead:
        :param deterministic:
        """
        throughput_predictor = StepEstimator(consider_last_n_steps=1,
                                             predictor_function=np.mean,
                                             robust_estimate=False)  # Dummy Value so we can use the rest of the function
        super().__init__(abr_name, 1.0, 1.0, throughput_predictor,
                         max_quality_change, lookahead, deterministic)

    def copy(self):
        return Optimal(self.abr_name, self.max_quality_change,
                 self.lookahead, self.deterministic)

    def solve_lookahead(self, streaming_enviroment, lookahead_to_go, video_chunk_counter, last_level, future_bandwidth,
                        buffer_size_s, data_used_bytes_relative):
        if lookahead_to_go == 0:
            return 0, 0, 0
        if video_chunk_counter >= len(streaming_enviroment.byte_size_match):
            return 0, 0, 0
        # Set possible quality shifts
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        level_looked_at = []
        for next_level in quality_choices:
            if next_level in level_looked_at :
                # Reuse the last calculated current_reward and future_reward
                # three levels max 2 changes [0,0,1,2,2]
                # three levels max 2 changes [calc,take_last,calculate,calculate,take_last]
                reward_list.append((next_level, current_reward + future_reward))
                continue
            streaming_enviroment_state_save = streaming_enviroment.save_state()
            observation, current_reward, end_of_video, info = streaming_enviroment.get_video_chunk(next_level)
            future_reward, _, _ = self.solve_lookahead(streaming_enviroment,
                                                       lookahead_to_go=lookahead_to_go - 1,
                                                       video_chunk_counter = video_chunk_counter+1,
                                                       last_level=next_level,
                                                       future_bandwidth=None,
                                                       buffer_size_s=None,
                                                       data_used_bytes_relative=None)
            reward_list.append((next_level, current_reward + future_reward))
            streaming_enviroment.set_state(streaming_enviroment_state_save)
        return self.select_reward_to_propagate(reward_list)


class Worst(Optimal):
    """
    Derivative of MPC which simulates the future with the actual bandwidth and chooses the actions which minimise the QoE
    """
    def copy(self):
        return Worst(self.abr_name, self.max_quality_change,
                 self.lookahead, self.deterministic)

    def solve_lookahead(self, streaming_enviroment, lookahead_to_go, video_chunk_counter, last_level, future_bandwidth,
                        buffer_size_s, data_used_bytes_relative):
        if lookahead_to_go == 0:
            return 0, 0, 0
        if video_chunk_counter >= len(streaming_enviroment.byte_size_match):
            return 0, 0, 0
        # Set possible quality shifts
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        level_looked_at = []
        for next_level in quality_choices:
            if next_level in level_looked_at :
                # Reuse the last calculated current_reward and future_reward
                # three levels max 2 changes [calc,take_last,calculate,calculate,take_last]
                reward_list.append((next_level, current_reward + future_reward))
                continue
            level_looked_at.append(next_level)
            streaming_enviroment_state_save = streaming_enviroment.save_state()
            observation, current_reward, end_of_video, info = streaming_enviroment.get_video_chunk(next_level)
            future_reward, _, _ = self.solve_lookahead(streaming_enviroment,
                                                       lookahead_to_go=lookahead_to_go - 1,
                                                       video_chunk_counter = video_chunk_counter+1,
                                                       last_level=next_level,
                                                       future_bandwidth=None,
                                                       buffer_size_s=None,
                                                       data_used_bytes_relative=None)
            reward_list.append((next_level, current_reward + future_reward))
            streaming_enviroment.set_state(streaming_enviroment_state_save)
        return self.select_reward_to_propagate(reward_list,inverse=True)
