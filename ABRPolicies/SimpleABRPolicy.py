import logging

import numpy as np

from ABRPolicies import ThroughputEstimator
from ABRPolicies.ABRPolicy import ABRPolicy
from ABRPolicies.ComplexABRPolicy import MPC

LOGGING_LEVEL = logging.INFO
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


class RandomPolicy(ABRPolicy):

    def __init__(self, abr_name, max_quality_change, deterministic):
        """
        Select random quality at each instance
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.likelihood_last_decision_val = np.ones(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val /= sum(self.likelihood_last_decision_val)
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def next_quality(self, observation, reward):
        streaming_enviroment = observation['streaming_enviroment']
        current_level = observation['current_level'][-1]
        switch = np.random.choice(self.quality_change_arr, 1)
        next_level = current_level + switch
        next_level = np.clip(next_level, a_min=0, a_max=streaming_enviroment.max_quality_level)[0]
        return next_level

    def reset(self):
        self.likelihood_last_decision_val = np.ones(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val /= sum(self.likelihood_last_decision_val)
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def copy(self):
        return RandomPolicy(self.abr_name, self.max_quality_change, self.deterministic)


class PerceptualMaximizer(ABRPolicy):
    def __init__(self, max_quality_change):
        """
        Always play at max quality
        :param max_quality_change:
        """
        super().__init__(abr_name='PerceptualMaximizer', max_quality_change=max_quality_change, deterministic=True)
        self.likelihood_last_decision_val = np.zeros(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val[-1] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def next_quality(self, observation, reward):
        streaming_enviroment = observation['streaming_environment']
        last_level = observation['current_level'][-1]
        return min(last_level + self.max_quality_change, streaming_enviroment.max_quality_level)

    def reset(self):
        self.likelihood_last_decision_val = np.zeros(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val[-1] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def copy(self):
        return PerceptualMaximizer(self.max_quality_change)


class RebufferingMinimizer(ABRPolicy):
    def __init__(self):
        """
        Always play at lowest quality
        """
        super().__init__(abr_name='RebufferingMaximizer', max_quality_change=1, deterministic=True)
        self.likelihood_last_decision_val = np.zeros(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val[0] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def next_quality(self, observation, reward):
        return 0

    def reset(self):
        self.likelihood_last_decision_val = np.zeros(len(self.likelihood_last_decision_val))
        self.likelihood_last_decision_val[0] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def copy(self):
        return RebufferingMinimizer()


class Rate(MPC):

    def __init__(self, abr_name, throughput_predictor: ThroughputEstimator, upscale_factor, downscale_factor,
                 max_quality_change):
        """
        Simple rate based ABR algorithm
        -> Choose highest quality which doesn't lead to buffer drain in the next step with the given throughput estimator
        :param abr_name:
        :param upscale_factor: If we want upgrade we overestimate the downloadtime by factor x as a safety measure
        :param downscale_factor: If we want downgrade we underestimate the downloadtime by factor x as a safety measure
        :param throughput_predictor: Which throughput predictor are we using for the
        :param max_quality_change: What are the quality changes which are allowed  1 -> we can only change one level of quality
        """
        super().__init__(abr_name, upscale_factor, downscale_factor, throughput_predictor,
                         max_quality_change)

    def solve_lookahead(self, streaming_enviroment, lookahead_to_go, video_chunk_counter, last_level, future_bandwidth,
                        buffer_size_s, data_used_bytes_relative):
        if lookahead_to_go == 0:
            return 0, 0, 0
        if streaming_enviroment.video_chunk_counter >= len(streaming_enviroment.byte_size_match):
            return 0, 0, 0
        # Set possible quality shifts
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        for next_level in quality_choices:
            bitrate_bit = streaming_enviroment.bitrate_match.iloc[streaming_enviroment.video_chunk_counter, next_level]
            bitrate_mbit = 1e-6 * bitrate_bit
            video_chunk_size_byte = streaming_enviroment.byte_size_match.iloc[
                streaming_enviroment.video_chunk_counter, next_level]
            if next_level > last_level:
                future_bandwidth = future_bandwidth * self.upscale_factor  # Used in the Shaka Player
            elif next_level == last_level:
                future_bandwidth = future_bandwidth  # No Scaling factor I guess
            else:
                future_bandwidth = future_bandwidth * self.downscale_factor  # Used in the Shaka Player
            if bitrate_mbit / future_bandwidth > 1.0:
                current_reward = -bitrate_mbit  # Ratio is wrong
            else:
                current_reward = bitrate_mbit
            reward_list.append((next_level, current_reward))
        return self.select_reward_to_propagate(reward_list)

    def copy(self):
        return Rate(self.abr_name, self.throughput_predictor.copy(), self.upscale_factor, self.downscale_factor,
                    self.max_quality_change)


class BBA(ABRPolicy):
    def __init__(self, abr_name, cushion, reservoir, max_quality_change, deterministic):
        """
        Simple buffer based algorithm inspired by http://yuba.stanford.edu/~nickm/papers/sigcomm2014-video.pdf
        :param abr_name:
        :param cushion:
        :param reservoir:
        :param max_quality_change:
        :param deterministic:
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.cushion = cushion
        self.reservoir = reservoir

    def solve_lookahead(self, streaming_enviroment, lookahead_to_go, video_chunk_counter, last_level, future_bandwidth,
                        buffer_size_s, data_used_bytes_relative):
        buffer_level_available = np.linspace(self.reservoir, self.cushion, num=streaming_enviroment.max_quality_level)
        if lookahead_to_go == 0:
            return 0, 0, 0
        if video_chunk_counter >= len(streaming_enviroment.byte_size_match):
            return 0, 0, 0
        # Set possible quality shifts
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        for next_level in quality_choices:
            if buffer_size_s < buffer_level_available[quality_choices]:  # Ratio is wrong
                current_reward = -next_level
            else:
                current_reward = next_level
            reward_list.append((next_level, current_reward))
        return self.select_reward_to_propagate(reward_list)

    def copy(self):
        return BBA(self.abr_name, self.cushion, self.reservoir, self.max_quality_change, self.deterministic)


class DownloadTime(MPC):

    def __init__(self, abr_name, throughput_predictor: ThroughputEstimator, upscale_factor,
                 downscale_factor, max_quality_change, reservoir):
        """
        Simple downloadtime based algorithm
            -> Choose highest quality which doesn't lead to rebuffering in the next step with the given throughput estimator
        :param abr_name:
        :param throughput_predictor:
        :param upscale_factor:
        :param downscale_factor:
        :param max_quality_change:
        :param reservoir:
        """
        super().__init__(abr_name, upscale_factor, downscale_factor, throughput_predictor,
                         max_quality_change)
        self.reservoir = reservoir

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
        for next_level in quality_choices:
            bitrate_bit = streaming_enviroment.bitrate_match.iloc[streaming_enviroment.video_chunk_counter, next_level]
            bitrate_mbit = 1e-6 * bitrate_bit
            video_chunk_size_byte = streaming_enviroment.byte_size_match.iloc[
                streaming_enviroment.video_chunk_counter, next_level]
            size_mbit = 8e-6 * video_chunk_size_byte
            if next_level > last_level:
                download_time_s = size_mbit / (
                        future_bandwidth * self.upscale_factor)  # Used in the Shaka Player
            elif next_level == last_level:
                download_time_s = size_mbit / future_bandwidth  # No Scaling factor I guess
            else:
                download_time_s = size_mbit / (
                        future_bandwidth * self.downscale_factor)  # Used in the Shaka Player

            if buffer_size_s - download_time_s < self.reservoir:  # Ratio is wrong
                current_reward = -bitrate_mbit
            else:
                current_reward = bitrate_mbit
            reward_list.append((next_level, current_reward))
        return self.select_reward_to_propagate(reward_list)

    def copy(self):
        return DownloadTime(self.abr_name, self.throughput_predictor.copy(), self.upscale_factor,
                   self.downscale_factor, self.max_quality_change, self.reservoir)
