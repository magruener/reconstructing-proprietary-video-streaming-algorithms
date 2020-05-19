import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import clear_session

from ABRPolicies.ABRPolicy import ABRPolicy
from ABRPolicies.PensieveMultiVideo.multi_a3c import MultiActorNetwork
from ABRPolicies.PensieveSingleVideo.single_a3c import SingleActorNetwork
from ABRPolicies.ThroughputEstimator import ThroughputEstimator

LOGGING_LEVEL = logging.DEBUG
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


class PensieveMultiNN(ABRPolicy):
    def __init__(self, abr_name, nn_model_path, rate_choosen, max_quality_change, deterministic,
                 buffer_norm_factor=10.):
        """
        Code adapted for our framework from https://github.com/hongzimao/pensieve, can handle different bitrate ladders
        :param abr_name:
        :param nn_model_path: Where's the neuronal network saved
        :param rate_choosen:
        :param max_quality_change: What are the quality changes which are allowed  1 -> we can only change one level of quality
        :param deterministic: Unused
        :param buffer_norm_factor: buffer normalisation value, choosen as in the original implementation
        """

        super().__init__(abr_name, max_quality_change, deterministic)
        self.rate_choosen = rate_choosen
        clear_session()
        self.buffer_norm_factor = buffer_norm_factor
        self.info_dimensions = 7
        self.lookback_frames = 10
        self.nn_model_path = nn_model_path
        self.action_dimension = len(rate_choosen)
        action_vec = np.zeros(self.action_dimension)
        action_vec[0] = 1
        self.s_batch = [np.zeros((self.info_dimensions, self.lookback_frames))]
        self.a_batch = [action_vec]
        sess = tf.Session()
        actor_learning_rate = 0.0001
        critic_learning_rate = 0.001

        self.actor = MultiActorNetwork(sess, state_dim=[self.info_dimensions, self.lookback_frames],
                                       action_dim=self.action_dimension,
                                       learning_rate=actor_learning_rate)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters
        logger.info('Loading Model from %s' % nn_model_path)
        saver.restore(sess, nn_model_path)

    def copy(self):
        return PensieveMultiNN(self.abr_name,
                               self.nn_model_path,
                               self.rate_choosen,
                               self.max_quality_change,
                               self.deterministic,
                               self.buffer_norm_factor)

    def reset(self):
        super().reset()
        action_vec = np.zeros(self.action_dimension)
        action_vec[0] = 1
        self.s_batch = [np.zeros((self.info_dimensions, self.lookback_frames))]
        self.a_batch = [action_vec]

    def bitrate_to_action(self, bitrate, mask):
        cumsum_mask = np.cumsum(mask) - 1
        action = np.where(cumsum_mask == bitrate)[0][0]
        return action

    def next_quality(self, observation, reward):
        """
        :param observation: dictionary containing the current state of the environment over time
        measurement : [... measurement_t-2,measurement_t-1,measurement_t0]
        :param reward: reward associated by the environment with the state
        :return:
        """
        current_level = observation['current_level'][-1]
        streaming_enviroment = observation['streaming_environment']
        video_chunk_size_byte = observation['video_chunk_size_byte'][-1]
        buffer_size_s = observation['buffer_size_s'][-1]
        download_time_s = observation['download_time_s'][-1]
        download_time_ms = download_time_s * 1000.

        relative_chunk_remain = observation['relative_chunk_remain'][-1]
        kbit_rate = streaming_enviroment.get_encoded_bitrate(current_level) / 1000.
        kbit_rate_max = streaming_enviroment.get_encoded_bitrate(streaming_enviroment.max_quality_level) / 1000.
        next_chunk_mbyte_size = []
        for next_quality in range(streaming_enviroment.max_quality_level + 1):
            next_chunk_mbyte_size.append(streaming_enviroment.byte_size_match.iloc[
                                             streaming_enviroment.video_chunk_counter, next_quality] / 1000. / 1000.)
        next_chunk_mbyte_size = np.array(next_chunk_mbyte_size)
        next_chunk_kbit_rate = []
        for next_quality in range(streaming_enviroment.max_quality_level + 1):
            next_chunk_kbit_rate.append(streaming_enviroment.bitrate_match.iloc[
                                            streaming_enviroment.video_chunk_counter, next_quality] / 1000.)
        next_chunk_kbit_rate = np.array(next_chunk_mbyte_size)

        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((self.info_dimensions, self.lookback_frames))]
        else:
            state = np.array(self.s_batch[-1], copy=True)
        # dequeue discriminator_history record
        state = np.roll(state, -1, axis=1)

        state[0, -1] = kbit_rate / kbit_rate_max
        state[1, -1] = buffer_size_s / self.buffer_norm_factor  # 10 sec
        state[2, -1] = float(video_chunk_size_byte) / download_time_ms / 1000.  # kilo byte / ms
        state[3, -1] = download_time_ms / 1000. / self.buffer_norm_factor  # 10 sec
        state[4, -1] = relative_chunk_remain
        state[5, :] = -1

        indices = [np.argmin(np.abs(self.rate_choosen - v)) for v in next_chunk_kbit_rate]
        encountered = set()
        for idx, i in enumerate(indices):
            while indices[idx] in encountered:
                indices[idx] += 1
            encountered.add(indices[idx])
        indices = np.array(indices)
        state[5, indices] = next_chunk_mbyte_size
        mask = np.zeros(len(self.rate_choosen))
        mask[indices] = 1.0
        state[6, -len(self.rate_choosen):] = mask

        action_prob = self.actor.predict(np.reshape(state, (1, self.info_dimensions, self.lookback_frames)))[0]
        # the action probability should correspond to number of bit rates
        assert len(action_prob) == np.sum(mask)
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(1, 1000) / float(1000)).argmax()
        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states
        next_quality = self.bitrate_to_action(bit_rate, mask)
        # We need to reshape this in order to fit our framework
        action_prob_transformed = [action_prob.flatten()[current_level]]
        for i in range(1, self.max_quality_change + 1):
            lower_index = max(current_level - i, 0)
            upper_index = min(current_level + i, streaming_enviroment.max_quality_level)
            action_prob_transformed = [action_prob[lower_index]] + action_prob_transformed + [action_prob[upper_index]]
        action_prob_transformed = np.array(action_prob_transformed)
        action_prob_transformed = self.softmax(action_prob_transformed)
        self.likelihood_last_decision_val = action_prob_transformed.reshape(1, -1)

        next_quality = np.clip(next_quality, a_min=current_level - self.max_quality_change,
                               a_max=current_level + self.max_quality_change)
        self.s_batch.append(state)
        return int(next_quality)


class PensieveNN(ABRPolicy):

    def __init__(self, abr_name, nn_model_path, max_quality_change, deterministic, buffer_norm_factor=10.):
        """
        Code adapted for our framework from https://github.com/hongzimao/pensieve, can handle only one bitrate ladder
        :param abr_name:
        :param nn_model_path: Where's the neuronal network saved
        :param rate_choosen:
        :param max_quality_change: What are the quality changes which are allowed  1 -> we can only change one level of quality
        :param deterministic: Unused
        :param buffer_norm_factor: buffer normalisation value, choosen as in the original implementation
        """

        super().__init__(abr_name, max_quality_change, deterministic)
        clear_session()
        self.buffer_norm_factor = buffer_norm_factor
        self.info_dimensions = 6
        self.lookback_frames = 8
        self.action_dimension = 6
        self.nn_model_path = nn_model_path
        action_vec = np.zeros(self.action_dimension)
        action_vec[0] = 1
        self.s_batch = [np.zeros((self.info_dimensions, self.lookback_frames))]
        self.a_batch = [action_vec]
        sess = tf.Session()
        actor_learning_rate = 0.0001
        critic_learning_rate = 0.001
        self.actor = SingleActorNetwork(sess,
                                        state_dim=[self.info_dimensions, self.lookback_frames],
                                        action_dim=self.action_dimension,
                                        learning_rate=actor_learning_rate)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters
        logger.info('Loading Model from %s' % nn_model_path)
        saver.restore(sess, nn_model_path)

    def copy(self):
        return PensieveNN(self.abr_name,
                          self.nn_model_path,
                          self.max_quality_change,
                          self.deterministic,
                          self.buffer_norm_factor)

    def reset(self):
        super().reset()
        action_vec = np.zeros(self.action_dimension)
        action_vec[0] = 1
        self.s_batch = [np.zeros((self.info_dimensions, self.lookback_frames))]
        self.a_batch = [action_vec]

    def next_quality(self, observation, reward):
        """
        :param observation: dictionary containing the current state of the environment over time
        measurement : [... measurement_t-2,measurement_t-1,measurement_t0]
        :param reward: reward associated by the environment with the state
        :return:
        """
        current_level = observation['current_level'][-1]
        streaming_enviroment = observation['streaming_environment']
        video_chunk_size_byte = observation['video_chunk_size_byte'][-1]
        buffer_size_s = observation['buffer_size_s'][-1]
        download_time_s = observation['download_time_s'][-1]
        download_time_ms = download_time_s * 1000.

        relative_chunk_remain = observation['relative_chunk_remain'][-1]
        kbit_rate = streaming_enviroment.get_encoded_bitrate(current_level) / 1000.
        kbit_rate_max = streaming_enviroment.get_encoded_bitrate(streaming_enviroment.max_quality_level) / 1000.
        next_video_chunk_sizes = []
        for next_quality in range(streaming_enviroment.max_quality_level + 1):
            next_video_chunk_sizes.append(streaming_enviroment.byte_size_match.iloc[
                                              streaming_enviroment.video_chunk_counter, next_quality] / 1000. / 1000.)
        next_video_chunk_sizes = np.array(next_video_chunk_sizes)
        # retrieve previous state
        if len(self.s_batch) == 0:
            state = [np.zeros((self.info_dimensions, self.lookback_frames))]
        else:
            state = np.array(self.s_batch[-1], copy=True)
        # dequeue discriminator_history record
        state = np.roll(state, -1, axis=1)

        # this should be info_dimensions number of terms
        state[0, -1] = kbit_rate / kbit_rate_max  # last quality
        state[1, -1] = buffer_size_s / self.buffer_norm_factor  # 10 sec
        state[2, -1] = float(video_chunk_size_byte) / download_time_ms / 1000.  # kilo byte / ms
        state[3, -1] = download_time_ms / 1000. / self.buffer_norm_factor  # 10 sec
        state[4, :self.action_dimension] = next_video_chunk_sizes
        state[5, -1] = relative_chunk_remain
        action_prob = self.actor.predict(np.reshape(state, (1, self.info_dimensions, self.lookback_frames))).flatten()
        # We need to reshape this in order to fit our framework
        action_prob_transformed = [action_prob[current_level]]
        for i in range(1, self.max_quality_change + 1):
            lower_index = max(current_level - i, 0)
            upper_index = min(current_level + i, streaming_enviroment.max_quality_level)
            action_prob_transformed = [action_prob[lower_index]] + action_prob_transformed + [action_prob[upper_index]]
        action_prob_transformed = np.array(action_prob_transformed)
        action_prob_transformed = self.softmax(action_prob_transformed)
        self.likelihood_last_decision_val = action_prob_transformed.reshape(1, -1)
        if self.deterministic:
            next_quality = action_prob.argmax()
        else:
            probability = np.array(action_prob)
            next_quality = np.random.choice(np.arange(len(probability)), size=1, p=probability)

        next_quality = np.clip(next_quality, a_min=current_level - self.max_quality_change,
                               a_max=current_level + self.max_quality_change)
        self.s_batch.append(state)
        return int(next_quality)


class MPC(ABRPolicy):

    def likelihood_last_decision(self):
        return self.likelihood_last_decision_val

    def __init__(self, abr_name, upscale_factor, downscale_factor,
                 throughput_predictor: ThroughputEstimator, max_quality_change, lookahead=5
                 , deterministic=True, approximate=False):

        """
        Slightly adapted version of the algorithm found in https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p325.pdf
        :param abr_name: 
        :param upscale_factor: If we want upgrade we overestimate the downloadtime by factor x as a safety measure
        :param downscale_factor: If we want downgrade we underestimate the downloadtime by factor x as a safety measure
        :param throughput_predictor: Which throughput predictor are we using for the
        :param max_quality_change: What are the quality changes which are allowed  1 -> we can only change one level of quality
        :param lookahead: How far do we plan ahead
        :param deterministic: not relevant here
        :param approximate: not relevant here
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.approximate = approximate
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.throughput_predictor = throughput_predictor
        self.lookahead = lookahead
        self.lookahead_dict = {}

    def copy(self):
        return MPC(self.abr_name, self.upscale_factor, self.downscale_factor,
                   self.throughput_predictor.copy(), self.max_quality_change, self.lookahead,
                   self.deterministic, self.approximate)

    def reset(self):
        super().reset()
        self.throughput_predictor.reset()
        self.lookahead_dict = {}

    def next_quality(self, observation, reward):
        """
        :param observation: dictionary containing the current state of the environment over time
        measurement : [... measurement_t-2,measurement_t-1,measurement_t0]
        :param reward: reward associated by the environment with the state
        :return:
        """
        current_level = observation['current_level'][-1]
        streaming_enviroment = observation['streaming_environment']
        video_chunk_size_byte = observation['video_chunk_size_byte'][-1]
        timestamp_s = observation['timestamp_s'][-1]
        download_time_s = observation['download_time_s'][-1]
        buffer_size_s = observation['buffer_size_s'][-1]
        data_used_bytes_relative = observation['data_used_bytes_relative'][-1]
        tput_estimate = (8e-6 * video_chunk_size_byte) / download_time_s
        self.throughput_predictor.add_sample(tput_estimate=tput_estimate, timestamp_s=timestamp_s)
        future_bandwidth = self.throughput_predictor.predict_future_bandwidth()
        value_next_quality, next_quality, probability_next_quality = self.solve_lookahead(streaming_enviroment,
                                                                                          self.lookahead,
                                                                                          video_chunk_counter=streaming_enviroment.video_chunk_counter,
                                                                                          last_level=current_level,
                                                                                          future_bandwidth=future_bandwidth,
                                                                                          buffer_size_s=buffer_size_s,
                                                                                          data_used_bytes_relative=data_used_bytes_relative)
        # print('Predict next quality %d likelihood %s' % (next_quality,str(probability_next_quality)))
        # print('Solving for the next quality took %.2f, %d to go' % (time() - start_time))
        self.likelihood_last_decision_val = probability_next_quality
        return int(next_quality)

    def solve_lookahead(self, streaming_enviroment, lookahead_to_go, video_chunk_counter, last_level, future_bandwidth,
                        buffer_size_s, data_used_bytes_relative):
        """
        Recursive lookahead solver for the MPC planning step
        :param streaming_enviroment:
        :param lookahead_to_go: how far ahead do we still have to look
        :param video_chunk_counter: Which chunk are we looking at
        :param last_level: What was the last abr quality level
        :param future_bandwidth: What is the future bandwidth estimate in Mbps
        :param buffer_size_s: Current buffer size in s
        :param data_used_bytes_relative: relative to what we had to sue if we were to download at the highest quality how much have we used
        :return:
        """
        dynamic_prog_key = '%d_%d_%d_%.1f_%d' % (
            lookahead_to_go, video_chunk_counter, last_level, future_bandwidth, int(buffer_size_s))
        if dynamic_prog_key in self.lookahead_dict:
            return self.lookahead_dict[dynamic_prog_key]
        if lookahead_to_go == 0:
            return 0, 0, 0
        if video_chunk_counter >= len(streaming_enviroment.byte_size_match):
            return 0, 0, 0
        # Set possible quality shifts
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        single_bitrate_list = []
        level_looked_at = []

        for next_level in quality_choices:
            if next_level in level_looked_at:
                # Reuse the last calculated current_reward and future_reward
                # three levels max 2 changes [calc,take_last,calculate,calculate,take_last]
                reward_list.append((next_level, current_reward + future_reward))
                continue
            level_looked_at.append(next_level)

            streaming_enviroment_state_save = streaming_enviroment.save_state()

            video_chunk_size_byte = streaming_enviroment.byte_size_match.iloc[
                video_chunk_counter, next_level]
            encoded_mbitrate = streaming_enviroment.get_encoded_bitrate(next_level) * 1e-6
            current_mbitrate = streaming_enviroment.bitrate_match.iloc[
                                   video_chunk_counter, next_level] * 1e-6
            vmaf = streaming_enviroment.vmaf_match.iloc[video_chunk_counter, next_level]
            size_mbit = 8e-6 * video_chunk_size_byte
            if next_level > last_level:
                download_time_s = size_mbit / (
                        future_bandwidth * self.upscale_factor)  # Used in the Shaka Player
            elif next_level == last_level:
                download_time_s = size_mbit / future_bandwidth  # No Scaling factor I guess
            else:
                download_time_s = size_mbit / (
                        future_bandwidth * self.downscale_factor)  # Used in the Shaka Player
            buffer_size_new = buffer_size_s - download_time_s
            rebuffer_level_s = 0
            if buffer_size_new < 0:
                rebuffer_level_s = np.abs(buffer_size_new)
                buffer_size_new = 0
            single_bitrate_list.append(current_mbitrate)
            buffer_size_new += streaming_enviroment.video_information_csv.seg_len_s[
                video_chunk_counter]
            data_used_bytes_new = data_used_bytes_relative + video_chunk_size_byte
            ####################################################################
            streaming_enviroment.data_used_relative.append(data_used_bytes_new)
            streaming_enviroment.quality_arr.append(next_level)
            streaming_enviroment.buffer_size_arr.append(buffer_size_new)
            streaming_enviroment.encoded_mbitrate_arr.append(encoded_mbitrate)
            streaming_enviroment.single_mbitrate_arr.append(current_mbitrate)
            streaming_enviroment.vmaf_arr.append(vmaf)
            streaming_enviroment.rebuffer_time_arr.append(rebuffer_level_s)
            ####################################################################
            observation = streaming_enviroment.generate_observation_dictionary()
            assert observation['current_level'][-1] == next_level, "Adding to the array didn't go as planned"
            assert observation['rebuffer_time_s'][-1] == rebuffer_level_s, "Adding to the array didn't go as planned"
            current_reward = streaming_enviroment.reward_function.return_reward_observation(observation)
            future_reward, _, _ = self.solve_lookahead(streaming_enviroment,
                                                       video_chunk_counter=video_chunk_counter + 1,
                                                       lookahead_to_go=lookahead_to_go - 1,
                                                       last_level=next_level,
                                                       future_bandwidth=future_bandwidth,
                                                       buffer_size_s=buffer_size_new,
                                                       data_used_bytes_relative=data_used_bytes_new)
            reward_list.append((next_level, current_reward + future_reward))
            streaming_enviroment.set_state(streaming_enviroment_state_save)
        self.lookahead_dict[dynamic_prog_key] = self.select_reward_to_propagate(reward_list)
        #print(reward_list,lookahead_to_go)
        # print('\t' * (2 - lookahead_to_go) +'last quality : %d' % last_level + 'reward list : '+ str(reward_list) + ' bitrate list ' + str(single_bitrate_list))
        return self.lookahead_dict[dynamic_prog_key]
