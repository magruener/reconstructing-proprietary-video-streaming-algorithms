import logging

import numpy as np
import pandas as pd

from SimulationEnviroment.SimulatorEnviroment import OfflineStreaming, Trajectory, StreamingSessionEvaluation

LOGGING_LEVEL = logging.WARNING

handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(cc_session_identifier)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


class LegacyTransformer(OfflineStreaming):
    def __init__(self, bw_trace_file, fps_smoothing_s, max_lookback, max_lookahead,
                 max_switch_allowed, video_information_csv_path, reward_function, buffer_threshold_ms=60.0 * 1000.):
        """
        Maps measurement to our simulator. Choose the best possible action within your action limitation which corresponds
        clostest to the actual quality taken. **Does** rely on html5 player measurements
        :param bw_trace_file:
        :param fps_smoothing_s: @Deprecated
        :param max_lookback: How many steps do we consider in the past
        :param max_lookahead: How many steps into the future do we plan
        :param max_switch_allowed: How many levels can we change in one go
        :param video_information_csv_path: where does the video information reside
        :param reward_function: which reward function are we judging observations by. Look at SimulationEnviroment/Rewards.py
        :param buffer_threshold_ms: How large can the buffer be maximum
        """

        super().__init__(bw_trace_file,
                         video_information_csv_path,
                         reward_function, max_lookback, max_lookahead,
                         max_switch_allowed, buffer_threshold_ms=buffer_threshold_ms)
        self.fps_smoothing_s = fps_smoothing_s
        self.quality_change_arr = np.arange(-max_switch_allowed, max_switch_allowed + 1)

    def map_resolution(self, videoWidth, videoHeight):
        n_pixel_search = videoHeight * videoWidth
        return self.map_resolution_n_pixels(n_pixel_search)

    def map_experimental_fps_avail_fps(self, experimental_fps):
        min_idx = np.argmin(np.abs(self.fps_avail_arr - experimental_fps))
        return self.fps_avail_arr[min_idx]

    def map_resolution_n_pixels(self, n_pixel_search):
        return np.argmin(np.abs(self.n_pixel_arr - n_pixel_search))

    def map_switch_idx(self, switch: int):
        return list(self.quality_change_arr).index(switch)

    def reset(self):
        super().reset()

    def transform_csv(self, to_transform_path):
        """
        Main transformation method
        :param to_transform_path:
        :return:
        """
        self.reset()
        trace_video_pair_name = to_transform_path.split('/')[-2]
        client_logger_file = to_transform_path.replace('raw_dataframe.csv', 'local_client_state_logger.csv')
        client_logger_file = pd.read_csv(client_logger_file, index_col=0)
        client_logger_file['buffer_level'] = client_logger_file['buffered_until'] - client_logger_file['played_until']

        client_logger_file['time_elapsed'] = pd.to_timedelta(
            client_logger_file.timestamp_s - client_logger_file.timestamp_s.iloc[0], unit='s')
        client_logger_file['fps'] = client_logger_file.decodedFrames.diff()
        decoded_frame_unit = client_logger_file[['time_elapsed', 'fps']]
        client_logger_file = client_logger_file.drop('fps', axis=1)
        decoded_frame_unit = decoded_frame_unit.set_index('time_elapsed')
        decoded_frame_unit = decoded_frame_unit.resample('%ds' % self.fps_smoothing_s).sum() / float(
            self.fps_smoothing_s)
        decoded_frame_unit['fps'] = decoded_frame_unit['fps'].map(self.map_experimental_fps_avail_fps)
        decoded_frame_unit = decoded_frame_unit.reset_index()
        client_logger_file = pd.merge_asof(client_logger_file, decoded_frame_unit, on='time_elapsed', direction=
        'nearest')

        client_logger_file['current_level'] = (
                client_logger_file['videoWidth'] * client_logger_file['videoHeight'] * client_logger_file['fps']).map(
            self.map_resolution_n_pixels)
        client_logger_file = client_logger_file.sort_values('timestamp_s')
        ### We have a quality level progression which we need to simulate
        quality_level_progression_list = []
        insertion_points = (self.video_information_csv.time_s - self.video_information_csv.seg_len_s).values
        for n_segment, points in enumerate(insertion_points):
            client_logger_index = np.searchsorted(client_logger_file['played_until'], points)
            if points >= client_logger_file['played_until'].iloc[-1]:
                break
            quality_level_progression_list.append(client_logger_file['current_level'].iloc[client_logger_index])

        trajectory_object = Trajectory()
        trajectory_object.new_trace_video_pair_name(trace_video_pair_name)
        previous_quality = 0
        current_quality = 0
        previous_observation = self.generate_observation_dictionary()
        previous_likelihood = np.zeros((1, len(self.quality_change_arr)))
        previous_likelihood[0, len(previous_likelihood) // 2 + 1] = 1.
        del previous_observation['streaming_environment']  # We can't make use of this in the trajectory
        video_finished = False
        quality_iteration_idx = 0
        logging_list = []
        while not video_finished:
            observation, reward, video_finished, info = self.get_video_chunk(
                current_quality)
            switch = current_quality - previous_quality
            action_idx = self.map_switch_idx(switch)
            previous_quality = current_quality
            if video_finished or quality_iteration_idx >= len(quality_level_progression_list):
                # Add the last observation
                del observation['streaming_environment']  # We can't make use of this in the trajectory
                trajectory_object.add_trajectory_triple(previous_observation, observation,
                                                        action_idx)
                trajectory_object.add_likelihood(previous_likelihood, False)
                break
            # We actually need to map this
            # which is the closest quality to the played we can reach
            current_quality = quality_level_progression_list[quality_iteration_idx]
            # Should we actually map this. I probaly should map the quality progression to the closes mapping of the quality switches
            switch = int(np.clip(current_quality - previous_quality, a_min=-self.max_switch_allowed,
                                 a_max=self.max_switch_allowed))
            current_quality = previous_quality + switch
            del observation['streaming_environment']  # We can't make use of this in the trajectory
            trajectory_object.add_trajectory_triple(previous_observation, observation,
                                                    action_idx)
            trajectory_object.add_likelihood(previous_likelihood, False)
            previous_likelihood = np.zeros((1, len(self.quality_change_arr)))
            previous_likelihood[0, action_idx] = 1.
            previous_observation = observation
            quality_iteration_idx += 1

        streaming_session_evaluation = pd.DataFrame(self.return_log_state(),
                                                    columns=self.get_logging_columns())
        logging_list.append(StreamingSessionEvaluation(streaming_session_evaluation=streaming_session_evaluation,
                                                       name=trace_video_pair_name,
                                                       max_buffer_length_s=self.buffer_threshold_ms / 1000.,
                                                       max_switch_allowed=self.max_switch_allowed))

        return logging_list, trajectory_object


class EvaluationTransformer(LegacyTransformer):
    """
    Maps measurement to our simulator. Choose the best possible action within your action limitation which corresponds
    clostest to the actual quality taken. Doesn't rely on html5 player measurements
    """
    def transform_csv(self, to_transform_path):
        self.reset()
        trace_video_pair_name = to_transform_path.split('/')[-1]
        evaluation_dataframe = pd.read_csv(to_transform_path, index_col=0)
        quality_level_progression_list = evaluation_dataframe['quality_level_chosen'].values
        trajectory_object = Trajectory()
        trajectory_object.new_trace_video_pair_name(trace_video_pair_name)
        previous_quality = 0
        current_quality = 0
        previous_observation = self.generate_observation_dictionary()
        previous_likelihood = np.zeros((1, len(self.quality_change_arr)))
        previous_likelihood[0, len(previous_likelihood) // 2 + 1] = 1.
        del previous_observation['streaming_environment']  # We can't make use of this in the trajectory
        video_finished = False
        quality_iteration_idx = 0
        logging_list = []
        while not video_finished:
            observation, reward, video_finished, info = self.get_video_chunk(
                current_quality)
            switch = current_quality - previous_quality
            action_idx = self.map_switch_idx(switch)
            previous_quality = current_quality
            if video_finished or quality_iteration_idx >= len(quality_level_progression_list):
                # Add the last observation
                del observation['streaming_environment']  # We can't make use of this in the trajectory
                trajectory_object.add_trajectory_triple(previous_observation, observation,
                                                        action_idx)
                trajectory_object.add_likelihood(previous_likelihood, False)
                break
            current_quality = quality_level_progression_list[quality_iteration_idx]
            # Should we actually map this. I probaly should map the quality progression to the closes mapping of the quality switches
            switch = int(np.clip(current_quality - previous_quality, a_min=-self.max_switch_allowed,
                                 a_max=self.max_switch_allowed))
            current_quality = previous_quality + switch
            del observation['streaming_environment']  # We can't make use of this in the trajectory
            trajectory_object.add_trajectory_triple(previous_observation, observation,
                                                    action_idx)
            trajectory_object.add_likelihood(previous_likelihood, False)
            previous_likelihood = np.zeros((1, len(self.quality_change_arr)))
            previous_likelihood[0, action_idx] = 1.
            previous_observation = observation
            quality_iteration_idx += 1

        streaming_session_evaluation = pd.DataFrame(self.return_log_state(),
                                                    columns=self.get_logging_columns())
        logging_list.append(StreamingSessionEvaluation(streaming_session_evaluation=streaming_session_evaluation,
                                                       name=trace_video_pair_name,
                                                       max_buffer_length_s=self.buffer_threshold_ms / 1000.,
                                                       max_switch_allowed=self.max_switch_allowed))

        return logging_list, trajectory_object
