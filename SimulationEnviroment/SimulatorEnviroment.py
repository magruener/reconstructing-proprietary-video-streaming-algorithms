"""
Taken from
https://github.com/hongzimao/pensieve

"""
import logging
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ABRPolicies.ABRPolicy import ABRPolicy

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
LOGGING_TYPES = ['physical', 'virtual']
M_IN_K = 1000.0

LOGGING_LEVEL = logging.DEBUG
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(cc_session_identifier)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)


class StreamingEnviroment:
    def __init__(self, bw_trace_file,
                 video_information_csv_path,
                 reward_function,
                 max_lookback,
                 max_lookahead,
                 max_switch_allowed,
                 buffer_threshold_ms=60.0 * MILLISECONDS_IN_SECOND):
        """
        Streaming Environment inspired by the code found in https://github.com/hongzimao/pensieve
        :param bw_trace_file:
        :param video_information_csv_path:
        :param reward_function:
        :param max_lookback:
        :param max_lookahead:
        :param max_switch_allowed:
        :param buffer_threshold_ms:
        """
        self.max_switch_allowed = max_switch_allowed
        self.max_lookahead = max_lookahead
        self.max_lookback = max_lookback
        self.video_information_csv_path = video_information_csv_path
        self.video_information_csv = None
        self.bw_trace_file = bw_trace_file
        self.max_data_used = 0
        self.max_rate_encoded = 0
        self.load_video_information_csv(video_information_csv_path)

        """
        Load trace for the video in the enviroment
        """
        self.cooked_time = []
        self.cooked_bw = []
        self.mahimahi_ptr = None
        self.last_mahimahi_time = None

        self.load_bw_trace(bw_trace_file)

        self.reward_function = reward_function
        self.buffer_threshold_ms = buffer_threshold_ms
        self.logging_file = []
        self.video_chunk_counter = 0
        self.buffer_size_ms = 0
        self.data_used_bytes = 0
        self.last_quality = 0
        self.timestamp_s = 0

    def load_bw_trace(self, bw_trace_file):
        self.cooked_time = []
        self.cooked_bw = []
        with open(self.bw_trace_file, 'rb') as f:
            for line in f:
                parse = line.split()
                self.cooked_time.append(float(parse[0]))
                self.cooked_bw.append(float(parse[1]))
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

    def copy(self):
        return StreamingEnviroment(self.bw_trace_file,
                                   self.video_information_csv_path,
                                   self.reward_function,
                                   self.max_lookback,
                                   self.max_lookahead,
                                   self.max_switch_allowed,
                                   self.buffer_threshold_ms)

    def impute_NaN_inplace(self, dataframe):
        if np.isnan(dataframe).sum().sum() != 0:
            print('Input Dataframe contains NaN Values')
            dataframe.fillna(dataframe.mean(), inplace=True)

    def load_video_information_csv(self, video_information_csv_path):
        def extract_sorted(key_str, column):
            column = list(filter(lambda c: key_str in c, column))
            column = sorted(column,
                            key=lambda c: np.array(c.split('_')[0].split('x')).astype(float).prod()
                            )
            return column

        self.video_information_csv = pd.read_csv(video_information_csv_path, index_col=0)
        self.impute_NaN_inplace(self.video_information_csv)
        self.video_information_csv['time_s'] = self.video_information_csv.seg_len_s.cumsum()
        self.byte_size_match = extract_sorted('byte', self.video_information_csv.columns)
        self.byte_size_match = self.video_information_csv[self.byte_size_match]
        self.vmaf_match = extract_sorted('vmaf', self.video_information_csv.columns)
        self.vmaf_match = self.video_information_csv[self.vmaf_match]
        self.bitrate_match = extract_sorted('bitrate', self.video_information_csv.columns)
        self.bitrate_match = self.video_information_csv[self.bitrate_match]
        self.max_quality_level = self.byte_size_match.shape[1] - 1
        self.video_duration = self.video_information_csv.seg_len_s.sum()
        self.n_video_chunk = len(self.bitrate_match)
        self.max_data_used = self.byte_size_match.sum().max()
        self.max_rate_encoded = self.bitrate_match.mean().max()
        self.n_pixel_arr = np.array(
            sorted([np.array(c.split('_')[0].split('x')).astype(float).prod() for c in self.byte_size_match.columns]))
        if len(self.byte_size_match.columns[0].split('_')[0].split('x')) > 2:  # There's fps information
            self.fps_avail_arr = np.array(
                sorted([np.array(c.split('_')[0].split('x')).astype(float)[-1] for c in self.byte_size_match.columns]))
        else:
            self.fps_avail_arr = np.array([1])

    def set_new_enviroment(self, bw_trace_file, video_information_csv_path):
        self.bw_trace_file = bw_trace_file
        self.load_bw_trace(bw_trace_file)
        self.load_video_information_csv(video_information_csv_path)
        self.reset()

    def get_encoded_bitrate(self, quality):
        return self.bitrate_match.iloc[:, quality].mean()

    def get_vmaf(self, index, quality):
        assert len(self.vmaf_match) > index, 'Index is to big %d %d' % (len(self.vmaf_match), index)
        return self.vmaf_match.iloc[index, quality]

    def get_bitrate(self, index, quality):
        assert len(self.bitrate_match) > index, 'Index is to big'
        return self.bitrate_match.iloc[index, quality]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_video_chunk(self, quality, activate_logging=True):
        pass

    @abstractmethod
    def generate_observation_dictionary(self):
        pass

    def return_log_state(self):
        return self.logging_file

    def log_state(self, current_level,
                  rebuffering_seconds, video_chunk_size_byte, download_time_s, reward):
        estimated_bandwidth_mbit = (8e-6 * video_chunk_size_byte) / download_time_s
        current_iterator = np.min([self.video_chunk_counter, self.n_video_chunk - 1])
        self.logging_file.append([self.timestamp_s,
                                  self.get_encoded_bitrate(current_level) * 1e-6,
                                  self.get_bitrate(current_iterator, current_level) * 1e-6,
                                  self.get_vmaf(current_iterator, current_level),
                                  self.buffer_size_ms / MILLISECONDS_IN_SECOND,
                                  rebuffering_seconds,
                                  video_chunk_size_byte,
                                  self.video_information_csv.iloc[current_iterator].seg_len_s,
                                  download_time_s,
                                  estimated_bandwidth_mbit,
                                  current_level,
                                  self.data_used_bytes,
                                  reward])

    def get_logging_columns(self):
        """
        Returns all columns which are logged in raw format by the environment
        :return:
        """
        return ['timestamp_s',
                'encoded_mbitrate',
                'single_mbitrate',
                'vmaf',
                'buffer_size_s',
                'rebuffering_seconds',
                'video_chunk_size_bytes',
                'segment_length_s',
                'download_time_s',
                'estimated_bandwidth_mbit',
                'current_level',
                'data_used_bytes_relative',
                'reward']

    def create_enviroment_state(self, current_level,
                                rebuffering_seconds,
                                download_time_s):
        last_iterator = np.max([self.video_chunk_counter - 1, 0])
        current_iterator = np.min([self.video_chunk_counter, self.n_video_chunk - 1])
        return {
            'last_vmaf': self.get_vmaf(last_iterator, self.last_quality),
            'current_vmaf': self.get_vmaf(current_iterator, current_level),
            'last_encoded_mbitrate': self.get_bitrate(last_iterator, self.last_quality) * 1e-6,
            'current_encoded_mbitrate': self.get_bitrate(current_iterator, current_level) * 1e-6,
            'last_single_mbitrate': self.get_bitrate(last_iterator, self.last_quality) * 1e-6,
            'current_single_mbitrate': self.get_bitrate(current_iterator, current_level) * 1e-6,
            'rebuffering_seconds': rebuffering_seconds,
            'buffer_size_s': self.buffer_size_ms / MILLISECONDS_IN_SECOND,
            'download_time_s': download_time_s,
            'segment_length_s': self.video_information_csv.iloc[current_iterator].seg_len_s,
            'data_used_bytes_relative': self.data_used_bytes,
        }


class OfflineStreaming(StreamingEnviroment):
    def __init__(self, bw_trace_file, video_information_csv_path, reward_function, max_lookback: int,
                 max_lookahead: int, max_switch_allowed: int, buffer_threshold_ms=60 * MILLISECONDS_IN_SECOND,
                 drain_buffer_sleep_ms=500.0, packet_payload_portion=0.95, link_rtt_ms=200, packet_size_byte=1500):
        super().__init__(bw_trace_file, video_information_csv_path, reward_function, max_lookback, max_lookahead,
                         max_switch_allowed, buffer_threshold_ms)
        self.drain_buffer_sleep_ms = drain_buffer_sleep_ms
        self.packet_payload_portion = packet_payload_portion
        self.link_rtt_ms = link_rtt_ms
        self.packet_size_byte = packet_size_byte

        """
        Initialise the values as zeros
        """
        self.timestamp_arr = [0] * self.max_lookback
        self.data_used_relative = [0] * self.max_lookback
        self.quality_arr = [0] * self.max_lookback
        self.downloadtime_arr = [0] * self.max_lookback
        self.sleep_time_arr = [0] * self.max_lookback
        self.buffer_size_arr = [0] * self.max_lookback
        self.rebuffer_time_arr = [0] * self.max_lookback
        self.video_chunk_remain_arr_relative = [0] * self.max_lookback
        self.video_chunk_size_arr = [0] * self.max_lookback
        self.rate_played_relative_arr = [0] * self.max_lookback
        self.segment_length_arr = [0] * self.max_lookback
        self.encoded_mbitrate_arr = [0] * self.max_lookback
        self.single_mbitrate_arr = [0] * self.max_lookback
        self.vmaf_arr = [0] * self.max_lookback

    def copy(self):
        return OfflineStreaming(self.bw_trace_file, self.video_information_csv_path, self.reward_function,
                                self.max_lookback,
                                self.max_lookahead, self.max_switch_allowed, self.buffer_threshold_ms,
                                self.drain_buffer_sleep_ms, self.packet_payload_portion, self.link_rtt_ms,
                                self.packet_size_byte)

    def set_state(self, state):
        """
        Set state from an observation
        :param state:
        :return:
        """
        self.video_chunk_counter = state['video_chunk_counter']
        self.mahimahi_ptr = state['mahimahi_ptr']
        self.buffer_size_ms = state['buffer_size_ms']
        self.last_mahimahi_time = state['last_mahimahi_time']
        self.last_quality = state['last_quality']
        self.timestamp_s = state['timestamp_s']
        self.data_used_bytes = state['data_used_bytes_relative']
        self.timestamp_arr = self.timestamp_arr[:state['timestamp_s_arr_ptr']]
        self.data_used_relative = self.data_used_relative[:state['data_used_bytes_arr_ptr']]
        self.quality_arr = self.quality_arr[:state['quality_arr_ptr']]
        self.downloadtime_arr = self.downloadtime_arr[:state['downloadtime_arr_ptr']]
        self.sleep_time_arr = self.sleep_time_arr[:state['sleep_time_arr_ptr']]
        self.buffer_size_arr = self.buffer_size_arr[:state['buffer_size_arr_ptr']]
        self.rebuffer_time_arr = self.rebuffer_time_arr[:state['rebuffer_time_arr_ptr']]
        self.video_chunk_remain_arr_relative = self.video_chunk_remain_arr_relative[:state['video_chunk_ptr']]
        self.video_chunk_size_arr = self.video_chunk_size_arr[:state['video_chunk_size_ptr']]
        self.rate_played_relative_arr = self.rate_played_relative_arr[:state['rate_played_relative_ptr']]
        self.segment_length_arr = self.segment_length_arr[:state['segment_length_ptr']]
        self.encoded_mbitrate_arr = self.encoded_mbitrate_arr[:state['encoded_mbitrate_ptr']]
        self.single_mbitrate_arr = self.single_mbitrate_arr[:state['single_mbitrate_ptr']]
        self.vmaf_arr = self.vmaf_arr[:state['vmaf_ptr']]
        assert len(self.logging_file) >= state['logging_file_ptr'], 'We somehow lost logging data on the way'
        self.logging_file = self.logging_file[:state['logging_file_ptr']]

    def save_state(self):
        return {'mahimahi_ptr': self.mahimahi_ptr,
                'buffer_size_ms': self.buffer_size_ms,
                'last_mahimahi_time': self.last_mahimahi_time,
                'last_quality': self.last_quality,
                'video_chunk_counter': self.video_chunk_counter,
                'timestamp_s': self.timestamp_s,
                'data_used_bytes_relative': self.data_used_bytes,
                'video_chunk_size_ptr': len(self.video_chunk_size_arr),
                'timestamp_s_arr_ptr': len(self.timestamp_arr),
                'data_used_bytes_arr_ptr': len(self.data_used_relative),
                'quality_arr_ptr': len(self.quality_arr),
                'downloadtime_arr_ptr': len(self.downloadtime_arr),
                'sleep_time_arr_ptr': len(self.sleep_time_arr),
                'buffer_size_arr_ptr': len(self.buffer_size_arr),
                'rebuffer_time_arr_ptr': len(self.rebuffer_time_arr),
                'video_chunk_ptr': len(self.video_chunk_remain_arr_relative),
                'logging_file_ptr': len(self.logging_file),
                'rate_played_relative_ptr': len(self.rate_played_relative_arr),
                'segment_length_ptr': len(self.segment_length_arr),
                'encoded_mbitrate_ptr': len(self.encoded_mbitrate_arr),
                'single_mbitrate_ptr': len(self.single_mbitrate_arr),
                'vmaf_ptr': len(self.vmaf_arr)}

    def get_video_chunk(self, quality, activate_logging=True):
        """
        Simulation routine
        :param quality:
        :param activate_logging:
        :return:
        """

        assert quality >= 0

        video_chunk_size = self.byte_size_match.iloc[self.video_chunk_counter, quality]
        relative_encoded_bitrate = self.get_encoded_bitrate(quality) / self.get_encoded_bitrate(-1)
        segment_length_ms = self.video_information_csv.iloc[
                                self.video_chunk_counter].seg_len_s * 1000.
        encoded_mbitrate = self.get_encoded_bitrate(quality) * 1e-6
        current_mbitrate = self.bitrate_match.iloc[self.video_chunk_counter, quality] * 1e-6
        vmaf = self.vmaf_match.iloc[self.video_chunk_counter, quality]

        downloadtime_ms = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * self.packet_payload_portion

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / self.packet_payload_portion
                downloadtime_ms += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            downloadtime_ms += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        downloadtime_ms *= MILLISECONDS_IN_SECOND
        downloadtime_ms += self.link_rtt_ms

        # rebuffer time
        rebuffer_time_ms = np.maximum(downloadtime_ms - self.buffer_size_ms, 0.0)

        # update the buffer
        self.buffer_size_ms = np.maximum(self.buffer_size_ms - downloadtime_ms, 0.0)

        # add in the new chunk

        self.buffer_size_ms += segment_length_ms  # buffer size is in ms
        # sleep if buffer gets too large
        sleep_time_ms = 0
        if self.buffer_size_ms > self.buffer_threshold_ms:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the download_time_s
            drain_buffer_time = self.buffer_size_ms - self.buffer_threshold_ms
            sleep_time_ms = np.ceil(drain_buffer_time / self.drain_buffer_sleep_ms) * \
                            self.drain_buffer_sleep_ms
            self.buffer_size_ms -= sleep_time_ms

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time_ms / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time_ms / MILLISECONDS_IN_SECOND
                    break
                sleep_time_ms -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video

        self.video_chunk_counter += 1
        video_chunk_remain = self.n_video_chunk - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.n_video_chunk:
            end_of_video = True

        self.data_used_bytes += video_chunk_size
        self.timestamp_s += downloadtime_ms / MILLISECONDS_IN_SECOND + sleep_time_ms / MILLISECONDS_IN_SECOND

        self.timestamp_arr.append(self.timestamp_s)
        self.data_used_relative.append(self.data_used_bytes / self.max_data_used)
        self.quality_arr.append(quality)
        self.downloadtime_arr.append(downloadtime_ms / MILLISECONDS_IN_SECOND)
        self.sleep_time_arr.append(sleep_time_ms / MILLISECONDS_IN_SECOND)
        self.buffer_size_arr.append(self.buffer_size_ms / MILLISECONDS_IN_SECOND)
        self.rebuffer_time_arr.append(rebuffer_time_ms / MILLISECONDS_IN_SECOND)
        self.video_chunk_size_arr.append(video_chunk_size)
        self.video_chunk_remain_arr_relative.append(video_chunk_remain / float(self.n_video_chunk))
        self.rate_played_relative_arr.append(relative_encoded_bitrate)
        self.segment_length_arr.append(segment_length_ms / 1000.)
        self.encoded_mbitrate_arr.append(encoded_mbitrate)
        self.single_mbitrate_arr.append(current_mbitrate)
        self.vmaf_arr.append(vmaf)

        observation = self.generate_observation_dictionary()

        info = {}

        reward = self.reward_function.return_reward_observation(observation)

        if activate_logging:
            self.log_state(current_level=quality,
                           rebuffering_seconds=rebuffer_time_ms / MILLISECONDS_IN_SECOND,
                           video_chunk_size_byte=video_chunk_size,
                           download_time_s=downloadtime_ms / MILLISECONDS_IN_SECOND, reward=reward)

        self.last_quality = quality
        return observation, reward, end_of_video, info

    def generate_observation_dictionary(self):
        quality = self.quality_arr[-1]
        observation = []
        observation.append(self.timestamp_arr[-self.max_lookback:])
        observation.append(self.data_used_relative[-self.max_lookback:])
        observation.append(self.quality_arr[-self.max_lookback:])
        observation.append(self.downloadtime_arr[-self.max_lookback:])
        observation.append(self.sleep_time_arr[-self.max_lookback:])
        observation.append(self.buffer_size_arr[-self.max_lookback:])
        observation.append(self.rebuffer_time_arr[-self.max_lookback:])
        observation.append(self.video_chunk_size_arr[-self.max_lookback:])
        observation.append(self.video_chunk_remain_arr_relative[-self.max_lookback:])
        observation.append(self.rate_played_relative_arr[-self.max_lookback:])
        observation.append(self.segment_length_arr[-self.max_lookback:])
        observation.append(self.encoded_mbitrate_arr[-self.max_lookback:])
        observation.append(self.single_mbitrate_arr[-self.max_lookback:])
        observation.append(self.vmaf_arr[-self.max_lookback:])
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.byte_size_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.bitrate_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.vmaf_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)

        observation.append(self)
        observation = {obs_key: obs_value for obs_key, obs_value in zip(self.get_obs_names(), observation)}
        return observation

    def get_past_dims(self):
        return len([v for v in self.get_obs_names() if 'future' not in v]) - 1

    def get_future_dims(self):
        return len([v for v in self.get_obs_names() if 'future' in v])

    def get_obs_names(self):
        column_switches = ['future_chunk_size_byte_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        column_switches += ['future_chunk_bitrate_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        column_switches += ['future_chunk_vmaf_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        return ['timestamp_s', 'data_used_bytes_relative', 'current_level', 'download_time_s', 'sleep_time_s',
                'buffer_size_s', 'rebuffer_time_s', 'video_chunk_size_byte',
                'relative_chunk_remain', 'relative_rate_played', 'segment_length_s', 'encoded_mbitrate',
                'single_mbitrate',
                'vmaf'] + column_switches + [
                   'streaming_environment']

    def reset(self):
        self.video_chunk_counter = 0
        self.buffer_size_ms = 0
        self.mahimahi_ptr = 1
        self.last_quality = 0
        self.logging_file = []
        self.timestamp_s = 0
        self.data_used_bytes = 0

        """
        Padding and stuff
        """
        self.timestamp_arr = [0] * self.max_lookback
        self.data_used_relative = [0] * self.max_lookback
        self.quality_arr = [0] * self.max_lookback
        self.downloadtime_arr = [0] * self.max_lookback
        self.sleep_time_arr = [0] * self.max_lookback
        self.buffer_size_arr = [0] * self.max_lookback
        self.rebuffer_time_arr = [0] * self.max_lookback
        self.video_chunk_remain_arr_relative = [0] * self.max_lookback
        self.video_chunk_size_arr = [0] * self.max_lookback
        self.rate_played_relative_arr = [0] * self.max_lookback
        self.segment_length_arr = [0] * self.max_lookback
        self.encoded_mbitrate_arr = [0] * self.max_lookback
        self.single_mbitrate_arr = [0] * self.max_lookback
        self.vmaf_arr = [0] * self.max_lookback


class StreamingSessionEvaluation:
    """
    Container Class
    """
    def __init__(self, streaming_session_evaluation: pd.DataFrame,
                 name: str, max_buffer_length_s: float, max_switch_allowed: int):
        self.max_buffer_length_s = max_buffer_length_s
        self.name = name
        self.streaming_session_evaluation = streaming_session_evaluation
        self.max_switch_allowed = max_switch_allowed

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Trajectory:


    def __init__(self):
        """
        Base class for trajectory of actions for video streaming
        """
        self.trace_video_pair_identifier = 0
        self.trajectory_list = []
        self.likelihood_list = []
        self.trajectory_state_t_arr = None
        self.trajectory_state_t_future = None  # Contains the future chunk sizes
        self.trajectory_state_t_1_arr = None
        self.trajectory_state_t_1_future = None  # Contains the future chunk sizes
        self.trajectory_action_t_arr = None
        self.trajectory_likelihood = None
        self.trajectory_column = None
        self.n_trajectories = None
        self.class_association = {}
        self.trajectory_sample_association = {}
        self.time_normalization = 10.
        self.n_features_observation = 0

    def scale_observation(self, state_unscaled):

        state_scaled = state_unscaled.copy()
        if sum(state_unscaled['timestamp_s']) != 0:
            state_scaled['timestamp_s'] = [state_unscaled['timestamp_s'][-1] - v for v in state_unscaled['timestamp_s']]
            state_scaled['timestamp_s'] = [v / state_scaled['timestamp_s'][0] for v in
                                           state_scaled[
                                               'timestamp_s']]  # How far are we away from the current measurement
        state_scaled['video_chunk_size_byte'] = [v * 1e-6 for v in state_unscaled['video_chunk_size_byte']]  # in mbyte
        del state_scaled['current_level']  # This value in of itself doesn't mean anything
        # encode that in mbyte to have the same scaling for all features
        state_scaled['encoded_mbitrate'] = [v / 8.0 for v in
                                            state_unscaled['encoded_mbitrate']]
        state_scaled['single_mbitrate'] = [v / 8.0 for v in
                                           state_unscaled['single_mbitrate']]

        state_scaled['download_time_s'] = [v / self.time_normalization for v in
                                           state_unscaled['download_time_s']]  # in mbyte
        state_scaled['sleep_time_s'] = [v / self.time_normalization for v in
                                        state_unscaled['sleep_time_s']]  # in mbyte
        state_scaled['rebuffer_time_s'] = [v / self.time_normalization for v in
                                           state_unscaled['rebuffer_time_s']]  # in mbyte
        state_scaled['throughput_mbyte'] = [(size * 1e-6) / time_s if time_s > 0 else 0 for size, time_s in zip(
            state_unscaled['video_chunk_size_byte'], state_unscaled['download_time_s'])]  # in mbyte
        state_scaled['segment_length_s'] = [v / self.time_normalization for v in
                                            state_unscaled['segment_length_s']]
        state_scaled['vmaf'] = [v / 100. for v in
                                state_unscaled['vmaf']]

        for k, v in state_scaled.items():
            if 'future_chunk_size_byte' in k:
                state_scaled[k] = [v * 1e-6 for v in state_unscaled[k]]
            if 'future_chunk_bitrate' in k:
                state_scaled[k] = [v * 1.25e-7 for v in state_unscaled[k]]
        return state_scaled

    def add_trajectory(self, other):

        other_trajectory_sample_association = other.trajectory_sample_association.copy()
        for k in other_trajectory_sample_association.keys():
            other_trajectory_sample_association[k] = [v + len(self.trajectory_list) for v in
                                                      other_trajectory_sample_association[k]]
        self.trajectory_sample_association.update(other_trajectory_sample_association)
        self.trajectory_list += other.trajectory_list
        self.likelihood_list += other.likelihood_list
        self.n_trajectories = len(self.trajectory_list)
        if self.trajectory_action_t_arr is not None:
            self.trajectory_state_t_arr = None
            self.trajectory_state_t_future = None  # Contains the future chunk sizes
            self.trajectory_state_t_1_arr = None
            self.trajectory_state_t_1_future = None  # Contains the future chunk sizes
            self.trajectory_action_t_arr = None
            self.trajectory_likelihood = None
            self.trajectory_column = None
            self.class_association = {}
            self.trajectory_sample_association = {}
            self.convert_list()

    def extract_trajectory(self, trace_video_pair_list):

        extraced_trajectory = Trajectory()
        try:
            for trace_video_pair_name,target_length in trace_video_pair_list:
                extraced_trajectory.new_trace_video_pair_name(trace_video_pair_name=trace_video_pair_name)
                indices = self.trajectory_sample_association[trace_video_pair_name]
                assert len(indices) > 0,"Didn't find any indices for extract trajectory"
                assert target_length == len(indices),'We didnt find all samples %d != %d %s' % (target_length,len(indices),trace_video_pair_name)
                for i in indices:
                    observation, observation_new, current_quality = self.trajectory_list[i]
                    likelihood, is_random = self.likelihood_list[i]
                    extraced_trajectory.add_trajectory_triple(observation, observation_new, current_quality)
                    extraced_trajectory.add_likelihood(likelihood, is_random)
        except:
            for trace_video_pair_name in trace_video_pair_list:
                extraced_trajectory.new_trace_video_pair_name(trace_video_pair_name=trace_video_pair_name)
                indices = self.trajectory_sample_association[trace_video_pair_name]
                assert len(indices) > 0,"Didn't find any indices for extract trajectory"
                #assert target_length == len(indices),'We didnt find all samples %d != %d %s' % (target_length,len(indices),trace_video_pair_name)
                for i in indices:
                    observation, observation_new, current_quality = self.trajectory_list[i]
                    likelihood, is_random = self.likelihood_list[i]
                    extraced_trajectory.add_trajectory_triple(observation, observation_new, current_quality)
                    extraced_trajectory.add_likelihood(likelihood, is_random)
        return extraced_trajectory

    def add_likelihood(self, likelihood, is_random):
        self.likelihood_list.append((likelihood, is_random))

    def add_observation_to_past_arr(self, arr, trajectory):
        trajectory = [v for k, v in sorted(
            trajectory.items()) if
                      'future' not in k]  # Needs to be sorted otherwise we don't know whether they are in the same order
        trajectory = np.array(trajectory).T
        trajectory = np.expand_dims(trajectory, axis=0)
        if arr is None:
            arr = trajectory
        else:
            arr = np.vstack([arr, trajectory])
        return arr

    def add_observation_to_future_arr(self, arr, trajectory):
        trajectory = [v for k, v in sorted(
            trajectory.items()) if
                      'future' in k]  # Needs to be sorted otherwise we don't know whether they are in the same order
        trajectory = np.array(trajectory).T
        trajectory = np.expand_dims(trajectory, axis=0)
        if arr is None:
            arr = trajectory
        else:
            arr = np.vstack([arr, trajectory])
        return arr

    def convert_list(self):
        if self.trajectory_column is None:
            self.trajectory_column = [k for k, v in sorted(
                self.trajectory_list[0][0].items())]  # Sorted columns contained in the trajectory_arr
            n_samples_already_converted = 0
        else:
            n_samples_already_converted = len(
                self.trajectory_action_t_arr)  # We have already gone through this loop
            if n_samples_already_converted >= len(self.trajectory_list):
                return
            state_t, _, _ = self.trajectory_list[n_samples_already_converted]
            state_t = [k for k, v in sorted(state_t.items())]
            assert len(self.trajectory_column) == len(state_t), 'We have more features than we had in the beginning'

        for trajectory_index, trajectory in enumerate(self.trajectory_list[n_samples_already_converted:]):
            state_t, state_t_1, action = trajectory
            state_t = self.scale_observation(state_t)
            state_t_1 = self.scale_observation(
                state_t_1)  # We need to do some scaling otherwise the GRU has problems to really learn from this
            self.trajectory_state_t_arr = self.add_observation_to_past_arr(self.trajectory_state_t_arr, state_t)
            self.trajectory_state_t_1_arr = self.add_observation_to_past_arr(self.trajectory_state_t_1_arr, state_t_1)
            self.trajectory_state_t_future = self.add_observation_to_future_arr(self.trajectory_state_t_future, state_t)
            self.trajectory_state_t_1_future = self.add_observation_to_future_arr(self.trajectory_state_t_1_future,
                                                                                  state_t_1)
            if self.trajectory_action_t_arr is None:
                self.trajectory_action_t_arr = np.array([action])  # Start an array
            else:
                self.trajectory_action_t_arr = np.vstack([self.trajectory_action_t_arr, action])
            likelihood_dist, _ = self.likelihood_list[trajectory_index]
            if self.trajectory_likelihood is None:
                self.trajectory_likelihood = likelihood_dist  # Start an array
            else:
                self.trajectory_likelihood = np.vstack([self.trajectory_likelihood, likelihood_dist])

            sample_idx = trajectory_index + n_samples_already_converted
            if action in self.class_association:
                self.class_association[action].append(sample_idx)
            else:
                self.class_association[action] = [sample_idx]

    def add_trajectory_triple(self, observation, observation_new, current_quality):
        self.trajectory_list.append((observation, observation_new, current_quality))
        self.n_trajectories = len(self.trajectory_list)
        self.n_features_observation = sum([len(v) for k, v in sorted(
            observation.items())])
        if self.trace_video_pair_identifier in self.trajectory_sample_association:
            self.trajectory_sample_association[self.trace_video_pair_identifier] += [self.n_trajectories - 1]
        else:
            self.trajectory_sample_association[self.trace_video_pair_identifier] = [self.n_trajectories - 1]

    def return_from_indices(self, indices):
        return_arr = [self.trajectory_state_t_arr[indices]]
        return_arr += [self.trajectory_state_t_future[indices]]
        return_arr += [self.trajectory_state_t_1_arr[indices]]
        return_arr += [self.trajectory_state_t_1_future[indices]]
        return_arr += [self.trajectory_action_t_arr[indices]]
        return_arr += [self.trajectory_likelihood[indices]]
        return return_arr

    def sample_equal_weight(self, size):
        if self.trajectory_state_t_arr is None or len(self.trajectory_list) > len(self.trajectory_state_t_arr):
            self.convert_list()
        indices_per_class = size // len(self.class_association)
        indices = []
        for k, v in self.class_association.items():
            indices += list(np.random.choice(v, size=indices_per_class))
        return self.return_from_indices(indices)

    def sample(self, size):
        if self.trajectory_state_t_arr is None or len(self.trajectory_list) > len(self.trajectory_state_t_arr):
            self.convert_list()
        indices = np.random.choice(np.arange(len(self.trajectory_list)), size=size)
        return self.return_from_indices(indices)

    def increment_trace_video_pair_identifier(self):
        if isinstance(self.trace_video_pair_identifier, int):
            self.trace_video_pair_identifier += 1
        else:
            raise ValueError('Identifier is a cc_session_identifier not a number')

    def new_trace_video_pair_name(self, trace_video_pair_name):
        self.trace_video_pair_identifier = trace_video_pair_name


class TrajectoryVideoStreaming:

    def __init__(self, abr_algorithm: ABRPolicy, streaming_enviroment: StreamingEnviroment, trace_list,
                 video_csv_list):
        """
        Used to create video streaming trajectories for a given abr-algorithm
        :param abr_algorithm:
        :param streaming_enviroment:
        :param trace_list:
        :param video_csv_list:
        """
        self.streaming_enviroment = streaming_enviroment
        self.video_csv_list = video_csv_list
        self.trace_list = trace_list
        self.abr_algorithm = abr_algorithm

    def determine_logging_path(self, current_log_path, video_id, all_file_names, net_env):
        log_path = os.path.join(current_log_path, 'video_{video_id}_file_id_{file_name}'.format(
            video_id=video_id, file_name=all_file_names[net_env.trace_idx]))
        logger.debug('Proposing %s to write the log' % log_path)
        while os.path.isfile(log_path) and len(open(log_path, 'report_var').read().split('\n')) >= len(
                net_env.video_information_csv):
            if (net_env.trace_idx + 1) >= len(all_file_names):
                logger.debug("There's nothing more to log")
                return None
            net_env.next_trace()  # We have already seen this trace
            log_path = os.path.join(current_log_path, 'video_{video_id}_file_id_{file_name}'.format(
                video_id=video_id, file_name=all_file_names[net_env.trace_idx]))
            logger.debug('Proposing %s to write the log' % log_path)
        if os.path.isfile(log_path):
            os.remove(log_path)
        return log_path

    def run_experiment(self, bw_trace_path, video_information_path, random_action_probability, is_parallel=True):
        """
        Runs experiment for the set algorithm, bandwidth trace and the video
        :param bw_trace_path:
        :param video_information_path:
        :param random_action_probability: Random action
        :param is_parallel:
        :return:
        """
        if is_parallel:
            stream_env = self.streaming_enviroment.copy()
            algo = self.abr_algorithm.copy()
        else:
            stream_env = self.streaming_enviroment
            algo = self.abr_algorithm
        algo.reset()
        trajectory_obj = Trajectory()
        video_id = video_information_path.split('/')[-1].replace('_video_info', '')
        trace_video_pair_name = 'video_' + video_id + '_file_id_' + bw_trace_path.split('/')[-1]
        stream_env.set_new_enviroment(bw_trace_path, video_information_path)
        trajectory_obj.new_trace_video_pair_name(trace_video_pair_name)
        previous_quality = 0
        current_quality = 0
        previous_observation = stream_env.generate_observation_dictionary()
        previous_likelihood = algo.likelihood_last_decision()
        del previous_observation['streaming_environment']  # We can't make use of this in the trajectory
        video_finished = False
        is_random = False
        while not video_finished:
            observation, reward, video_finished, info = stream_env.get_video_chunk(
                current_quality)
            switch = current_quality - previous_quality
            previous_quality = current_quality
            if video_finished:
                # Add the last observation
                del observation['streaming_environment']  # We can't make use of this in the trajectory
                action_idx = algo.map_switch_idx(switch)
                trajectory_obj.add_trajectory_triple(previous_observation, observation,
                                                     action_idx)
                trajectory_obj.add_likelihood(previous_likelihood, is_random)
                break
            is_random = np.random.random() <= random_action_probability
            if is_random:
                rnd_switch = np.random.randint(-algo.max_quality_change,
                                               algo.max_quality_change)
                current_quality = np.clip(current_quality + rnd_switch,
                                          a_min=0,
                                          a_max=stream_env.max_quality_level)
            else:
                current_quality = algo.next_quality(observation, reward)
            del observation['streaming_environment']  # We can't make use of this in the trajectory
            action_idx = algo.map_switch_idx(switch)
            trajectory_obj.add_trajectory_triple(previous_observation, observation,
                                                 action_idx)
            trajectory_obj.add_likelihood(previous_likelihood, is_random)

            if is_random:
                uniform_likelhood = np.ones(algo.likelihood_last_decision().shape)
                previous_likelihood = uniform_likelhood / sum(uniform_likelhood)
            else:
                previous_likelihood = algo.likelihood_last_decision()
            previous_observation = observation
            """
            Add the expert policy
            """
        streaming_session_evaluation = pd.DataFrame(stream_env.return_log_state(),
                                                    columns=stream_env.get_logging_columns())
        streaming_session_evaluation = StreamingSessionEvaluation(
            streaming_session_evaluation=streaming_session_evaluation,
            name=trace_video_pair_name,
            max_buffer_length_s=stream_env.buffer_threshold_ms / 1000.,
            max_switch_allowed=algo.max_quality_change)
        return [streaming_session_evaluation], trajectory_obj

    def create_trajectories(self, random_action_probability=0.0, tqdm_activated=False, cores_avail=1):
        """
        Run experiment with the set algorithm traces and videos
        :param random_action_probability:
        :param tqdm_activated:
        :param cores_avail:
        :return:
        """

        trajectory_obj = Trajectory()
        logging_list = []
        trajectory_zip = list(zip(self.trace_list, self.video_csv_list))  # We want to have an expected runtime
        if tqdm_activated:
            trajectory_zip = tqdm(trajectory_zip, desc='Trajectory Counter : ')
        if cores_avail == 1:
            results = [
                self.run_experiment(bw_trace_path, video_information_path, random_action_probability, is_parallel=False)
                for
                bw_trace_path, video_information_path in trajectory_zip]
        else:
            results = Parallel(n_jobs=cores_avail)(delayed(self.run_experiment)(bw_trace_path,
                                                                                video_information_path,
                                                                                random_action_probability, True) for
                                                   bw_trace_path, video_information_path in trajectory_zip)
        for sse, tr in results:
            logging_list += sse
            trajectory_obj.add_trajectory(tr)
        return np.array(logging_list).flatten(), trajectory_obj
