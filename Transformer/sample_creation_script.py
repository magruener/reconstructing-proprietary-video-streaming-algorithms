import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from SimulationEnviroment.Rewards import ClassicPerceptualReward
from Transformer.LegacyTransformer import LegacyTransformer, EvaluationTransformer

##### Load all video and trace files
video_info_files = []
for root, dirs, files in os.walk('../Data/Video_Info/'):
    for name in files:
        if name.endswith('_video_info') and 'Phone' not in root:
            video_info_files.append(os.path.join(root, name))

trace_files = []
for root, dirs, files in os.walk('../Data/Traces/'):
    for name in files:
        trace_files.append(os.path.join(root, name))


def find_video_info(path):
    """
    Utility function
    :param path:
    :return:
    """
    video_id = path.split('/')[-2]
    if '_file_id_' in video_id:
        video_id = video_id.split('_file_id_')[0]
    elif 'bw' in video_id:
        video_id = video_id.split('_bw_')[0]
    else:
        video_id = video_id.split('_epoch_')[0]
    video_id = video_id.replace('video_', '')
    video_info_file = list(filter(lambda path_to_csv: video_id in path_to_csv, video_info_files))
    assert len(video_info_file) > 0, video_id
    assert len(video_info_file) == 1, video_info_file
    return video_info_file[0]


def find_trace_file(path):
    """
    Utility function
    :param path:
    :return:
    """
    trace_file_id = path.split('/')[-2]
    if '_file_id_' in trace_file_id:
        trace_file_id = trace_file_id.split('_file_id_')[-1]
    else:
        raise ValueError('We dont have a generating file')
    trace_file = list(filter(lambda path_to_bw_trace: trace_file_id in path_to_bw_trace, trace_files))
    assert len(trace_file) == 1, trace_file
    return trace_file[0]


if __name__ == "__main__":
    for provider in os.listdir('../Data/FeedbackResults'):
        if '.DS_Store' in provider:
            continue
        trajectory_list = None
        streaming_evaluation_dataframe_list = []
        if 'Phone' in provider:  ### Phone data is parsed a bit different as we can't access the html5 element
            parent_path = '../Data/FeedbackResults/%s' % provider
            buffer_level_estimate = []
            for evaluation_frame in os.listdir(parent_path):
                local_path = os.path.join(parent_path, evaluation_frame)
                evaluation_frame = pd.read_csv(local_path)
                buffer_level_estimate += [evaluation_frame['buffer_level_at_timestamp_finish'].quantile(.75)]
            buffer_max_seconds = np.around((np.percentile(buffer_level_estimate, 90).astype(
                int) / 10.)) * 10  ### Reference buffer which we can fill. We have to account for noise
            for evaluation_frame in tqdm(os.listdir(parent_path),
                                         'Conversion buffer size : %s' % provider):
                local_path = os.path.join(parent_path, evaluation_frame)
                try:
                    video_information_csv_path = find_video_info(
                        local_path.replace('62085745', '62092214') + '/')  ### Naming issue with the vimeo data
                    bw_trace_file = find_trace_file(local_path + '/')
                except:
                    print('Skipping %s' % evaluation_frame)
                    continue
                eval_df = pd.read_csv(local_path, index_col=0)
                eval_df['estimated_bandwidth_mbit'] = (eval_df['body_size_byte'] * 8e-6) / eval_df['t_download_s']
                eval_df['delta_t'] = eval_df['timestamp_finish'] - eval_df['timestamp_finish'].iloc[0]
                target_bw = pd.read_csv(bw_trace_file, sep=' ', names=['time', 'bw'])
                target_bw = target_bw[target_bw['time'] <= eval_df['delta_t'].max()]
                eval_df = eval_df[eval_df['delta_t'] <= target_bw['time'].max()]
                deviaton_mbit = eval_df['estimated_bandwidth_mbit'].mean() - target_bw['bw'].mean()
                if np.abs(deviaton_mbit) < 2.:
                    tr = EvaluationTransformer(buffer_threshold_ms=buffer_max_seconds * 1000.,
                                               max_lookahead=3, max_lookback=10,
                                               max_switch_allowed=2, fps_smoothing_s=4,
                                               video_information_csv_path=video_information_csv_path,
                                               reward_function=ClassicPerceptualReward(),
                                               bw_trace_file=bw_trace_file)
                    streaming_evaluation_dataframe, trajectory_obj = tr.transform_csv(local_path)
                    streaming_evaluation_dataframe_list += streaming_evaluation_dataframe
                    if trajectory_list is None:
                        trajectory_list = trajectory_obj
                    else:
                        trajectory_list.add_trajectory(trajectory_obj)

        else:
            result_folders = []
            for root, dirs, files in os.walk('../Data/FeedbackResults/%s' % provider):
                for name in dirs:
                    if name.startswith('video_') and 'Proxy' not in root:
                        result_folders.append(os.path.join(root, name))
            buffer_level_estimate = []
            for folder in tqdm(result_folders, 'Calculate buffer size : %s' % provider):
                if 'local_client_state_logger.csv' not in os.listdir(folder):
                    continue
                client_logger_file = folder + '/local_client_state_logger.csv'
                client_logger_file = pd.read_csv(client_logger_file)
                client_logger_file['buffer_level'] = client_logger_file['buffered_until'] - client_logger_file[
                    'played_until']
                buffer_level_estimate += [client_logger_file['buffer_level'].quantile(.75)]
            buffer_max_seconds = np.around((np.percentile(buffer_level_estimate, 90).astype(int) / 10.)) * 10
            buffer_max_seconds = max(buffer_max_seconds,
                                     60)  ### Reference buffer which we can fill. We have to account for noise
            print('Buffer in Seconds %.2f' % buffer_max_seconds)
            for folder in tqdm(result_folders, 'Conversion : %s' % provider):
                if len(os.listdir(folder)) < 3:
                    #### Experiment has not fully been concluded
                    continue
                if 'raw_dataframe.csv' not in os.listdir(folder):
                    #### Experiment has not fully been concluded
                    continue
                to_transform_path = folder + '/raw_dataframe.csv'
                video_information_csv_path = find_video_info(to_transform_path)
                bw_trace_file = find_trace_file(to_transform_path)
                tr = LegacyTransformer(buffer_threshold_ms=buffer_max_seconds * 1000.,
                                       max_lookahead=3, max_lookback=10,
                                       max_switch_allowed=2, fps_smoothing_s=4,
                                       video_information_csv_path=video_information_csv_path,
                                       reward_function=ClassicPerceptualReward(),
                                       bw_trace_file=bw_trace_file)
                streaming_evaluation_dataframe, trajectory_obj = tr.transform_csv(to_transform_path)

                if trajectory_list is None or list(trajectory_obj.trajectory_sample_association.keys())[
                    0] not in trajectory_list.trajectory_sample_association.keys():
                    streaming_evaluation_dataframe_list += streaming_evaluation_dataframe
                    if trajectory_list is None:
                        trajectory_list = trajectory_obj
                    else:
                        trajectory_list.add_trajectory(trajectory_obj)
        len_eval_total = [len(ev_df.streaming_session_evaluation) for ev_df in streaming_evaluation_dataframe_list]
        assert len(trajectory_list.trajectory_list) == sum(
            len_eval_total), 'Wrong number of comparing instances %d != %d Training' % (
            len(trajectory_list.trajectory_list), sum(len_eval_total))
        for frame in streaming_evaluation_dataframe_list:
            cond = len(trajectory_list.trajectory_sample_association[frame.name]) == len(
                frame.streaming_session_evaluation)
            assert cond
        provider_folder = os.path.join('../Data/ParsedResults/', provider, 'Online')
        if not os.path.exists(provider_folder):
            os.makedirs(provider_folder)
        with open(provider_folder + '/evaluation_list', 'wb') as dump_file:
            pickle.dump(streaming_evaluation_dataframe_list, dump_file)
        with open(provider_folder + '/trajectory_list', 'wb') as dump_file:
            pickle.dump(trajectory_list, dump_file)
