import os
import pickle

import numpy as np
from scipy.stats import hmean

from ABRPolicies.ComplexABRPolicy import MPC, PensieveMultiNN
from ABRPolicies.OptimalABRPolicy import Optimal
from ABRPolicies.SimpleABRPolicy import Rate
from ABRPolicies.ThroughputEstimator import StepEstimator
from SimulationEnviroment.Rewards import ClassicPerceptualReward
from SimulationEnviroment.SimulatorEnviroment import TrajectoryVideoStreaming, OfflineStreaming

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


"""
Define Bandwidth estimator
"""
future_bandwidth_estimator = StepEstimator(consider_last_n_steps=5,
                                           predictor_function=hmean,
                                           robust_estimate=True)

"""
Add industry standards we want to test
"""

algorithms = [
]
algorithms += [Rate(abr_name='Robust Rate',
                    throughput_predictor=future_bandwidth_estimator,
                    downscale_factor=1.0,
                    upscale_factor=1.0,
                    max_quality_change=2)]

pensieve_nn_rate_choosen = [200, 1100, 2000, 2900, 3800, 4700, 5600, 6500, 7400, 8400]

algorithms += [PensieveMultiNN(abr_name='Pensieve MultiVideo',
                               max_quality_change=2,
                               deterministic=True,
                               rate_choosen=pensieve_nn_rate_choosen,
                               nn_model_path='../ABRPolicies/PensieveMultiVideo/models/nn_model_ep_96000.ckpt')
               ]

algorithms += [MPC(abr_name='Robust MPC',
                   throughput_predictor=future_bandwidth_estimator,
                   downscale_factor=1.0,
                   upscale_factor=1.0,
                   max_quality_change=2,
                   lookahead=3)]
algorithms += [Optimal(abr_name='Optimal',
                       max_quality_change=2,
                       deterministic=True,
                       lookahead=3)]

# %%
#################################################################3
# Get the local transfromers
provider = 'SRF'  ### Choose the providers data you want to evaluate the algorithms on !!! PensieveNN needs to be trained on corresponding data
trace_list = []
video_list = []
for root, dirs, files in os.walk('../Data/FeedbackResults/%s' % provider):
    for name in dirs:
        if name.startswith('video_') and 'Proxy' not in root:
            to_transform_path = os.path.join(root, name) + '/raw_dataframe.csv'
            video_information_csv_path = find_video_info(to_transform_path)
            bw_trace_file = find_trace_file(to_transform_path)
            trace_list.append(bw_trace_file)
            video_list.append(video_information_csv_path)

provider_folder = os.path.join('../Data/ParsedResults/', provider, 'Online')
with open(provider_folder + '/evaluation_list', 'rb') as dump_file:
    eval_files = pickle.load(dump_file)
max_buffer_s = np.median([f.max_buffer_length_s for f in eval_files])  ## Use the buffer data from the p;ayer
streaming_enviroment_local = OfflineStreaming(bw_trace_file=trace_list[0],
                                              video_information_csv_path=video_list[0],
                                              reward_function=ClassicPerceptualReward(),
                                              max_lookback=10,
                                              max_lookahead=3,
                                              max_switch_allowed=2,
                                              buffer_threshold_ms=max_buffer_s * 1000.)
if __name__ == "__main__":

    for alg in algorithms:
        print('Evaluating %s' % alg.abr_name)
        offline_generator = TrajectoryVideoStreaming(alg, streaming_enviroment_local,
                                                     trace_list=trace_list,
                                                     video_csv_list=video_list)
        offline_evaluation_frame, offline_trajectory = offline_generator.create_trajectories(
            tqdm_activated=True,
        )
        provider_folder = os.path.join('../Data/ParsedResults/', provider, '_'.join(alg.abr_name.split()))
        if not os.path.exists(provider_folder):
            os.makedirs(provider_folder)
        with open(provider_folder + '/evaluation_list', 'wb') as dump_file:
            pickle.dump(offline_evaluation_frame, dump_file)  ### Save the raw evaluation
        with open(provider_folder + '/trajectory_list', 'wb') as dump_file:
            pickle.dump(offline_trajectory, dump_file)  ### Saw the trajectory of the player
