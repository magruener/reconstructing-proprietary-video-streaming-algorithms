# %%
import argparse
import os
import pickle
from time import time

import dill as dill
import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from ABRPolicies.OptimalABRPolicy import Optimal
from ABRPolicies.ThroughputEstimator import StepEstimator
from BehaviourCloning.ActionCloning import BehavioralCloningIterative, BehavioralCloning
from BehaviourCloning.ImitationLearning import DAGGERCloner, VIPERCloner
from BehaviourCloning.MLABRPolicy import ABRPolicyClassifierSimple, ABRPolicyValueFunctionLearner, \
    ABRPolicyClassifierAutomatedFeatureEngineering, ABRPolicyClassifierHandFeatureEngineering, ABRPolicyRate
from BehaviourCloning.RewardShapingCloning import GAILPPO, RandomExpertDistillation
from SimulationEnviroment.Rewards import ClassicPerceptualReward
from SimulationEnviroment.SimulatorEnviroment import OfflineStreaming, TrajectoryVideoStreaming

# %%

PARENT_FOLDER = 'Data'
video_selection_dataframe = PARENT_FOLDER + '/VideoSelectionList/VideoListRecoded.csv'
video_selection_dataframe = pd.read_csv(video_selection_dataframe, index_col=0)
provider_index = video_selection_dataframe.index
provider_index = ['SRF' if t == 'SFR' else t for t in provider_index]
video_selection_dataframe.index = provider_index
ALGORITHM_TYPE = ['Online',
                  'Robust_Rate',
                  'Robust_MPC',
                  'Pensieve_MultiVideo',
                  'Optimal']

parser = argparse.ArgumentParser(description='Run Experiments')
parser.add_argument('provider',
                    help='Select one of the providers shown here %s for analysis' % str(np.unique(provider_index)))
parser.add_argument('algorithm',
                    help='Select one of the algorithms which is in' % ALGORITHM_TYPE)
parser.add_argument('cores_avail', default=1,
                    help='We add a bit of multicoring as we have multiple available')
args = parser.parse_args()

parsed_results_folder = PARENT_FOLDER + '/ParsedResults'
assert args.provider in os.listdir(parsed_results_folder), 'The provider is not parsed'
provider = args.provider
parsed_results_folder = os.path.join(parsed_results_folder, provider)
assert args.algorithm in os.listdir(parsed_results_folder), 'The algorithm is not parsed'
algorithm_type = args.algorithm
parsed_results_folder = os.path.join(parsed_results_folder, algorithm_type)

cores_avail = int(args.cores_avail)

print('Analyzing %s' % provider)
GRANULARITY_AMOUNT_DATA = 5
MAX_AMOUNT_DATA = 2500
MIN_AMOUNT_DATA = 300
EXPERIMENT_FULL_NAME = 'provider_full_evaluation.csv'
MAX_INTERPRETABILITY = 20
IMITATION_EPOCHS = 50

# %%

video_info_files = []
for root, dirs, files in os.walk(PARENT_FOLDER + '/Video_Info/'):
    for name in files:
        if name.endswith('_video_info') and 'Phone' not in root:
            video_info_files.append(os.path.join(root, name))


def find_video_info(path):
    video_id = path.split('/')[-2]
    if '_file_id_' in video_id:
        video_id = video_id.split('_file_id_')[0]
    elif 'bw' in video_id:
        video_id = video_id.split('_bw_')[0]
    else:
        video_id = video_id.split('_epoch_')[0]
    video_id = video_id.replace('video_', '')
    video_info_file = list(filter(lambda path_to_csv: video_id in path_to_csv, video_info_files))
    assert len(video_info_file) == 1, video_info_file
    return video_info_file[0]


trace_files = []
for root, dirs, files in os.walk(PARENT_FOLDER + '/Traces'):
    for name in files:
        trace_files.append(os.path.join(root, name))

print('We have %d traces in the collection' % len(trace_files))


def find_trace_file(path):
    trace_file_id = path.split('/')[-2]
    if '_file_id_' in trace_file_id:
        trace_file_id = trace_file_id.split('_file_id_')[-1]
    elif '_trace_' in trace_file_id:
        trace_file_id = trace_file_id.split('_trace_')[-1]
    else:
        raise ValueError('We dont have a generating file %s' % trace_file_id)
    trace_file = list(filter(lambda path_to_bw_trace: trace_file_id in path_to_bw_trace, trace_files))
    assert len(trace_file) == 1, trace_file
    return trace_file[0]


# %%

reward_function_used = ClassicPerceptualReward()
RANDOM_FIXED_SEED = 42
evaluation_dataframe = {}


def experiment_has_finished(experiment_path):
    return EXPERIMENT_FULL_NAME in os.listdir(experiment_path)


# %%

if __name__ == "__main__":
    try:
        video_selection = video_selection_dataframe.loc[provider.replace('_Phone', '')]
    except:
        video_selection = video_selection_dataframe.loc[provider.replace('_Phone', '').lower()]
    training_videos = video_selection[video_selection['Data Type'] != 'validation']['Video Url'].values
    training_videos = list(training_videos)
    expert_trajectory = os.path.join(parsed_results_folder, 'trajectory_list')
    with open(expert_trajectory, 'rb') as expert_trajectory:
        expert_trajectory = pickle.load(expert_trajectory)
    expert_evaluation = os.path.join(parsed_results_folder, 'evaluation_list')
    with open(expert_evaluation, 'rb') as expert_evaluation:
        expert_evaluation = pickle.load(expert_evaluation)
    expert_evaluation = np.array(expert_evaluation)
    expert_traces = [find_trace_file(f.name + '/') for f in expert_evaluation]
    expert_traces = np.array(expert_traces)
    if '_Phone' in provider:  # Quick fix for the phone data
        expert_videos = [find_video_info(f.name.replace('62085745', '62092214') + '/') for f in expert_evaluation]
    else:
        expert_videos = [find_video_info(f.name + '/') for f in expert_evaluation]
    expert_videos = np.array(expert_videos)
    ######################################################################
    if '_Phone' in provider: #Quick fix for the phone data
        video_ids = [f.name.split('_file_id_')[0].replace('video_', '').replace('62085745', '62092214') for f in expert_evaluation]
    else:
        video_ids = [f.name.split('_file_id_')[0].replace('video_', '') for f in
                     expert_evaluation]
    training_indices = []
    for id in video_ids:
        mapping = [id in url for url in training_videos]
        assert sum(mapping) <= 1
        if sum(mapping) == 1:
            training_indices.append(True)
        else:
            training_indices.append(False)
    training_indices = np.array(training_indices)
    validation_indices = np.where(1.0 - training_indices)[0].astype(int)
    training_indices = np.where(training_indices)[0].astype(int)
    np.random.seed(RANDOM_FIXED_SEED)

    expert_evaluation_training = expert_evaluation[training_indices]
    expert_trajectory_training = [expert_evaluation[idx].name for idx in training_indices]
    expert_trajectory_training = expert_trajectory.extract_trajectory(expert_trajectory_training)
    expert_traces_training = expert_traces[training_indices]
    expert_videos_training = expert_videos[training_indices]

    expert_evaluation_validation = expert_evaluation[validation_indices]
    expert_trajectory_validation = [expert_evaluation[idx].name for idx in validation_indices]
    expert_trajectory_validation = expert_trajectory.extract_trajectory(
        expert_trajectory_validation)
    expert_traces_validation = expert_traces[validation_indices]
    expert_videos_validation = expert_videos[validation_indices]
    n_training_samples_full = len(expert_trajectory_training.trajectory_list)
    n_full_experiments = len(expert_evaluation_training)
    sample_experiment_float = [len(fr.streaming_session_evaluation) for fr in expert_evaluation_training]
    avg_sample_experiment_int = np.mean(sample_experiment_float)
    max_buffer_s = np.median([f.max_buffer_length_s for f in expert_evaluation_training])
    streaming_enviroment = OfflineStreaming(bw_trace_file=expert_traces_training[0],
                                            video_information_csv_path=expert_videos_training[0],
                                            reward_function=reward_function_used,
                                            max_lookback=10,
                                            max_lookahead=3,
                                            max_switch_allowed=2,
                                            buffer_threshold_ms=max_buffer_s * 1000.)

    print('Setting maximum buffer to %.2f' % (max_buffer_s))
    print('%d Training Samples | %d Validation Samples' % (len(expert_trajectory_training.trajectory_list),
                                                           len(expert_trajectory_validation.trajectory_list)))
    algorithms = [Optimal(abr_name='Optimal',max_quality_change=2,
                           deterministic=True,
                           lookahead=3)]
    ##############################################################################################################

    offline_evaluation_list = []
    streaming_enviroment_local = streaming_enviroment
    for alg in [algorithms[0]]:
        provider_folder = os.path.join(PARENT_FOLDER + '/ParsedResults/', provider, '_'.join(alg.abr_name.split()))
        if not os.path.exists(provider_folder):
            os.makedirs(provider_folder)
        else:
            continue
        print('Evaluating %s' % alg.abr_name)
        offline_generator = TrajectoryVideoStreaming(alg, streaming_enviroment_local,
                                                     trace_list=expert_traces_validation,
                                                     video_csv_list=expert_videos_validation)
        offline_evaluation_frame, offline_trajectory = offline_generator.create_trajectories(
            tqdm_activated=True,
        )
        with open(provider_folder + '/evaluation_list', 'wb') as dump_file:
            pickle.dump(offline_evaluation_frame, dump_file)
        with open(provider_folder + '/trajectory_list', 'wb') as dump_file:
            pickle.dump(offline_trajectory, dump_file)