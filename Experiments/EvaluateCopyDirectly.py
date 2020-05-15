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

from ABRPolicies.ComplexABRPolicy import PensieveMultiNN, MPC
from ABRPolicies.OptimalABRPolicy import Optimal
from ABRPolicies.SimpleABRPolicy import Rate
from ABRPolicies.ThroughputEstimator import StepEstimator
from BehaviourCloning.ActionCloning import BehavioralCloningIterative, BehavioralCloning
from BehaviourCloning.ImitationLearning import DAGGERCloner, VIPERCloner
from BehaviourCloning.MLABRPolicy import ABRPolicyClassifierSimple, ABRPolicyValueFunctionLearner, \
    ABRPolicyClassifierAutomatedFeatureEngineering, ABRPolicyClassifierHandFeatureEngineering, ABRPolicyRate
from BehaviourCloning.RewardShapingCloning import GAILPPO, RandomExpertDistillation
from Scripts.EvaluateCopyExample import MAX_INTERPRETABILITY, IMITATION_EPOCHS
from SimulationEnviroment.Rewards import ClassicPerceptualReward
from SimulationEnviroment.SimulatorEnviroment import OfflineStreaming

# %%

PARENT_FOLDER = 'Data'
video_selection_dataframe = PARENT_FOLDER + '/VideoSelectionList/VideoListRecoded.csv'
video_selection_dataframe = pd.read_csv(video_selection_dataframe, index_col=0)
provider_index = video_selection_dataframe.index
provider_index = ['SRF' if t == 'SFR' else t for t in provider_index]
video_selection_dataframe.index = provider_index
ALGORITHM_TYPE = ['Robust_Rate',
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
print('Analyzing %s and Copying %s directly' % (provider, algorithm_type))
expert_algorithm = None

future_bandwidth_estimator = StepEstimator(consider_last_n_steps=5,
                                           predictor_function=hmean,
                                           robust_estimate=True)
if algorithm_type == 'Robust_Rate':
    expert_algorithm = Rate(abr_name='Robust_Rate_Direct',
                            throughput_predictor=future_bandwidth_estimator,
                            downscale_factor=1.0,
                            upscale_factor=1.0,
                            max_quality_change=2)
elif algorithm_type == 'Robust_MPC':
    expert_algorithm = MPC(abr_name='Robust_MPC_Direct',
                           throughput_predictor=future_bandwidth_estimator,
                           downscale_factor=1.0,
                           upscale_factor=1.0,
                           max_quality_change=2,
                           lookahead=3)

elif algorithm_type == 'Pensieve_MultiVideo':
    VIDEO_BIT_RATE = [200, 1100, 2000, 2900, 3800, 4700, 5600, 6500, 7400, 8400]
    expert_algorithm = PensieveMultiNN(abr_name='Pensieve_MultiVideo_Direct',
                                       max_quality_change=2,
                                       deterministic=True,
                                       rate_choosen=VIDEO_BIT_RATE,
                                       nn_model_path='ABRPolicies/PensieveMultiVideo/models/nn_model_ep_96000.ckpt')
elif algorithm_type == 'Optimal':
    expert_algorithm = Optimal(abr_name='Optimal_Direct',
                               max_quality_change=2,
                               deterministic=True,
                               lookahead=3)
else:
    raise ValueError('Algorithm is not available')

GRANULARITY_AMOUNT_DATA = 5
MAX_AMOUNT_DATA = 2500
MIN_AMOUNT_DATA = 300
EXPERIMENT_FULL_NAME = 'provider_full_evaluation.csv'

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
    video_ids = [f.name.split('_file_id_')[0].replace('video_', '') for f in expert_evaluation]
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
    ##############################################################################################################
    # %%
    experiment_folder_template = parsed_results_folder.replace('ParsedResults', 'MethodEvaluation')
    if not os.path.exists(experiment_folder_template):
        os.makedirs(experiment_folder_template)
    value_function_learner = ABRPolicyValueFunctionLearner(abr_name='XGB Regressor',
                                                           max_quality_change=2,
                                                           regressor=XGBRegressor())
    ############################################################################################################
    ### Advanced Cloning Techniques with DAGGER and VIPER
    print('DAGGER/VIPER Copying Pipeline')
    simple_classifier = ABRPolicyClassifierSimple(
        abr_name='No Feature Engineering', max_quality_change=2,
        deterministic=True,
        max_lookahead=3,
        max_lookback=10,
        classifier=DecisionTreeClassifier(max_leaf_nodes=MAX_INTERPRETABILITY))
    imitation_learning_names = ['DAGGER', 'VIPER']
    for imitation_learning_technique, imitation_learning_technique_name in zip([DAGGERCloner, VIPERCloner],
                                                                               imitation_learning_names):
        cloning_method_name = '%s -> %s' % (expert_algorithm.abr_name, imitation_learning_technique_name)
        experiment_name = '%s -> %s %d leaf nodes' % (cloning_method_name, simple_classifier.abr_name,
                                                      simple_classifier.classifier.max_leaf_nodes)
        file_name = '_'.join(experiment_name.split(" "))
        experiment_folder_name = os.path.join(experiment_folder_template, file_name)
        if not os.path.exists(experiment_folder_name):
            os.makedirs(experiment_folder_name)
        elif experiment_has_finished(experiment_folder_name):
            print("We've already done %s " % experiment_name)
            evaluation_dataframe = pd.read_csv(os.path.join(
                experiment_folder_name, EXPERIMENT_FULL_NAME), index_col=0).to_dict(orient='list')
            continue
        start_time = time()
        imitation_learning_technique = imitation_learning_technique(
            imitation_learning_technique_name,
            max_quality_change=2,
            deterministic=True,
            training_epochs=IMITATION_EPOCHS,
            abr_policy_learner=simple_classifier,
            value_function_learner=value_function_learner, cores_avail=cores_avail)
        imitation_learning_technique.clone_from_expert(
            expert_algorithm=expert_algorithm,
            streaming_enviroment=streaming_enviroment,
            trace_list=expert_traces_training,
            video_csv_list=expert_videos_training,
            show_progress=True
        )
        imitation_learning_technique_scoring, imitation_learning_technique_evaluation = imitation_learning_technique.score(
            expert_evaluation=expert_evaluation_validation,
            expert_trajectory=expert_trajectory_validation,
            streaming_enviroment=streaming_enviroment,
            trace_list=expert_traces_validation,
            video_csv_list=expert_videos_validation,
        )
        imitation_learning_technique_scoring['Provider'] = [provider]
        imitation_learning_technique_scoring['Base Classifier'] = [simple_classifier.abr_name]
        imitation_learning_technique_scoring['Cloning Method'] = [cloning_method_name]
        imitation_learning_technique_scoring['Training Samples'] = [n_training_samples_full]
        imitation_learning_technique_scoring['Leaf Complexity'] = [simple_classifier.classifier.max_leaf_nodes]
        imitation_learning_technique_scoring['Training Time'] = [time() - start_time]

        for k, v in imitation_learning_technique_scoring.items():
            if k in evaluation_dataframe:
                evaluation_dataframe[k] += v
            else:
                evaluation_dataframe[k] = v

        print('%s took %.2f s' % (experiment_name, time() - start_time))
        with open(os.path.join(experiment_folder_name, 'classifier'), 'wb') as output_file:
            dill.dump(simple_classifier, output_file)
        with open(os.path.join(experiment_folder_name, 'evaluation'), 'wb') as output_file:
            dill.dump(imitation_learning_technique_evaluation, output_file)
        pd.DataFrame(imitation_learning_technique.policy_history).to_csv(
            os.path.join(experiment_folder_name, 'policy_learning_history.csv'))
        pd.DataFrame(evaluation_dataframe).to_csv(os.path.join(
            experiment_folder_name, EXPERIMENT_FULL_NAME))
