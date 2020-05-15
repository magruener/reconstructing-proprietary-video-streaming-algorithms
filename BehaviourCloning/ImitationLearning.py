import logging
import os
from abc import abstractmethod

import dill
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ABRPolicies.ABRPolicy import ABRPolicy
from BehaviourCloning.MLABRPolicy import ABRPolicyLearner, ABRPolicyValueFunctionLearner
from SimulationEnviroment.SimulatorEnviroment import Trajectory, TrajectoryVideoStreaming

N_BINS_DIST = 15
LOGGING_LEVEL = logging.DEBUG
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)





class ABRCloner(ABRPolicy):

    def __init__(self, abr_name, max_quality_change, deterministic, training_epochs,
                 abr_policy_learner: ABRPolicyLearner, value_function_learner: ABRPolicyValueFunctionLearner,
                 validation_split=0.2, future_reward_discount=0.99, random_action_probability_init=.9,
                 cores_avail=1, min_random_action_probability=0.05, exploration_percentage=0.25,
                 weight_samples_method = 'Viper'):
        """
        Base class for imitation learner implemented below
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param training_epochs:
        :param abr_policy_learner:
        :param value_function_learner:
        :param validation_split:
        :param future_reward_discount:
        :param random_action_probability_init:
        :param cores_avail:
        :param min_random_action_probability:
        :param exploration_percentage:
        :param weight_samples_method: Assigns importance to choosing the specific action in the training set.
        VIPER : Weigh action by difference in reward you gain from choosing this action and the worst action
        Divergence : Weight action by difference in reward you gain from choosing the best action and the worst action in this situation.
        """
        self.valid_weight_samples_method = ['Divergence','Viper']
        assert weight_samples_method in self.valid_weight_samples_method,'Choose correct sampling method in %s' % self.valid_weight_samples_method
        self.weight_samples_method = weight_samples_method
        super().__init__(abr_name, max_quality_change, deterministic)
        self.cores_avail = cores_avail
        self.value_function_learner = value_function_learner
        self.abr_policy_learner = abr_policy_learner
        self.anomaly_scorer = IsolationForest(contamination='auto', behaviour='new')
        self.training_epochs = training_epochs
        self.feature_names = []
        self.data_collection_list = []
        self.validation_split = validation_split
        self.future_reward_discount = future_reward_discount
        self.random_action_probability_init = random_action_probability_init
        self.policy_history = {}
        self.min_random_action_probability = min_random_action_probability
        self.random_action_probability_decay = (self.min_random_action_probability / self.random_action_probability_init) ** (
                                                           1. / int(
                                                       training_epochs * exploration_percentage))
        self.exploration_percentage = exploration_percentage

    def next_quality(self, observation, reward):
        return self.abr_policy_learner.next_quality(observation, reward)

    def fit_value_function(self, algorithm_to_imitate: ABRPolicy, streaming_enviroment, trace_list,
                           video_csv_list):
        imitation_trajectory_generator = TrajectoryVideoStreaming(algorithm_to_imitate,
                                                                  streaming_enviroment,
                                                                  trace_list=trace_list,
                                                                  video_csv_list=video_csv_list)
        to_imitate_evaluation, to_imitate_trajectory = imitation_trajectory_generator.create_trajectories()
        transformed_observations = np.array(
            [self.value_function_learner.extract_features_observation(state_t) for state_t, _, _ in
             tqdm(to_imitate_trajectory.trajectory_list, desc='transforming')])
        transformed_observations = pd.DataFrame(transformed_observations,
                                                columns=self.value_function_learner.extract_features_names())
        self.anomaly_scorer.fit(transformed_observations)
        ################################33
        ### self.anomaly_scorer.score_samples(self,X) Opposite of the anomaly score defined in the original paper.
        ### The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
        anomaly_score = self.anomaly_scorer.score_samples(transformed_observations)

        i = 0
        transformed_reward = []
        for evaluation_dataframe in to_imitate_evaluation:
            i_end = i + len(evaluation_dataframe.streaming_session_evaluation)
            reward_transform = list(anomaly_score[i:i_end].copy())
            # We ignore the last reward obtained as we don't have a corresponding state
            for j in range(1, len(reward_transform))[::-1]:
                exponent = (len(reward_transform) - j)
                reward_transform[j - 1] += reward_transform[j] * self.future_reward_discount ** exponent
            i = i_end
            transformed_reward += reward_transform
        transformed_reward = np.array(transformed_reward)
        self.value_function_learner.fit(transformed_observations, transformed_reward.reshape(-1, 1))
        return self.value_function_learner

    def estimate_advantage(self, previous_observation, streaming_enviroment,proposed_action_idx):
        last_level = previous_observation['current_level'][-1]
        quality_choices = np.arange(last_level - self.max_quality_change, last_level + self.max_quality_change + 1)
        # Limit the choices to possible quality shifts
        quality_choices = np.clip(quality_choices, a_min=0, a_max=streaming_enviroment.max_quality_level)
        reward_list = []
        for next_level in quality_choices:
            streaming_enviroment_state_save = streaming_enviroment.save_state()
            observation, current_reward, end_of_video, info = streaming_enviroment.get_video_chunk(next_level)
            observation_transformed = np.array(
                self.value_function_learner.extract_features_observation(observation)).reshape(1, -1)
            observation_transformed = pd.DataFrame(observation_transformed,
                                                   columns=self.value_function_learner.extract_features_names())
            observation_future_reward = self.value_function_learner.predict(observation_transformed).flatten()[0]

            reward_list.append(observation_future_reward)
            streaming_enviroment.set_state(streaming_enviroment_state_save)
        if self.weight_samples_method == 'Divergence':
            advantage = np.max(reward_list) - np.min(reward_list)  ### Divergence
        elif self.weight_samples_method == 'Viper':
            advantage = reward_list[proposed_action_idx] - np.min(reward_list)  ### VIPER
        else:
            raise ValueError('Use weighting method in %s' % self.valid_weight_samples_method)
        #
        # advantage = reward_list[proposed_action_idx] - np.mean(reward_list) ### v4 (MY STUFF)
        advantage = advantage.flatten()[0]
        return max(advantage, 0)
    def clone_from_expert(self, expert_algorithm: ABRPolicy, streaming_enviroment, trace_list,
                          video_csv_list,expert_trajectory,show_progress=False, log_steps=False):

        self.policy_history = {}
        trace_list = np.array(trace_list)
        video_csv_list = np.array(video_csv_list)
        train_idx, test_idx = train_test_split(np.arange(len(video_csv_list)),
                                               test_size=self.validation_split)
        training_traces = trace_list[train_idx]
        training_video = video_csv_list[train_idx]
        imitator_generator = TrajectoryVideoStreaming(expert_algorithm, streaming_enviroment,
                                                      trace_list=trace_list[test_idx],
                                                      video_csv_list=video_csv_list[test_idx])
        expert_evaluation_validation, expert_trajectory_validation = imitator_generator.create_trajectories()
        self.fit_value_function(expert_algorithm, streaming_enviroment, training_traces, training_video)

        iteration = 1
        training_data = list(zip(training_traces, training_video))
        iterations = range(self.training_epochs)
        if show_progress:
            iterations = tqdm(iterations, desc='Training progress')
        weight_filepaths = []
        random_action_probability = self.random_action_probability_init
        self.fit_clustering_scorer(expert_trajectory)

        for cloning_iteration in iterations:
            showcase_list = []
            trace, video_csv = training_data[np.random.randint(0, len(training_data))]
            print('Current random instance %s' % trace)  # Take a random instance
            streaming_enviroment.set_new_enviroment(trace, video_csv)
            previous_quality = 0
            current_quality = 0
            current_quality_expert = 0
            action_idx_list = []
            previous_observation = streaming_enviroment.generate_observation_dictionary()
            del previous_observation['streaming_environment']  # We can't make use of this in the trajectory
            video_finished = False
            action_idx = 2
            while not video_finished:
                advantage = self.estimate_advantage(previous_observation, streaming_enviroment,action_idx)
                observation, reward, video_finished, info = streaming_enviroment.get_video_chunk(
                    current_quality)
                switch = current_quality_expert - previous_quality
                action_idx = self.abr_policy_learner.map_switch_idx(switch)
                previous_quality = current_quality
                if video_finished:
                    # Add the last observation
                    del observation['streaming_environment']  # We can't make use of this in the trajectory

                    showcase_list.append((previous_observation, action_idx, advantage))
                    break
                is_random = np.random.random() <= random_action_probability
                if iteration == 1 or is_random:
                    rnd_switch = np.random.randint(-self.abr_policy_learner.max_quality_change,
                                                   self.abr_policy_learner.max_quality_change)
                    current_quality = np.clip(current_quality + rnd_switch,
                                              a_min=0,
                                              a_max=streaming_enviroment.max_quality_level)
                else:
                    current_quality = self.abr_policy_learner.next_quality(observation, reward)

                current_quality_expert = expert_algorithm.next_quality(observation, reward)
                del observation['streaming_environment']
                showcase_list.append((previous_observation, action_idx, advantage))
                action_idx_list.append(action_idx)
                previous_observation = observation
            # if log_steps :
            #    print('Actions %s' % str(pd.value_counts(action_idx_list)))
            if iteration != 1:  # We start decreasing the probability once the we actually using the decision tree
                random_action_probability *= self.random_action_probability_decay
                random_action_probability = max(random_action_probability,
                                                self.min_random_action_probability)  # Linear random action probability decay
            self.data_collection_list.append(showcase_list)
            self.retrain()
            iteration += 1
            scoring_history, behavioural_cloning_evaluation = self.score(expert_evaluation_validation,
                                                                         expert_trajectory_validation,
                                                                         streaming_enviroment,
                                                                         trace_list[test_idx],
                                                                         video_csv_list[test_idx],add_data = False)
            for k, v in scoring_history.items():
                if k in self.policy_history:
                    self.policy_history[k] += scoring_history[k]
                else:
                    self.policy_history[k] = scoring_history[k]
            if log_steps:
                logging_folder = 'logging_%s' % self.abr_name
                if not os.path.exists(logging_folder):
                    os.makedirs(logging_folder)
                with open(os.path.join(logging_folder, 'logging_iteration_%d' % cloning_iteration),
                          'wb') as output_file:
                    dill.dump(behavioural_cloning_evaluation, output_file)

            weight_filepath = self.rnd_id + '_policy_network_iteration_%d.h5' % cloning_iteration
            with open(weight_filepath, 'wb') as output_file:
                dill.dump(self.abr_policy_learner, output_file)
            weight_filepaths.append(weight_filepath)
        best_iteration = self.opt_policy_opt_operator(self.policy_history[self.opt_policy_value_name])
        with open(weight_filepaths[best_iteration], 'rb') as input_file:
            self.abr_policy_learner = dill.load(input_file)
        logger.info('Restoring best iteration %d' % best_iteration)
        for path in weight_filepaths:
            os.remove(path)

    def extract_array_from_list(self, data_list):
        X = []
        Y = []
        Z = []
        for showcase_list in data_list:
            for obs, label, advantage in showcase_list:
                obs = self.abr_policy_learner.extract_features_observation(obs)
                X.append(obs)
                Y.append(label)
                Z.append(advantage)
        X = pd.DataFrame(np.array(X), columns=self.abr_policy_learner.extract_features_names())
        return X, np.array(Y), np.array(Z)

    @abstractmethod
    def retrain(self):
        pass

    def reset(self):
        super().reset()

    def score(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
              video_csv_list, add_data=False):
        expert_trajectory.convert_list()
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list,
                                                                               video_csv_list=video_csv_list)
        state_t = np.array([self.abr_policy_learner.extract_features_observation(state_t) for state_t, _, _ in
                            tqdm(expert_trajectory.trajectory_list, desc='transforming')])
        state_t = pd.DataFrame(state_t, columns=self.abr_policy_learner.extract_features_names())
        expert_action = expert_trajectory.trajectory_action_t_arr
        approx_action = self.abr_policy_learner.predict(state_t)

        behavioural_cloning_evaluation, behavioural_cloning_evaluation_trajectory = behavioural_cloning_trace_generator_testing.create_trajectories(
            random_action_probability=0, cores_avail=1)
        return self.score_comparison(expert_evaluation=expert_evaluation,
                                     expert_trajectory=expert_trajectory,
                                     expert_action=expert_action,
                                     approx_evaluation=behavioural_cloning_evaluation,
                                     approx_trajectory=behavioural_cloning_evaluation_trajectory,
                                     approx_action=approx_action, add_data=add_data)

class VIPERCloner(ABRCloner):

    def __init__(self, abr_name, max_quality_change, deterministic, training_epochs,
                 abr_policy_learner: ABRPolicyLearner, value_function_learner: ABRPolicyValueFunctionLearner,
                 validation_split=0.2, future_reward_discount=0.99, random_action_probability_init=.9, cores_avail=1,
                 min_random_action_probability=0.05, exploration_percentage=0.5, n_training_samples=3000,
                 weight_samples_method = 'Viper'):
        """
        Implementation of the VIPER imitation policy proposed in https://papers.nips.cc/paper/7516-verifiable-reinforcement-learning-via-policy-extraction
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param training_epochs:
        :param abr_policy_learner:
        :param value_function_learner:
        :param validation_split:
        :param future_reward_discount:
        :param random_action_probability_init:
        :param cores_avail:
        :param min_random_action_probability:
        :param exploration_percentage:
        :param n_training_samples:
        :param weight_samples_method:
        """
        super().__init__(abr_name, max_quality_change, deterministic, training_epochs, abr_policy_learner,
                         value_function_learner, validation_split, future_reward_discount,
                         random_action_probability_init, cores_avail, min_random_action_probability,
                         exploration_percentage,weight_samples_method = weight_samples_method)
        self.n_training_samples = n_training_samples

    def copy(self):
        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.abr_policy_learner, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            abr_policy_learner_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.value_function_learner, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            value_function_learner_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        return VIPERCloner(self.abr_name,
                           self.max_quality_change,
                           self.deterministic,
                           self.training_epochs,
                           abr_policy_learner_copy,
                           value_function_learner_copy,
                           self.validation_split, self.future_reward_discount,
                           self.random_action_probability_init, self.cores_avail,
                           self.min_random_action_probability, self.exploration_percentage, self.n_training_samples)

    def retrain(self):
        """
        We need to weigh the samples by their importance
        :return:
        """
        X_train, Y_train, advantage_train = self.extract_array_from_list(self.data_collection_list)
        n_samples = len(X_train)
        advantage_train -= np.min(advantage_train)  # Center around 0
        advantage_train = advantage_train / sum(advantage_train)
        choice_idx = np.random.choice(np.arange(n_samples), size=self.n_training_samples, p=advantage_train)
        self.abr_policy_learner.fit(X_train.iloc[choice_idx], Y_train[choice_idx])


class DAGGERCloner(ABRCloner):
    """
    Straight implementation of https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf
    """

    def retrain(self):
        X_train, Y_train, _ = self.extract_array_from_list(self.data_collection_list)
        self.abr_policy_learner.fit(X_train, Y_train)

    def copy(self):
        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.abr_policy_learner, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            abr_policy_learner_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.value_function_learner, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            value_function_learner_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        return DAGGERCloner(self.abr_name,
                            self.max_quality_change,
                            self.deterministic,
                            self.training_epochs,
                            abr_policy_learner_copy,
                            value_function_learner_copy,
                            self.validation_split, self.future_reward_discount,
                            self.random_action_probability_init, self.cores_avail,
                            self.min_random_action_probability, self.exploration_percentage)
