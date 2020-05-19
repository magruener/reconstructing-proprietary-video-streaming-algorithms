"""
@inproceedings{torabi2018bco,
  author = {Faraz Torabi and Garrett Warnell and Peter Stone},
  title = {Behavioral Cloning from Observation},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  year = {2018}
}
https://github.com/tsujuifu/pytorch_bco
"""
import logging
import os

import dill
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import GRU, concatenate, Dense
from tensorflow.python.keras.utils import to_categorical
from tqdm import tqdm
from xgboost import XGBRegressor

from ABRPolicies.ABRPolicy import ABRPolicy
from BehaviourCloning.MLABRPolicy import ABRPolicyLearner, ABRPolicyValueFunctionLearner
from SimulationEnviroment.SimulatorEnviroment import TrajectoryVideoStreaming, Trajectory

LOGGING_LEVEL = logging.DEBUG
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)
N_BINS_DIST = 15
RANDOM_SEED = 42


class KerasPolicy:

    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 action_dimension,
                 drop_prob=0.2):
        """
        Build model, predict only next quality
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param hidden_dim:
        :param action_dimension:
        :param drop_prob:
        """
        self.policy_past_input = Input(shape=(None, past_measurement_dimensions))
        self.policy_past_GRU = GRU(units=hidden_dim,
                                   return_sequences=False, dropout=drop_prob)(self.policy_past_input)
        self.policy_future_input = Input(shape=(None, future_measurements_dimensions))
        self.policy_future_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)(
            self.policy_future_input)
        self.policy_dense1 = Dense(units=hidden_dim, activation="relu")
        self.policy_dense2 = Dense(activation="softmax", units=action_dimension)
        concatenated = concatenate([self.policy_past_GRU, self.policy_future_GRU])
        concatenated = self.policy_dense1(concatenated)
        self.policy_action_output = self.policy_dense2(concatenated)
        self.model = Model(inputs=[self.policy_past_input, self.policy_future_input], outputs=self.policy_action_output)
        self.model.compile(loss="categorical_crossentropy", optimizer='adam')


class BehavioralCloning(ABRPolicy):

    def reset(self):
        super().reset()

    def __init__(self, classifier: ABRPolicyLearner, validation_split=0.2, cores_avail=1, weight_samples=False,
                 weight_samples_method='Divergence'):
        """
        https://papers.nips.cc/paper/7516-verifiable-reinforcement-learning-via-policy-extraction : VIPER paper

        Copy behaviour
        :param classifier:
        :param validation_split:
        :param cores_avail:
        :param weight_samples: Do you enable the weighting method ?
        :param weight_samples_method: Assigns importance to choosing the specific action in the training set.
        VIPER : Weigh action by difference in reward you gain from choosing this action and the worst action
        Divergence : Weight action by difference in reward you gain from choosing the best action and the worst action in this siutation.
        """
        self.valid_weight_samples_method = ['Divergence', 'Viper']
        assert weight_samples_method in self.valid_weight_samples_method, 'Use weighting method in %s' % self.valid_weight_samples_method
        super().__init__(classifier.abr_name, classifier.max_quality_change, classifier.deterministic)
        self.weight_samples_method = weight_samples_method
        self.weight_samples = weight_samples
        self.validation_split = validation_split
        self.classifier = classifier
        self.policy_history = None
        self.cores_avail = cores_avail
        self.anomaly_scorer = IsolationForest(contamination='auto', random_state=RANDOM_SEED)
        """
        Predict future reward from current observation
        """
        self.value_function_learner = ABRPolicyValueFunctionLearner(abr_name='XGB Regressor',
                                                                    max_quality_change=classifier.max_quality_change,
                                                                    regressor=XGBRegressor(random_state=RANDOM_SEED))
        self.future_reward_discount = 0.99

    def next_quality(self, observation, reward):
        return self.classifier.next_quality(observation, reward)

    def copy(self):
        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.classifier, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            classifier_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        return BehavioralCloning(classifier_copy, validation_split=self.validation_split, cores_avail=self.cores_avail)

    def fit_value_function(self, to_imitate_evaluation, to_imitate_trajectory):
        transformed_observations = np.array(
            [self.value_function_learner.extract_features_observation(state_t) for state_t, _, _ in
             tqdm(to_imitate_trajectory.trajectory_list, desc='transforming')])
        transformed_observations = pd.DataFrame(transformed_observations,
                                                columns=self.value_function_learner.extract_features_names())
        self.impute_NaN_inplace(transformed_observations, 'sample weighting')
        self.anomaly_scorer.fit(transformed_observations)
        ################################
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

    def estimate_advantage_observation(self, previous_observation, streaming_enviroment, proposed_action_idx):
        """
        Estimate the advantage
        :param previous_observation:
        :param streaming_enviroment:
        :param proposed_action_idx:
        :return:
        """
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
        # max_gain =  #- previous_observation_future_reward
        # max_loss =  #- previous_observation_future_reward
        if self.weight_samples_method == 'Divergence':
            advantage = np.max(reward_list) - np.min(reward_list)  ### v2 (MY STUFF)
        elif self.weight_samples_method == 'Viper':
            advantage = reward_list[proposed_action_idx] - np.min(reward_list)  ### v3 (VIPER)
        else:
            raise ValueError('Use weighting method in %s' % self.valid_weight_samples_method)
        #
        # advantage = reward_list[proposed_action_idx] - np.mean(reward_list) ### v4 (MY STUFF)
        advantage = advantage.flatten()[0]
        return max(advantage, 0)

    def estimate_advantage_frame(self, action_frame, bw_trace_path, video_information_path, stream_env):
        iterator = 0
        stream_env.set_new_enviroment(bw_trace_path, video_information_path)
        video_finished = False
        advantage_list = []
        observation = stream_env.generate_observation_dictionary()
        previous_quality = 0
        while not video_finished and iterator < len(action_frame.streaming_session_evaluation):
            current_quality = action_frame.streaming_session_evaluation.iloc[iterator].current_level
            current_quality = int(current_quality)
            switch = current_quality - previous_quality
            proposed_action_idx = self.map_switch_idx(switch)
            advantage_sample = self.estimate_advantage_observation(observation,
                                                                   streaming_enviroment=stream_env,
                                                                   proposed_action_idx=proposed_action_idx)
            previous_quality = current_quality
            observation, reward, video_finished, info = stream_env.get_video_chunk(
                current_quality)
            advantage_list.append(advantage_sample)
            iterator += 1
        return np.array(advantage_list)

    def calculate_reference_reward(self, expert_evaluation, test_idx):
        return [frame.reward.mean() for frame in [expert_evaluation[i].streaming_session_evaluation for i in test_idx]]

    def clone_from_trajectory(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
                              video_csv_list, log_steps=False):
        """
        Main function which will try to imitate the expert actions. Simply imitate the actions of an expert in a given situation
        :param expert_evaluation:
        :param expert_trajectory:
        :param streaming_enviroment:
        :param trace_list:
        :param video_csv_list:
        :param log_steps:
        :return:
        """
        logging_iteration = 0
        # Select the training/validation traces
        self.policy_history = None
        trace_list = np.array(trace_list)
        video_csv_list = np.array(video_csv_list)
        expert_evaluation = np.array(expert_evaluation)
        train_idx, test_idx = train_test_split(np.arange(len(expert_evaluation)),
                                               test_size=self.validation_split, random_state=RANDOM_SEED)
        trace_video_pair_list = [f.name for f in expert_evaluation[train_idx]]
        expert_trajectory_train = expert_trajectory.extract_trajectory(trace_video_pair_list=trace_video_pair_list)
        expert_trajectory_train.convert_list()
        trace_video_pair_list = [f.name for f in expert_evaluation[test_idx]]
        self.fit_clustering_scorer(expert_trajectory)
        ###########
        if self.weight_samples:
            self.fit_value_function(to_imitate_evaluation=expert_evaluation[train_idx],
                                    to_imitate_trajectory=expert_trajectory_train)
            advantage = []
            ## Add advante to the training data
            for index in train_idx:
                advantage += list(self.estimate_advantage_frame(expert_evaluation[index], trace_list[index],
                                                                video_csv_list[index], streaming_enviroment))
            advantage = np.array(advantage).flatten()
            advantage = advantage + np.min(
                advantage)  # We smooth the estimate so that the low advantages are a bit bolstered
            ## NO NEGATIV WEIGHTS !
            assert (advantage < 0).sum() == 0, 'advantage should be non negative everywhere'
        #### estimate advantage on the training samples

        expert_trajectory_test = expert_trajectory.extract_trajectory(trace_video_pair_list=trace_video_pair_list)

        state_t = np.array([self.classifier.extract_features_observation(state_t) for state_t, _, _ in
                            tqdm(expert_trajectory_train.trajectory_list, desc='transforming')])
        state_t = pd.DataFrame(state_t, columns=self.classifier.extract_features_names())
        self.impute_NaN_inplace(state_t)
        expert_action = expert_trajectory_train.trajectory_action_t_arr
        if self.weight_samples:
            self.classifier.fit(state_t, expert_action.ravel(), sample_weight=advantage)
            if log_steps:
                logging_folder = 'logging_%s' % self.abr_name
                if not os.path.exists(logging_folder):
                    os.makedirs(logging_folder)
                    with open(os.path.join(logging_folder, 'advantage_distribution'),
                              'wb') as output_file:
                        dill.dump(advantage, output_file)
        else:
            self.classifier.fit(state_t, expert_action.ravel())

        if self.policy_history is None:
            self.policy_history, behavioural_cloning_evaluation = self.score(expert_evaluation[test_idx],
                                                                             expert_trajectory_test,
                                                                             streaming_enviroment,
                                                                             trace_list[test_idx],
                                                                             video_csv_list[test_idx], add_data=False)
            if log_steps:
                with open(os.path.join(logging_folder, 'logging_iteration_%d' % logging_iteration),
                          'wb') as output_file:
                    dill.dump(behavioural_cloning_evaluation, output_file)

    def score(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
              video_csv_list, add_data=False):
        """
        Wrapper for the base scoring function
        :param expert_evaluation:
        :param expert_trajectory:
        :param streaming_enviroment:
        :param trace_list: Which traces did we evaluate
        :param video_csv_list: Which videos did we evaluate
        :param add_data:
        :return:
        """
        expert_trajectory.convert_list()
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list,
                                                                               video_csv_list=video_csv_list)
        state_t = np.array([self.classifier.extract_features_observation(state_t) for state_t, _, _ in
                            tqdm(expert_trajectory.trajectory_list, desc='transforming')])
        state_t = pd.DataFrame(state_t, columns=self.classifier.extract_features_names())
        self.impute_NaN_inplace(state_t)
        expert_action = expert_trajectory.trajectory_action_t_arr
        approx_action = self.classifier.predict(state_t)
        expert_action = expert_action.ravel()
        behavioural_cloning_evaluation, behavioural_cloning_evaluation_trajectory = behavioural_cloning_trace_generator_testing.create_trajectories(
            random_action_probability=0, cores_avail=1)
        return self.score_comparison(expert_evaluation=expert_evaluation,
                                     expert_trajectory=expert_trajectory,
                                     expert_action=expert_action,
                                     approx_evaluation=behavioural_cloning_evaluation,
                                     approx_trajectory=behavioural_cloning_evaluation_trajectory,
                                     approx_action=approx_action, add_data=add_data)


class BehavioralCloningIterative(ABRPolicy):

    def __init__(self, abr_name, max_quality_change, deterministic, past_measurement_dimensions,
                 future_measurements_dimensions, cloning_epochs, drop_prob=0.1,
                 hidden_dim=32, batch_size_cloning=64, validation_split=0.2, cores_avail=1, balanced=False):
        """
        Behavioral Cloning for Keras (GRU) policy
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param cloning_epochs:
        :param drop_prob:
        :param hidden_dim:
        :param batch_size_cloning:
        :param validation_split:
        :param cores_avail:
        :param balanced:
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.cores_avail = cores_avail
        self.future_measurements_dimensions = future_measurements_dimensions
        self.validation_split = validation_split
        self.balanced = balanced
        self.past_measurement_dimensions = past_measurement_dimensions
        self.n_actions = max_quality_change * 2 + 1
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.policy_network = KerasPolicy(past_measurement_dimensions=self.past_measurement_dimensions,
                                          future_measurements_dimensions=self.future_measurements_dimensions,
                                          hidden_dim=hidden_dim,
                                          action_dimension=self.n_actions,
                                          drop_prob=drop_prob)

        self.policy_history = None
        self.batch_size_cloning = batch_size_cloning
        self.cloning_epochs = cloning_epochs
        self.trajectory_dummy = Trajectory()

    def copy(self):
        copy_ = BehavioralCloningIterative(self.abr_name, self.max_quality_change, self.deterministic,
                                           self.past_measurement_dimensions,
                                           self.future_measurements_dimensions, self.cloning_epochs, self.drop_prob,
                                           self.hidden_dim, self.batch_size_cloning, self.validation_split,
                                           self.cores_avail)
        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        self.policy_network.model.save_weights(filepath=tmp_file_name)
        copy_.policy_network.model.load_weights(tmp_file_name)
        os.remove(tmp_file_name)
        return copy_

    def next_quality(self, observation, reward):
        current_level = observation['current_level'][-1]
        streaming_enviroment = observation['streaming_environment']
        observation = self.trajectory_dummy.scale_observation(
            observation)  # This is important as the learned representation is also scaled
        state_t = [v for k, v in sorted(
            observation.items()) if 'streaming_environment' != k and 'future' not in k]
        state_t = np.array(state_t).T
        state_t = np.expand_dims(state_t, axis=0)

        state_t_future = [v for k, v in sorted(
            observation.items()) if 'streaming_environment' != k and 'future' in k]
        state_t_future = np.array(state_t_future).T
        state_t_future = np.expand_dims(state_t_future, axis=0)

        action_prob = self.policy_network.model.predict([state_t, state_t_future])
        self.likelihood_last_decision_val = max(action_prob)
        if self.deterministic:
            next_quality_switch_idx = np.argmax(action_prob)
        else:
            probability = action_prob
            next_quality_switch_idx = np.random.choice(np.arange(len(probability)), size=1, p=probability)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality_switch_idx], a_min=0,
                               a_max=streaming_enviroment.max_quality_level)
        return next_quality

    def likelihood_last_decision(self):
        return self.likelihood_last_decision_val

    def reset(self):
        pass

    def reset_learning(self):
        self.policy_history = None
        self.policy_network = KerasPolicy(past_measurement_dimensions=self.past_measurement_dimensions,
                                          future_measurements_dimensions=self.future_measurements_dimensions,
                                          hidden_dim=self.hidden_dim,
                                          action_dimension=self.n_actions,
                                          drop_prob=self.drop_prob)

    def calculate_reference_reward(self, expert_evaluation, test_idx):
        return [frame.reward.mean() for frame in [expert_evaluation[i].streaming_session_evaluation for i in test_idx]]

    def clone_from_trajectory(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
                              video_csv_list, log_steps=False):
        self.reset_learning()
        self.policy_history = None
        self.fit_clustering_scorer(expert_trajectory)
        trace_list = np.array(trace_list)
        video_csv_list = np.array(video_csv_list)
        expert_evaluation = np.array(expert_evaluation)
        train_idx, test_idx = train_test_split(np.arange(len(expert_evaluation)),
                                               test_size=self.validation_split * 2.)
        test_idx, validation_idx = train_test_split(test_idx,
                                                    test_size=0.5)
        trace_video_pair_list = [f.name for f in expert_evaluation[train_idx]]
        expert_trajectory_train = expert_trajectory.extract_trajectory(trace_video_pair_list=trace_video_pair_list)
        expert_trajectory_train.convert_list()
        trace_video_pair_list = [f.name for f in expert_evaluation[test_idx]]
        expert_trajectory_test = expert_trajectory.extract_trajectory(trace_video_pair_list=trace_video_pair_list)
        expert_trajectory_test.convert_list()
        trace_video_pair_list = [f.name for f in expert_evaluation[validation_idx]]
        expert_trajectory_validation = expert_trajectory.extract_trajectory(trace_video_pair_list=trace_video_pair_list)
        expert_trajectory_validation.convert_list()

        state_t_training = expert_trajectory_train.trajectory_state_t_arr
        state_t_future_training = expert_trajectory_train.trajectory_state_t_future
        action_training = to_categorical(expert_trajectory_train.trajectory_action_t_arr, self.n_actions)

        state_t_testing = expert_trajectory_test.trajectory_state_t_arr
        state_t_future_testing = expert_trajectory_test.trajectory_state_t_future
        action_testing = to_categorical(expert_trajectory_test.trajectory_action_t_arr, self.n_actions)
        validation_data = ([state_t_testing, state_t_future_testing], action_testing)
        weight_filepaths = []
        keras_class_weighting = None
        self.fit_clustering_scorer(expert_trajectory)
        if self.balanced:
            keras_class_weighting = class_weight.compute_class_weight('balanced',
                                                                      np.unique(action_training.argmax(1)),
                                                                      action_training.argmax(1))
        for cloning_iteration in tqdm(range(self.cloning_epochs), desc='Cloning Epochs'):
            history = self.policy_network.model.fit([state_t_training, state_t_future_training],
                                                    action_training,
                                                    validation_data=validation_data, epochs=1,
                                                    verbose=0, class_weight=keras_class_weighting).history
            if self.policy_history is None:
                self.policy_history = history
            else:
                for k, v in history.items():
                    self.policy_history[k] += history[k]
            scoring_history, behavioural_cloning_evaluation = self.score(expert_evaluation[validation_idx],
                                                                         expert_trajectory_validation,
                                                                         streaming_enviroment,
                                                                         trace_list[validation_idx],
                                                                         video_csv_list[validation_idx])
            if log_steps:
                logging_folder = 'logging_%s' % self.abr_name
                if not os.path.exists(logging_folder):
                    os.makedirs(logging_folder)
                with open(os.path.join(logging_folder, 'logging_iteration_%d' % cloning_iteration),
                          'wb') as output_file:
                    dill.dump(behavioural_cloning_evaluation, output_file)

            for k, v in scoring_history.items():
                if k in self.policy_history:
                    self.policy_history[k] += scoring_history[k]
                else:
                    self.policy_history[k] = scoring_history[k]
            weight_filepath = self.rnd_id + '_policy_network_iteration_%d.h5' % cloning_iteration
            self.policy_network.model.save_weights(filepath=weight_filepath)
            weight_filepaths.append(weight_filepath)
        best_iteration = self.opt_policy_opt_operator(self.policy_history[self.opt_policy_value_name])
        self.policy_network.model.load_weights(weight_filepaths[best_iteration])
        logger.info('Restoring best iteration %d' % best_iteration)
        for path in weight_filepaths:
            os.remove(path)

    def save_model(self, weight_filepath):
        self.policy_network.model.save_weights(filepath=weight_filepath)

    def load_model(self, weight_filepath):
        self.policy_network.model.load_weights(weight_filepath)

    def score(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
              video_csv_list, add_data=False):
        expert_trajectory.convert_list()
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list,
                                                                               video_csv_list=video_csv_list)
        state_t_testing = expert_trajectory.trajectory_state_t_arr
        state_t_future_testing = expert_trajectory.trajectory_state_t_future
        expert_action = expert_trajectory.trajectory_action_t_arr
        approx_action = self.policy_network.model.predict([state_t_testing, state_t_future_testing]).argmax(-1)
        expert_action = expert_action.ravel()
        behavioural_cloning_evaluation, behavioural_cloning_evaluation_trajectory = behavioural_cloning_trace_generator_testing.create_trajectories(
            random_action_probability=0, cores_avail=1)
        return self.score_comparison(expert_evaluation=expert_evaluation,
                                     expert_trajectory=expert_trajectory,
                                     expert_action=expert_action,
                                     approx_evaluation=behavioural_cloning_evaluation,
                                     approx_trajectory=behavioural_cloning_evaluation_trajectory,
                                     approx_action=approx_action, add_data=add_data)
