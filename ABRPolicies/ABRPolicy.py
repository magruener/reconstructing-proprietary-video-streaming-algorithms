import logging
import string
from abc import abstractmethod
from random import choice
from typing import Dict

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from scipy.stats import entropy
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

LOGGING_LEVEL = logging.INFO
handler = logging.StreamHandler()
handler.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(handler)

N_BINS_DIST = 15 ## KL divergence between clustering scores how many bins do we use to estimate the probability distribution


class ABRPolicyClusteringFunctionLearner:

    def __init__(self, abr_name, max_quality_change, clustering):
        """
        Helper class which contains the clustering function;
        :param abr_name: what is the name of the clustering approach
        :param max_quality_change: how many quality level changes are allowed
        :param clustering: clustering function which returns -1 for outliers and 1 for inliers when we call predict on it
        """
        self.abr_name = abr_name
        self.max_quality_change = max_quality_change
        self.clustering = clustering
        self.output_shape_learned = []
        self.feature_names_learned = []
        self.saved_fitting_data = []

    def add_fit(self, fitting_data, sample_weight=None):
        if len(self.saved_fitting_data) == 0:
            self.saved_fitting_data = fitting_data
            self.saved_sample_weight = sample_weight
        else:
            if sample_weight is None and self.saved_sample_weight is not None:
                sample_weight = np.array([1.] * len(fitting_data))
                self.saved_sample_weight = np.vstack([sample_weight, self.saved_sample_weight])
            elif sample_weight is not None and self.saved_sample_weight is None:
                self.saved_sample_weight = np.array([1.] * len(self.saved_fitting_data))
                self.saved_sample_weight = np.vstack([sample_weight, self.saved_sample_weight])
            elif sample_weight is None and self.saved_sample_weight is None:
                self.saved_sample_weight = None
            else:
                self.saved_sample_weight = np.vstack([sample_weight, self.saved_sample_weight])
            self.saved_fitting_data = np.vstack([fitting_data, self.saved_fitting_data])

        self.clustering.fit(self.saved_fitting_data, self.saved_sample_weight)

    def fit(self, fitting_data, sample_weight=None):
        self.clustering.fit(fitting_data, sample_weight)

    def predict(self, prediction_data):
        prediction = self.clustering.predict(prediction_data)
        is_not_set = np.isnan(prediction) | np.isinf(prediction)
        prediction[is_not_set] = 1.
        return prediction

    def score_sample(self, prediction_data):
        scoring = self.clustering.decision_function(prediction_data)
        is_not_set = np.isnan(scoring) | np.isinf(scoring)
        scoring[is_not_set] = 1.
        return scoring

    def extract_features_names(self):
        return self.feature_names_learned

    def extract_features_observation(self, state_t):
        feature_arr = []
        feature_names_set = True
        if len(self.feature_names_learned) == 0:
            feature_names_set = False
        for k, v in state_t.items():
            if 'streaming_environment' not in k:  # We don't want the enviroment itself
                if not feature_names_set:
                    self.feature_names_learned += [k + '_-%d' % i for i in range(1, len(v) + 1)][::-1]
                feature_arr += v
        if not feature_names_set:
            self.output_shape_learned = len(feature_arr)
        assert self.output_shape_learned == len(feature_arr), 'Somehow we have now more features'
        return feature_arr


class ABRPolicy:

    def __init__(self,
                 abr_name,
                 max_quality_change,
                 deterministic
                 ):
        """
        Base class for ABR policy
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        """
        assert max_quality_change > 0, 'We have to be able to change at least for 1 quality level at a time'
        self.max_quality_change = max_quality_change
        self.quality_change_arr = np.arange(-max_quality_change, max_quality_change + 1)
        self.abr_name = abr_name
        self.deterministic = deterministic
        self.likelihood_last_decision_val = np.array([0] * (self.max_quality_change * 2 + 1))
        self.likelihood_last_decision_val[len(self.likelihood_last_decision_val) // 2] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

        self.early_stopping = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        self.rnd_string_length = 15
        self.rnd_id = self.randomString(self.rnd_string_length)
        self.clustering_scorer = ABRPolicyClusteringFunctionLearner('Isolation Forest',
                                                                    max_quality_change=max_quality_change,
                                                                    clustering=IsolationForest(contamination='auto'))
        self.opt_policy_value_name = 'f1_macro_clustering_score'
        self.opt_policy_opt_direction = 'max'
        if self.opt_policy_opt_direction == 'max':
            self.opt_policy_opt_operator = np.argmax
        elif self.opt_policy_opt_direction == 'min':
            self.opt_policy_opt_operator = np.argmin
        else:
            raise ValueError('Choose correct policy opt operator %s is not supported' % self.opt_policy_opt_direction)

    def impute_NaN_inplace(self, dataframe, context=''):
        if np.isnan(dataframe).sum().sum() != 0:
            print('Input Dataframe %s contains NaN Values' % context)
            dataframe.fillna(dataframe.mean(), inplace=True)

    def transform_trajectory(self, trajectory):
        transformed_observations = np.array(
            [self.clustering_scorer.extract_features_observation(state_t) for state_t, _, _ in
             tqdm(trajectory.trajectory_list, desc='transforming')])
        transformed_observations = pd.DataFrame(transformed_observations,
                                                columns=self.clustering_scorer.extract_features_names())
        self.impute_NaN_inplace(transformed_observations, 'Clustering Scorer')
        return transformed_observations

    def fit_clustering_scorer(self, trajectory):
        transformed_observations = self.transform_trajectory(trajectory)
        self.clustering_scorer.add_fit(transformed_observations)

    def predict_clustering_evaluation(self, evaluation, trajectory):
        """
        Uses the predict function of  the assigned ABRPolicyClusteringFunctionLearner
        1. transforms raw observations into feature space
        2. predict the anomaly score of the given observation (is it -1 for in or +1 for out)
        3. Assign the observation a reward pertaining to whether it is similiar to the training data or not
        :param evaluation:
        :param trajectory:
        :return:
        """
        transformed_observations = self.transform_trajectory(trajectory)
        anomaly_prediction = self.clustering_scorer.predict(transformed_observations)
        len_eval_total = [len(ev_df.streaming_session_evaluation) for ev_df in evaluation]
        assert np.isnan(anomaly_prediction).sum() == 0, anomaly_prediction
        assert len(anomaly_prediction) == sum(len_eval_total), 'Wrong number of comparing instances %d != %d' % (
            len(anomaly_prediction), sum(len_eval_total))

        i = 0
        transformed_reward = []
        for ev_df in evaluation:
            i_end = i + len(ev_df.streaming_session_evaluation)
            reward_transform = list(anomaly_prediction[i:i_end].copy())
            transformed_reward += [reward_transform]
            # print(i,i_end,'length_of_evaluation_thingy')
            i = i_end
        return transformed_reward

    def score_clustering_evaluation(self, evaluation, trajectory):
        """
        Uses the score_sample function of  the assigned ABRPolicyClusteringFunctionLearner
        1. transforms raw observations into feature space
        2. score the anomaly score of the given observation
        3. Assign the observation a reward pertaining to whether it is similiar to the training data or not
        :param evaluation:
        :param trajectory:
        :return:
        """
        transformed_observations = self.transform_trajectory(trajectory)
        anomaly_prediction = self.clustering_scorer.score_sample(transformed_observations)
        len_eval_total = [len(ev_df.streaming_session_evaluation) for ev_df in evaluation]
        assert np.isnan(anomaly_prediction).sum() == 0, anomaly_prediction
        assert len(anomaly_prediction) == sum(len_eval_total), 'Wrong number of comparing instances %d != %d' % (
        len(anomaly_prediction), sum(len_eval_total))
        i = 0
        transformed_reward = []
        for ev_df in evaluation:
            i_end = i + len(ev_df.streaming_session_evaluation)
            reward_transform = list(anomaly_prediction[i:i_end].copy())
            transformed_reward += [reward_transform]
            i = i_end
        return transformed_reward

    def score_comparison(self,
                         expert_evaluation,
                         expert_trajectory,
                         expert_action,
                         approx_evaluation,
                         approx_trajectory,
                         approx_action,
                         add_data=False,
                         ):
        """
        Calculate the scores on the evaluation
        :param expert_evaluation: reference
        :param expert_trajectory: reference
        :param expert_action: reference
        :param approx_evaluation: learner evaluation on the same data
        :param approx_trajectory: learner evaluation on the same data
        :param approx_action: learner evaluation on the same data
        :param add_data: Should the expert trajectory be added to clustering and considered as only reference observations
        :return: dictionary containing the evaluation,approx_evaluation
        """
        beta = 1.
        policy_history = {}
        accuracy_score = metrics.accuracy_score(y_pred=approx_action, y_true=expert_action)
        weighted_f1_score = metrics.f1_score(y_pred=approx_action, y_true=expert_action, average='weighted')
        macro_f1_score = metrics.f1_score(y_pred=approx_action, y_true=expert_action, average='macro')
        micro_f1_score = metrics.f1_score(y_pred=approx_action, y_true=expert_action, average='micro')
        mae_score = metrics.mean_absolute_error(y_pred=approx_action, y_true=expert_action)
        policy_history['accuracy_score'] = [accuracy_score]
        policy_history['weighted_f1_score'] = [weighted_f1_score]
        policy_history['macro_f1_score'] = [macro_f1_score]
        policy_history['micro_f1_score'] = [micro_f1_score]
        policy_history['mae_score'] = [mae_score]
        ########################################################################################################
        ###### Fit Clustering Method
        if add_data:
            self.fit_clustering_scorer(expert_trajectory)
        session_expert_len = [len(frame.streaming_session_evaluation) for frame in expert_evaluation]
        mbit_session_approx = [np.mean(session.streaming_session_evaluation.encoded_mbitrate.values[:session_len]) for
                               session, session_len in zip(
                approx_evaluation, session_expert_len)]
        mbit_session_expert = [np.mean(session.streaming_session_evaluation.encoded_mbitrate.values[:session_len]) for
                               session, session_len in zip(
                expert_evaluation, session_expert_len)]
        rebuffering_session_approx = [
            np.mean(session.streaming_session_evaluation.rebuffering_seconds.values[:session_len]) for
            session, session_len in zip(
                approx_evaluation, session_expert_len)]
        rebuffering_session_expert = [
            np.mean(session.streaming_session_evaluation.rebuffering_seconds.values[:session_len]) for
            session, session_len in zip(
                expert_evaluation, session_expert_len)]
        mbit_switch_session_approx = [
            np.mean(session.streaming_session_evaluation.encoded_mbitrate.diff().fillna(0).abs().values[:session_len])
            for session, session_len in zip(
                approx_evaluation, session_expert_len)]
        mbit_switch_session_expert = [
            np.mean(session.streaming_session_evaluation.encoded_mbitrate.diff().fillna(0).abs().values[:session_len])
            for session, session_len in zip(
                expert_evaluation, session_expert_len)]

        policy_history['avg_mbit_approx'] = [np.mean(mbit_session_approx)]
        policy_history['std_mbit_approx'] = [np.std(mbit_session_approx)]
        policy_history['avg_mbit_expert'] = [np.mean(mbit_session_expert)]
        policy_history['std_mbit_expert'] = [np.std(mbit_session_expert)]
        policy_history['avg_rebuffering_approx'] = [np.mean(rebuffering_session_approx)]
        policy_history['std_rebuffering_approx'] = [np.std(rebuffering_session_approx)]
        policy_history['avg_rebuffering_expert'] = [np.mean(rebuffering_session_expert)]
        policy_history['std_rebuffering_expert'] = [np.std(rebuffering_session_expert)]
        policy_history['avg_mbit_switch_approx'] = [np.mean(mbit_switch_session_approx)]
        policy_history['std_mbit_switch_approx'] = [np.std(mbit_switch_session_approx)]
        policy_history['avg_mbit_switch_expert'] = [np.mean(mbit_switch_session_expert)]
        policy_history['std_mbit_switch_expert'] = [np.std(mbit_switch_session_expert)]
        ########################################################################################################
        #####
        approx_clustering = self.predict_clustering_evaluation(approx_evaluation, approx_trajectory)
        assert len(approx_clustering) == len(
            session_expert_len), 'The length doesnt match %d (approx) != %d (expert)' % (
            len(approx_clustering), len(session_expert_len))

        approx_clustering_session = [np.nanmean(session[:session_len]) for session, session_len in zip(
            approx_clustering, session_expert_len)]
        policy_history['avg_clustering_approx'] = [np.nanmean(approx_clustering_session)]

        approx_clustering_score = self.score_clustering_evaluation(approx_evaluation, approx_trajectory)
        approx_clustering_score_session = [np.nanmean(session[:session_len]) for session, session_len in zip(
            approx_clustering_score, session_expert_len)]
        policy_history['avg_clustering_score_approx'] = [np.nanmean(approx_clustering_score_session)]

        reference_clustering_score = self.score_clustering_evaluation(expert_evaluation, expert_trajectory)
        reference_clustering_score_session = [np.nanmean(session) for session in reference_clustering_score]
        policy_history['avg_clustering_score_expert'] = [np.nanmean(reference_clustering_score_session)]
        hist_expert, bin_edges_reference = np.histogram(reference_clustering_score_session, bins=N_BINS_DIST)
        hist_reference, _ = np.histogram(approx_clustering_score_session, bins=bin_edges_reference)
        hist_expert = hist_expert.astype(float) + 1.
        hist_reference = hist_reference.astype(float) + 1.
        hist_expert = hist_expert / hist_expert.sum()
        hist_reference = hist_reference / hist_reference.sum()
        kld_dist = entropy(hist_reference, hist_expert) + entropy(hist_expert, hist_reference)
        policy_history['clustering_score_kl_divergence'] = [kld_dist]

        policy_history['mult_macro_clustering_score'] = [policy_history['macro_f1_score'] * policy_history[
            'avg_clustering_approx']]
        policy_history['add_macro_clustering_score'] = [policy_history['macro_f1_score'] + policy_history[
            'avg_clustering_approx']]
        policy_history['f1_macro_clustering_score'] = [(1 + beta ** 2) * policy_history[
            'mult_macro_clustering_score'] / (beta ** 2 * policy_history['macro_f1_score'] + policy_history[
            'avg_clustering_approx'])]

        return policy_history, approx_evaluation

    def randomString(self, stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase

    def likelihood_last_decision(self):
        return self.likelihood_last_decision_val

    @abstractmethod
    def reset(self):
        self.likelihood_last_decision_val = np.array([0] * (self.max_quality_change * 2 + 1))
        self.likelihood_last_decision_val[len(self.likelihood_last_decision_val) // 2] = 1
        self.likelihood_last_decision_val = self.likelihood_last_decision_val.reshape(1, -1)

    def map_switch_idx(self, switch: int):
        return list(self.quality_change_arr).index(switch)

    def softmax(self, value_list):
        """Compute softmax values for each sets of scores in input_tuple."""
        return np.exp(value_list) / np.sum(np.exp(value_list), axis=0)

    @abstractmethod
    def next_quality(self, observation, reward):
        pass

    @abstractmethod
    def copy(self):
        pass

    def keep_last_entry(self, history: Dict):
        history_copy = {}
        for k, v in history.items():
            history_copy[k] = [history[k][-1]]
        return history_copy

    def select_reward_to_propagate(self, reward_list, inverse=False):
        reward_list = sorted(reward_list)  # [(quality_idx,quality_reward) ... )
        probability_action = [v for k, v in reward_list]
        probability_action = self.softmax(probability_action)
        reward_list_sorted_value = sorted(reward_list, key=lambda value_pair: value_pair[1])
        if self.deterministic:
            if inverse:
                min_value = reward_list_sorted_value[0][1]
                min_next_quality = reward_list_sorted_value[0][0]
                # min_next_probability = min(probability_action)
                return min_value, min_next_quality, probability_action
            else:
                max_value = reward_list_sorted_value[-1][1]
                max_next_quality = reward_list_sorted_value[-1][0]
                # max_next_probability = max(probability_action)
                return max_value, max_next_quality, probability_action
        else:
            prob_index = np.random.choice(np.arange(len(probability_action)), size=1, p=probability_action)
            prob_value = reward_list[prob_index][1]
            prob_next_quality = reward_list[prob_index][0]
            # prob_next_probability = probability_action[prob_index]
            return prob_value, prob_next_quality, probability_action.reshape(1, -1)
