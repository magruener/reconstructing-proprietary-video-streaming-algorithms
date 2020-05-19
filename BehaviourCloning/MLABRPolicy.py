import os
from abc import ABC, abstractmethod

import dill
import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from ABRPolicies.ABRPolicy import ABRPolicy
from ABRPolicies.ThroughputEstimator import generate_ewma, generate_percentile, ThroughputEstimator, StepEstimator
from BehaviourCloning.GeneticFeatureEngineering import GeneticFeatureGenerator, projection_generator_function

MIN_DIFFERENCE_QUALITY = 2


class ABRPolicyLearner(ABRPolicy, ABC):

    @abstractmethod
    def extract_features_observation(self, state_t):
        pass

    @abstractmethod
    def extract_features_names(self):
        pass

    @abstractmethod
    def fit(self, X, Y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, prediction_data):
        pass

    @abstractmethod
    def score(self, X, Y):
        pass


class ABRPolicyValueFunctionLearner:

    def __init__(self, abr_name, max_quality_change, regressor):
        """
        Base wrapper for the  value prediction learner
        state -> sum(reward to be achieved in the future)
        :param abr_name:
        :param max_quality_change:
        :param regressor:
        """
        self.abr_name = abr_name
        self.max_quality_change = max_quality_change
        self.regressor = regressor
        self.output_shape_learned = []
        self.feature_names_learned = []

    def fit(self, fitting_data, fitting_label, sample_weight=None):
        self.regressor.fit(fitting_data, fitting_label, sample_weight)

    def predict(self, prediction_data):
        return self.regressor.predict(prediction_data)

    def score(self, fitting_data, fitting_label):
        return self.regressor.score(fitting_data, fitting_label)

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


class ABRPolicyRate(ABRPolicyLearner):

    def __init__(self, abr_name, max_quality_change, throughput_predictor: ThroughputEstimator):
        """
        Simple Rate based algorithm
        :param abr_name:
        :param max_quality_change:
        :param throughput_predictor:
        """
        super().__init__(abr_name, max_quality_change, True)
        self.throughput_predictor = throughput_predictor
        self.max_quality_change = max_quality_change
        self.output_shape_learned = []
        self.feature_names_learned = []

    def copy(self):
        _copy = ABRPolicyRate(self.abr_name, self.max_quality_change,
                              self.throughput_predictor.copy())
        _copy.output_shape_learned = self.output_shape_learned
        _copy.feature_names_learned = self.feature_names_learned[:]
        return _copy

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

    def fit(self, X, Y, sample_weight=None):
        print('Does not need to be fitted')

    def predict_proba(self, prediction_data):
        """
        We always return a probability array over quality switches possibility --
        sets the one by the rate algo choosen to 1
        :param prediction_data:
        :return:
        """
        y_pred = []
        columns_bitrate_future = 'future_chunk_bitrate_switch_'
        columns_bitrate_future = [c for c in list(prediction_data) if columns_bitrate_future in c]
        columns_byte_lookahead = str(min(np.unique([c.split('_')[-1] for c in columns_bitrate_future]).astype(int)))
        columns_bitrate_future = [c for c in columns_bitrate_future if c.endswith(columns_byte_lookahead)]
        columns_bitrate_future = sorted(columns_bitrate_future, key=lambda c_key: int(c_key.split('_')[-2]))
        columns_download_time = 'download_time_s'
        columns_download_time = [c for c in list(prediction_data) if columns_download_time in c]
        columns_download_time = sorted(columns_download_time, key=lambda c_key: int(c_key.split('_')[-1]))
        columns_size_byte = 'video_chunk_size_byte'
        columns_size_byte = [c for c in list(prediction_data) if columns_size_byte in c]
        columns_size_byte = sorted(columns_size_byte, key=lambda c_key: int(c_key.split('_')[-1]))
        columns_timestamp_s = 'timestamp_s'
        columns_timestamp_s = [c for c in list(prediction_data) if columns_timestamp_s in c]
        columns_timestamp_s = sorted(columns_timestamp_s, key=lambda c_key: int(c_key.split('_')[-1]))

        for k, row in prediction_data.iterrows():
            bw_history = row[columns_size_byte].values / row[columns_download_time].values
            bw_history = bw_history[row[columns_download_time].values > 0]
            ts_history = row[columns_timestamp_s][row[columns_download_time].values > 0]
            if len(bw_history) == 0:
                likelihood_arr = np.zeros(self.max_quality_change * 2 + 1)
                likelihood_arr[2] = 1.0
                y_pred.append(likelihood_arr)
            else:
                for tput_estimate, timestamp_s in zip(bw_history, ts_history):
                    self.throughput_predictor.add_sample(
                        tput_estimate=tput_estimate, timestamp_s=timestamp_s)
                bandwidth_estimate = self.throughput_predictor.predict_future_bandwidth()
                self.throughput_predictor.reset()
                columns_byte_values = row[columns_bitrate_future].values * 0.125
                columns_byte_values = columns_byte_values / bandwidth_estimate
                columns_byte_values = columns_byte_values <= 1.0
                columns_byte_values = max(np.sum(columns_byte_values) - 1, 0)
                likelihood_arr = np.zeros(self.max_quality_change * 2 + 1)
                likelihood_arr[columns_byte_values] = 1.0
                y_pred.append(likelihood_arr)
        # columns_byte_values = row[columns_bitrate_future].values * 0.125
        # columns_byte_values = columns_byte_values / bandwidth_estimate
        # print(y_pred[-1],bandwidth_estimate,columns_byte_values,row[columns_bitrate_future].values * 0.125)
        return np.array(y_pred)

    def predict(self, prediction_data):
        likelihood = self.predict_proba(prediction_data)
        return likelihood.argmax(-1)

    def score(self, X, Y):
        return accuracy_score(y_pred=self.predict(X), y_true=Y)

    def reset(self):
        self.throughput_predictor.reset()

    def next_quality(self, observation, reward):
        current_level = observation['current_level'][-1]
        streaming_environment = observation['streaming_environment']
        observation_features_extracted = np.array(self.extract_features_observation(observation)).reshape(1, -1)
        observation_features_extracted = pd.DataFrame(observation_features_extracted,
                                                      columns=self.extract_features_names())
        self.likelihood_last_decision_val = self.predict_proba(
            observation_features_extracted)  # Always predict probability to get a likelihood distribution
        next_quality = self.likelihood_last_decision_val.argmax()
        # print("Predicted next quality idx % d" % next_quality)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality], a_min=0,
                               a_max=streaming_environment.max_quality_level)
        return next_quality


class ABRPolicyClassifierSimple(ABRPolicyLearner):

    def __init__(self, abr_name, max_quality_change, deterministic, max_lookahead,
                 max_lookback, classifier, rate_correction=False):
        """
        ABR policy wrapper for a learned classifier which predicts quality switches no feature enginnering
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param max_lookahead:
        :param max_lookback:
        :param classifier:
        :param rate_correction: DEPRECATED
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.max_lookahead = max_lookahead
        self.classifier = classifier
        self.max_lookback = max_lookback
        self.output_shape_learned = []
        self.feature_names_learned = []
        self.rate_correction = rate_correction
        self.throughput_predictor = StepEstimator(consider_last_n_steps=5,
                                                  predictor_function=hmean,
                                                  robust_estimate=False)

    def rate_correct_prediction(self, observation):
        """
        Predict next quality as rate predicted quality. Could be used as safety fallback feature
        Was Deprecated in the final version
        :param observation:
        :return:
        """
        feature_names_learned = []
        feature_arr = []
        for k, v in observation.items():
            if 'streaming_environment' not in k:  # We don't want the enviroment itself
                feature_names_learned += [k + '_-%d' % i for i in range(1, len(v) + 1)][::-1]
                feature_arr += v

        observation = np.array(feature_arr).reshape(1, -1)
        observation = pd.DataFrame(observation, columns=feature_names_learned)
        y_pred = []
        columns_bitrate_future = 'future_chunk_bitrate_switch_'
        columns_bitrate_future = [c for c in list(observation) if columns_bitrate_future in c]
        columns_byte_lookahead = str(min(np.unique([c.split('_')[-1] for c in columns_bitrate_future]).astype(int)))
        columns_bitrate_future = [c for c in columns_bitrate_future if c.endswith(columns_byte_lookahead)]
        columns_bitrate_future = sorted(columns_bitrate_future, key=lambda c_key: int(c_key.split('_')[-2]))
        columns_download_time = 'download_time_s'
        columns_download_time = [c for c in list(observation) if columns_download_time in c]
        columns_download_time = sorted(columns_download_time, key=lambda c_key: int(c_key.split('_')[-1]))
        columns_size_byte = 'video_chunk_size_byte'
        columns_size_byte = [c for c in list(observation) if columns_size_byte in c]
        columns_size_byte = sorted(columns_size_byte, key=lambda c_key: int(c_key.split('_')[-1]))
        columns_timestamp_s = 'timestamp_s'
        columns_timestamp_s = [c for c in list(observation) if columns_timestamp_s in c]
        columns_timestamp_s = sorted(columns_timestamp_s, key=lambda c_key: int(c_key.split('_')[-1]))
        for k, row in observation.iterrows():
            bw_history = row[columns_size_byte].values / row[columns_download_time].values
            bw_history = bw_history[row[columns_download_time].values > 0]
            ts_history = row[columns_timestamp_s][row[columns_download_time].values > 0]
            if len(bw_history) == 0:
                likelihood_arr = np.zeros(self.max_quality_change * 2 + 1)
                likelihood_arr[2] = 1.0
                y_pred.append(likelihood_arr)
            else:
                for tput_estimate, timestamp_s in zip(bw_history, ts_history):
                    self.throughput_predictor.add_sample(
                        tput_estimate=tput_estimate, timestamp_s=timestamp_s)
                bandwidth_estimate = self.throughput_predictor.predict_future_bandwidth()
                self.throughput_predictor.reset()
                columns_byte_values = row[columns_bitrate_future].values * 0.125
                columns_byte_values = columns_byte_values / bandwidth_estimate
                columns_byte_values = columns_byte_values <= 1.0
                columns_byte_values = min(np.sum(columns_byte_values) - 1, 0)
                likelihood_arr = np.zeros(self.max_quality_change * 2 + 1)
                likelihood_arr[columns_byte_values] = 1.0
                y_pred.append(likelihood_arr)
        return np.array(y_pred)

    def copy(self):
        tmp_file_name = self.rnd_id + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.classifier, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            classifier_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        _copy = ABRPolicyClassifierSimple(self.abr_name, self.max_quality_change,
                                          self.deterministic, self.max_lookahead, self.max_lookback, classifier_copy)
        _copy.output_shape_learned = self.output_shape_learned
        _copy.feature_names_learned = self.feature_names_learned[:]
        return _copy

    def reset(self):
        super().reset()

    def fit(self, fitting_data, fitting_label, sample_weight=None):
        self.classifier.fit(fitting_data, fitting_label, sample_weight=sample_weight)

    def predict(self, prediction_data):
        return self.classifier.predict(prediction_data)

    def score(self, fitting_data, fitting_label):
        return self.classifier.score(fitting_data, fitting_label)

    def get_output_shape(self):
        return self.output_shape_learned

    def next_quality(self, observation, reward):
        streaming_environment = observation['streaming_environment']
        current_level = observation['current_level'][-1]
        observation_features_extracted = np.array(self.extract_features_observation(observation)).reshape(1, -1)
        observation_features_extracted = pd.DataFrame(observation_features_extracted,
                                                      columns=self.extract_features_names())
        try:
            self.likelihood_last_decision_val = self.classifier.predict_proba(
                observation_features_extracted)
            # Always predict probability to get a likelihood distribution
            next_quality = self.classifier.classes_[self.likelihood_last_decision_val.argmax()]
        except:
            print(observation)
            raise TypeError('Observation was erronous and contained wrong values')
        # print("Predicted next quality idx %d with classes %s" % (next_quality,str(self.classifier.classes_)))
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality], a_min=0,
                               a_max=streaming_environment.max_quality_level)
        if self.rate_correction:
            next_quality_rate = self.rate_correct_prediction(observation).argmax(1)[0]
            next_quality_rate = np.clip(current_level + self.quality_change_arr[next_quality_rate], a_min=0,
                                        a_max=streaming_environment.max_quality_level)
            if np.abs(next_quality - next_quality_rate) >= MIN_DIFFERENCE_QUALITY:
                next_quality = next_quality_rate
                print('Rate Corrected ')
        return next_quality

    def extract_features_names(self):
        return self.feature_names_learned

    def extract_features_observation(self, state_t):
        """
        Transforms observation into simple features contained
        :param state_t:
        :return:
        """
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


class ABRPolicyClassifierHandFeatureEngineering(ABRPolicyClassifierSimple):

    def __init__(self, abr_name, max_quality_change, deterministic, max_lookahead, max_lookback, classifier,
                 estimators=None, feature_complexity='complex', rate_correction=False):
        """
        Classifier which expects feature engineering in a first step. complexity is set by the appropriate feature
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param max_lookahead:
        :param max_lookback:
        :param classifier: What do we use as a classifer (Instance of Decisiontree classifier from sklearn)
        :param estimators: Future throughput estimators
        :param feature_complexity: Choose from ['normal', 'complex', 'very complex','very very complex']
        :param rate_correction: Deprecated
        """
        AVAIL_COMPLEXITY = ['normal', 'complex', 'very complex', 'very very complex']
        assert feature_complexity in AVAIL_COMPLEXITY, 'feature_complexity has to be in %s' % AVAIL_COMPLEXITY
        super().__init__(abr_name, max_quality_change, deterministic, max_lookahead, max_lookback, classifier,
                         rate_correction)
        self.feature_complexity = feature_complexity
        if estimators is None:
            self.throughput_estimators = [hmean, np.mean]
            if self.feature_complexity == 'complex':
                self.throughput_estimators += [generate_ewma(alpha) for alpha in np.linspace(0.15, 0.95, 5)]
                self.throughput_estimators += [generate_percentile(percentile) for percentile in
                                               np.linspace(0.15, 0.95, 5)]

    def copy(self):
        """
        Copy the instance
        :return:
        """
        tmp_file_name = self.rnd_id + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.classifier, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            classifier_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        _copy = ABRPolicyClassifierHandFeatureEngineering(self.abr_name, self.max_quality_change,
                                                          self.deterministic, self.max_lookahead, self.max_lookback,
                                                          classifier_copy, estimators=self.throughput_estimators[:],
                                                          feature_complexity=self.feature_complexity)
        _copy.output_shape_learned = self.output_shape_learned
        _copy.feature_names_learned = self.feature_names_learned[:]
        return _copy

    def get_output_shape(self):
        """
        How many features are we expecting
        :return:
        """
        n_derived_measurements = 0
        if self.feature_complexity == 'normal':
            n_derived_measurements = 2
        if self.feature_complexity in ['complex', 'very complex', 'very very complex']:
            n_derived_measurements = 4
        if self.feature_complexity in ['very very complex']:
            n_derived_measurements = 6
        else:
            ValueError('Wrong feature complexity')
        n_lookahead_measurement_types = self.max_quality_change * 2 + 1
        n_lookahead_measurement_types *= self.max_lookahead
        fixed_features = 1
        fixed_features += (self.max_lookback - 1)
        n_throughput_estimators = len(self.throughput_estimators) * (self.max_lookback - 1)
        if self.feature_complexity in ['very complex', 'very very complex']:
            fixed_features += 1
            n_statistics_planahead = 2
            ### Adding planahead features
            fixed_features += (self.max_quality_change * 2 + 1) * n_statistics_planahead * n_throughput_estimators
        return n_throughput_estimators * n_lookahead_measurement_types * n_derived_measurements + fixed_features

    def extract_features_names(self):
        """
        Feature names creation routine. Creates features names depending on the variable self.feature_complexity
        :return:
        """
        feature_names = ['buffer_size_s']
        if self.feature_complexity in ['very complex', 'very very complex']:
            feature_names += ['expected_segment_length_s']
        for lookback in range(1, self.max_lookback):
            feature_names += ['throughput_variance_-%d' % lookback]
        for tput_predictor_function in self.throughput_estimators:
            for lookback in range(1, self.max_lookback):
                switch_idx = np.arange(-self.max_quality_change, self.max_quality_change + 1)
                future_download_time_s_str = []
                for lookahead in range(self.max_lookahead):
                    future_download_time_s_str += [
                        ['future_download_time_switch_%d_lookahead_%d_lookback_%d_tput_predictor_%s' % (
                            switch, lookahead, lookback, tput_predictor_function.__name__) for switch in switch_idx]]
                future_download_time_s_str = np.array(future_download_time_s_str).T

                future_buffer_filling_str = []
                for lookahead in range(self.max_lookahead):
                    future_buffer_filling_str += [
                        ['future_buffer_filling_switch_%d_lookahead_%d_lookback_%d_tput_predictor_%s' % (
                            switch, lookahead, lookback, tput_predictor_function.__name__) for switch in switch_idx]]
                future_buffer_filling_str = np.array(future_buffer_filling_str).T

                if self.feature_complexity in ['complex', 'very complex', 'very very complex']:
                    future_byterate_ratio_str = []
                    for lookahead in range(self.max_lookahead):
                        future_byterate_ratio_str += [
                            ['future_mbit_discounted_switch_%d_lookahead_%d_lookback_%d_tput_predictor_%s' % (
                                switch, lookahead, lookback, tput_predictor_function.__name__) for switch in
                             switch_idx]]
                    future_byterate_ratio_str = np.array(future_byterate_ratio_str).T

                    future_vmaf_ratio_str = []
                    for lookahead in range(self.max_lookahead):
                        future_vmaf_ratio_str += [
                            ['future_vmaf_discounted_switch_%d_lookahead_%d_lookback_%d_tput_predictor_%s' % (
                                switch, lookahead, lookback, tput_predictor_function.__name__) for switch in
                             switch_idx]]
                    future_vmaf_ratio_str = np.array(future_vmaf_ratio_str).T
                    feature_names += list(future_byterate_ratio_str.flatten())
                    feature_names += list(future_vmaf_ratio_str.flatten())
                if self.feature_complexity in ['very complex', 'very very complex']:
                    linear_qoe_expected_normalized = []
                    for summary_function in ['max']:
                        linear_qoe_expected_normalized += [
                            ['linear_qoe_normalized_switch_%d_summary_function_%s_lookback_%d_tput_predictor_%s' % (
                                switch, summary_function, lookback, tput_predictor_function.__name__) for switch in
                             switch_idx]]
                    linear_qoe_expected_normalized = np.array(linear_qoe_expected_normalized).T
                    linear_qoe_expected = []
                    for summary_function in ['max']:
                        linear_qoe_expected += [
                            ['linear_qoe_raw_switch_%d_summary_function_%s_lookback_%d_tput_predictor_%s' % (
                                switch, summary_function, lookback, tput_predictor_function.__name__) for switch in
                             switch_idx]]
                    linear_qoe_expected = np.array(linear_qoe_expected).T

                    feature_names += list(linear_qoe_expected_normalized.flatten())
                    feature_names += list(linear_qoe_expected.flatten())
                if self.feature_complexity in [
                    'very very complex']:  ### Didn't show any improvement -- mostly overfitting
                    feature_names += [v + '+_normalized' for v in list(future_byterate_ratio_str.flatten())]
                    feature_names += [v + '+_normalized' for v in list(future_vmaf_ratio_str.flatten())]

                feature_names += list(future_download_time_s_str.flatten())
                feature_names += list(future_buffer_filling_str.flatten())
        assert len(feature_names) == self.get_output_shape(), 'Features are %d supposed to be %d %s' % (
            len(feature_names), self.get_output_shape(), self.feature_complexity)
        return feature_names

    def calculate_rebuffering(self, current_t, current_buffer, download_time, summary_function, segment_length_s):
        """
        How much rebuffering would we expect if we were to download at this specific quality
        :param current_t:
        :param current_buffer:
        :param download_time:
        :param summary_function:
        :param segment_length_s:
        :return:
        """
        max_shift_n, future_n = download_time.shape
        if current_t >= future_n:
            return None
        shift_rebuffering = []
        for current_quality in range(max_shift_n):
            new_buffer_size = current_buffer - download_time[current_quality, current_t]
            new_rebuffer = min(0, new_buffer_size)
            new_buffer_size = max(0, new_buffer_size) + segment_length_s
            rebuffer_return = self.calculate_rebuffering(current_t + 1, new_buffer_size, download_time,
                                                         summary_function, segment_length_s)
            if rebuffer_return is not None:
                new_rebuffer += rebuffer_return
            shift_rebuffering += [new_rebuffer]
        if current_t == 0:
            return shift_rebuffering
        else:
            return summary_function(shift_rebuffering)

    def calculate_linear_qoe(self,
                             current_t,
                             current_buffer,
                             encoded_mbit,
                             future_encoded_mbit,
                             future_download_time,
                             summary_function,
                             segment_length_s,
                             rebuffer_penality,
                             switching_penality):
        """
        Calculate linear QoE as in the MPC paper recursively
        :param current_t:
        :param current_buffer:
        :param encoded_mbit:
        :param future_encoded_mbit:
        :param future_download_time:
        :param summary_function:
        :param segment_length_s:
        :param rebuffer_penality:
        :param switching_penality:
        :return:
        """
        max_shift_n, future_n = future_download_time.shape
        if current_t >= future_n:
            return None
        shift_qoe_linear = []
        for current_quality in range(max_shift_n):
            new_buffer_size = current_buffer - future_download_time[current_quality, current_t]
            new_rebuffer = min(0, new_buffer_size)
            qoe_linear = future_encoded_mbit[current_quality, current_t]
            qoe_linear += rebuffer_penality * new_rebuffer
            qoe_linear -= switching_penality * np.abs(future_encoded_mbit[current_quality, current_t] - encoded_mbit)
            new_buffer_size = max(0, new_buffer_size) + segment_length_s
            qoe_linear_return = self.calculate_linear_qoe(current_t=current_t + 1,
                                                          current_buffer=new_buffer_size,
                                                          encoded_mbit=future_encoded_mbit[current_quality, current_t],
                                                          future_encoded_mbit=future_encoded_mbit,
                                                          future_download_time=future_download_time,
                                                          summary_function=summary_function,
                                                          segment_length_s=segment_length_s,
                                                          rebuffer_penality=rebuffer_penality,
                                                          switching_penality=switching_penality)
            if qoe_linear_return is not None:
                qoe_linear += qoe_linear_return
            shift_qoe_linear += [qoe_linear]
        if current_t == 0:
            return shift_qoe_linear
        else:
            return summary_function(shift_qoe_linear)

    def extract_features_observation(self, state_t):
        """
        Feature value creation routine. Creates feature values depending on the variable self.feature_complexity
        :return:
        """
        valid_indices = len(state_t['video_chunk_size_byte']) - sum([m == 0 for m in state_t[
            'video_chunk_size_byte']])
        video_chunk_size_byte = np.array(state_t['video_chunk_size_byte'][-valid_indices:])
        future_video_chunk_size_byte = []
        for k, v in state_t.items():
            if 'future' in k and 'byte' in k:
                future_video_chunk_size_byte.append(v)
        future_video_chunk_size_byte = np.array(future_video_chunk_size_byte)
        future_video_chunk_byterate = []
        for k, v in state_t.items():
            if 'future' in k and 'bitrate' in k:
                future_video_chunk_byterate.append(v)
        future_video_chunk_byterate = np.array(future_video_chunk_byterate) / 8.

        future_video_chunk_vmaf = []
        for k, v in state_t.items():
            if 'future' in k and 'vmaf' in k:
                future_video_chunk_vmaf.append(v)
        future_video_chunk_vmaf = np.array(future_video_chunk_vmaf) / 100.  # Vmaf goes from 0 to 1
        download_time_s = np.array(state_t['download_time_s'][-valid_indices:])
        buffer_size = state_t['buffer_size_s'][-1]
        segment_length_s = np.mean(state_t['segment_length_s'])
        encoded_mbit = state_t['encoded_mbitrate'][-1]
        feature_arr = [buffer_size]
        if self.feature_complexity in ['very complex', 'very very complex']:
            expected_segment_length = [segment_length_s]
            feature_arr += expected_segment_length
        for lookback in range(1, self.max_lookback):
            throughput_value = video_chunk_size_byte[-lookback:] / download_time_s[-lookback:]
            throughput_value = throughput_value[~np.isnan(throughput_value)]
            throughput_value = throughput_value[~np.isinf(throughput_value)]
            if len(throughput_value) == 0:
                return list(np.zeros(self.get_output_shape()))
            feature_arr += [np.std(throughput_value)]
        for tput_predictor_function in self.throughput_estimators:
            for i in range(1, self.max_lookback):
                if sum(np.isnan(video_chunk_size_byte[-i:])) > 0:
                    raise ValueError('video_chunk_size_byte is erronous %s' % video_chunk_size_byte[-i:])
                if sum(np.isnan(download_time_s[-i:])) > 0:
                    raise ValueError('video_chunk_size_byte is erronous %s' % download_time_s[-i:])
                if sum(np.isinf(video_chunk_size_byte[-i:])) > 0:
                    raise ValueError('video_chunk_size_byte is erronous %s' % video_chunk_size_byte[-i:])
                if sum(np.isinf(download_time_s[-i:])) > 0:
                    raise ValueError('video_chunk_size_byte is erronous %s' % download_time_s[-i:])
                throughput_value = video_chunk_size_byte[-i:] / download_time_s[-i:]
                throughput_value = throughput_value[~np.isnan(throughput_value)]
                throughput_value = throughput_value[~np.isinf(throughput_value)]
                if len(throughput_value) == 0:
                    return list(np.zeros(self.get_output_shape()))
                throughput_value = tput_predictor_function(throughput_value)
                future_download_time_s = future_video_chunk_size_byte / throughput_value

                future_buffer_filling_ratio = np.zeros(future_video_chunk_byterate.shape)
                future_buffer_filling_ratio[future_video_chunk_byterate != 0] = future_video_chunk_byterate[
                                                                                    future_video_chunk_byterate != 0] / throughput_value
                if self.feature_complexity in ['complex', 'very complex', 'very very complex']:
                    # vmaf_filling_ratio = np.zeros(future_video_chunk_vmaf.shape)
                    # vmaf_filling_ratio[future_video_chunk_vmaf != 0] = future_buffer_filling_ratio[
                    #                                                       future_video_chunk_vmaf != 0] / \
                    #                                                   future_video_chunk_vmaf[
                    #                                                       future_video_chunk_vmaf != 0]

                    # bitrate_filling_ratio = np.zeros(future_video_chunk_byterate.shape)
                    # bitrate_filling_ratio[future_video_chunk_byterate != 0] = future_buffer_filling_ratio[
                    #                                                              future_video_chunk_byterate != 0] / \
                    #                                                           future_video_chunk_byterate[
                    #                                                               future_video_chunk_byterate != 0]

                    ### How many seconds does it take to download one second of material
                    # if self.feature_complexity == 'complex':
                    #### What buffer situation does one point of vmaf entail
                    ### The larger the value the less bad the encountered drainage is
                    vmaf_filling_ratio = future_buffer_filling_ratio * future_video_chunk_vmaf  ## [0 ... 100] ,[0 ... 1]
                    bitrate_filling_ratio = future_buffer_filling_ratio * (
                            future_video_chunk_byterate * 1e-6)  ## [0 ... 100] ,[0 ... 1]

                    # rebuffering_projected = []
                    # rebuffering_vs_bitrate = []
                    # for summary_function in [np.mean, np.max, np.min]:
                    #    rebuffering_estimated = list(self.calculate_rebuffering(
                    #       current_t=0, current_buffer=buffer_size,
                    ##       future_download_time=future_download_time_s,
                    #      summary_function=summary_function,
                    #      segment_length_s=segment_length_s))
                    # rebuffering_projected += rebuffering_estimated
                    # rebuffering_estimated = np.array(rebuffering_estimated)
                    # rebuffering_vs_bitrate_instance = (future_video_chunk_byterate[:,0] * 1e-6) * (1./(1. - rebuffering_estimated))
                    # rebuffering_vs_bitrate += list(rebuffering_vs_bitrate_instance)

                    # rebuffering_projected = np.array(rebuffering_projected)
                    # rebuffering_vs_bitrate = np.array(rebuffering_vs_bitrate)
                    feature_arr += list(bitrate_filling_ratio.flatten())
                    feature_arr += list(vmaf_filling_ratio.flatten())
                    # feature_arr += list(rebuffering_projected.flatten())
                    # feature_arr += list(rebuffering_vs_bitrate.flatten())
                if self.feature_complexity in ['very complex', 'very very complex']:
                    summary_function = np.max
                    rebuffer_penality = 4.3
                    switching_penality = 1.
                    linear_qoe_expected = self.calculate_linear_qoe(
                        current_t=0,
                        current_buffer=buffer_size,
                        encoded_mbit=encoded_mbit,
                        future_encoded_mbit=future_video_chunk_byterate * 1e-6,
                        future_download_time=future_download_time_s,
                        summary_function=summary_function,
                        segment_length_s=segment_length_s,
                        rebuffer_penality=rebuffer_penality,
                        switching_penality=switching_penality)
                    linear_qoe_expected = np.array(linear_qoe_expected)
                    linear_qoe_expected_normalized = linear_qoe_expected - linear_qoe_expected.min()
                    if np.sum(linear_qoe_expected_normalized) > 0.:
                        linear_qoe_expected_normalized = linear_qoe_expected_normalized / linear_qoe_expected_normalized.sum()
                    feature_arr += list(linear_qoe_expected_normalized)
                    feature_arr += list(linear_qoe_expected)
                if self.feature_complexity in ['very very complex']:
                    feature_arr += list(
                        (vmaf_filling_ratio / (vmaf_filling_ratio.sum(0, keepdims=True) + 1e-10)).flatten())
                    feature_arr += list(
                        (bitrate_filling_ratio / (vmaf_filling_ratio.sum(0, keepdims=True) + 1e-10)).flatten())

                feature_arr += list(future_download_time_s.flatten())
                feature_arr += list(future_buffer_filling_ratio.flatten())
        assert len(feature_arr) == self.get_output_shape(), 'Features are %d supposed to be %d %s' % (len(feature_arr),
                                                                                                      self.get_output_shape(),
                                                                                                      self.feature_complexity)
        return feature_arr


class ABRPolicyClassifierAutomatedFeatureEngineering(ABRPolicyClassifierSimple):

    def __init__(self, abr_name, max_quality_change, deterministic, max_lookahead, max_lookback, classifier,
                 function_operators=None, time_budget_s=100, cores_avail=1, rate_correction=False):
        """
        Create features automatically with the help of GeneticFeatureGenerator
        :param abr_name:
        :param max_quality_change: quality level changes
        :param deterministic: Depreceated
        :param max_lookahead: How far do we plan ahead
        :param max_lookback: How many measurements in the past do we consider
        :param classifier: What is the final classifier we learn the features for
        :param function_operators: Which functions can we use
        :param time_budget_s: How much time does the feature learning function has
        :param cores_avail:
        :param rate_correction:
        """
        super().__init__(abr_name, max_quality_change, deterministic, max_lookahead, max_lookback, classifier,
                         rate_correction)
        if function_operators is None:
            self.function_operators = projection_generator_function(max_arity=5, projection="np.mean")
            self.function_operators += projection_generator_function(max_arity=5, projection="np.sum")
            self.function_operators += ['div', 'add', 'sub', 'mul']
        self.cores_avail = cores_avail
        self.time_budget_s = time_budget_s
        self.automated_transformer = GeneticFeatureGenerator(tree_estimator=XGBClassifier(),
                                                             function_set=self.function_operators,
                                                             max_formula_length=6,
                                                             hall_of_fame=None, n_jobs=cores_avail,
                                                             generations=300, time_budget_s=time_budget_s)

    def copy(self):
        tmp_file_name = self.rnd_id + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.classifier, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            classifier_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        tmp_file_name = self.rnd_id + 'tmp_id'
        with open(tmp_file_name, 'wb') as dill_temp:
            dill.dump(self.automated_transformer, dill_temp)
        with open(tmp_file_name, 'rb') as dill_temp:
            automated_transformer_copy = dill.load(dill_temp)
        os.remove(tmp_file_name)
        _copy = ABRPolicyClassifierAutomatedFeatureEngineering(self.abr_name, self.max_quality_change,
                                                               self.deterministic, self.max_lookahead,
                                                               self.max_lookback,
                                                               classifier_copy,
                                                               function_operators=self.function_operators[:],
                                                               time_budget_s=self.time_budget_s,
                                                               cores_avail=self.cores_avail)
        _copy.automated_transformer = automated_transformer_copy
        _copy.output_shape_learned = self.output_shape_learned
        _copy.feature_names_learned = self.feature_names_learned[:]
        return _copy

    def predict(self, prediction_data):
        return self.classifier.predict(self.automated_transformer.transform(prediction_data))

    def fit(self, fitting_data, fitting_label, sample_weight=None):
        self.automated_transformer.feature_names = self.feature_names_learned
        self.automated_transformer.fit(fitting_data, fitting_label, sample_weight=sample_weight)
        self.classifier.fit(self.automated_transformer.transform(fitting_data), fitting_label,
                            sample_weight=sample_weight)

    def score(self, fitting_data, fitting_label):
        return self.classifier.score(self.automated_transformer.transform(fitting_data),
                                     fitting_label)

    def next_quality(self, observation, reward):
        streaming_environment = observation['streaming_environment']
        current_level = observation['current_level'][-1]
        observation_features_extracted = np.array(self.extract_features_observation(observation)).reshape(1, -1)
        observation_features_extracted = pd.DataFrame(observation_features_extracted,
                                                      columns=self.extract_features_names())
        observation_features_extracted = self.automated_transformer.transform(observation_features_extracted)
        self.likelihood_last_decision_val = self.classifier.predict_proba(
            observation_features_extracted)  # Always predict probability to get a likelihood distribution

        next_quality = self.classifier.classes_[self.likelihood_last_decision_val.argmax()]
        # print("Predicted next quality idx % d" % next_quality)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality], a_min=0,
                               a_max=streaming_environment.max_quality_level)
        # print('Next quality is therefore %d' % next_quality)
        # print('--' * 10)
        if self.rate_correction:
            next_quality_rate = self.rate_correct_prediction(observation).argmax(1)[0]
            next_quality_rate = np.clip(current_level + self.quality_change_arr[next_quality_rate], a_min=0,
                                        a_max=streaming_environment.max_quality_level)
            if np.abs(next_quality - next_quality_rate) >= MIN_DIFFERENCE_QUALITY:
                next_quality = next_quality_rate
        return next_quality

    def get_output_shape(self):
        n_lookahead_measurement_types = self.max_quality_change * 2 + 1
        n_lookahead_measurement_types *= self.max_lookahead
        n_derived_measurements = 5
        n_throughput_estimators = len(self.function_operators) * (self.max_lookback - 1)
        return n_throughput_estimators * n_lookahead_measurement_types * n_derived_measurements + 1
