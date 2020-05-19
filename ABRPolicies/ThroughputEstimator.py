from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

"""
Contains throughput estimator base function and simple predictors
"""


def generate_ewma(alpha):
    # Common Sense
    func = lambda value_arr: pd.Series(value_arr).ewm(alpha=alpha, adjust=False).mean().values[-1]
    func.__name__ = 'exponential_weighted_moving_average_alpha_%.2f' % alpha
    return func


def generate_double_ewma(alpha, alpha_offset):
    # Taken from the Shaka Player
    if alpha + alpha_offset > 1:
        alpha_offset = 1.0 - alpha
    func = lambda value_arr: np.min([pd.Series(value_arr).ewm(alpha=alpha, adjust=False).mean().values[-1],
                                     pd.Series(value_arr).ewm(alpha=alpha + alpha_offset, adjust=False).mean().values[
                                         -1]])
    func.__name__ = 'exponential_weighted_moving_average_alpha_%.2f_alpha_offset_%.2f' % (alpha, alpha_offset)
    return func


def generate_weighted_moving_average():
    func = lambda value_arr: np.average(value_arr, weights=np.cumsum(np.arange(1, len(value_arr))))
    func.__name__ = 'weighted_moving_average'
    return func


def generate_percentile(percentile):
    # Why not use a percentile thingy
    func = lambda value_arr: np.nanpercentile(value_arr, q=percentile)
    func.__name__ = 'percentile_q_%.2f' % percentile
    return func


class ThroughputEstimator(ABC):
    def __init__(self, predictor_function, robust_estimate):
        self.predictor_function = predictor_function
        self.tput_history = []
        self.past_errors = []
        self.robust_estimate = robust_estimate

    def reset(self):
        self.tput_history = []
        self.past_errors = []

    def add_sample(self, tput_estimate, timestamp_s):
        if len(self.tput_history) > 0:
            my_estimate = self.__obtain_robust_estimate()
            curr_error = abs(my_estimate - tput_estimate) / float(tput_estimate)
            self.past_errors.append((timestamp_s, curr_error))
        self.tput_history.append((timestamp_s, tput_estimate))

    def predict_future_bandwidth(self):
        if self.robust_estimate:
            return self.__obtain_robust_estimate()
        else:
            return self.__obtain_estimate()

    def __obtain_estimate(self):
        assert len(self.tput_history) > 0, 'No recorded values no estimate'
        return self.predictor_function(self.select_valid(self.tput_history))

    def __obtain_robust_estimate(self):
        if len(self.past_errors) == 0:
            return self.__obtain_estimate()
        max_error = np.max(self.select_valid(self.past_errors))
        future_bandwidth = self.__obtain_estimate() / (1 + max_error)
        return future_bandwidth

    @abstractmethod
    def select_valid(self, value_arr):
        pass

    @abstractmethod
    def copy(self):
        pass


class GlobalEstimator(ThroughputEstimator):

    def copy(self):
        return GlobalEstimator(self.predictor_function,
                               self.robust_estimate)

    def select_valid(self, value_arr):
        value_arr = []
        for time, value in value_arr:
            value_arr.append(value)
        return value_arr


class StepEstimator(ThroughputEstimator):

    def __init__(self, consider_last_n_steps, predictor_function, robust_estimate):
        """
        Consider the last n steps of measurements to predict the future bandwidth with the function
        in predictor_function. Robust estimate corrects estimator as in the MPC implementation in
        https://github.com/hongzimao/pensieve
        :param consider_last_n_steps:
        :param predictor_function:
        :param robust_estimate:
        """
        super().__init__(predictor_function, robust_estimate)
        self.consider_last_n_steps = consider_last_n_steps

    def copy(self):
        return StepEstimator(self.consider_last_n_steps, self.predictor_function,
                             self.robust_estimate)

    def select_valid(self, value_arr):
        considered_steps = min([self.consider_last_n_steps, len(value_arr)])
        selected_value_arr = []
        for time, value in value_arr[-considered_steps:]:
            selected_value_arr.append(value)
        return selected_value_arr


class TimeEstimator(ThroughputEstimator):
    def __init__(self, consider_last_t_seconds, predictor_function, robust_estimate):
        """
        Consider the last n steps of measurements which happend less than consider_last_t_seconds seconds ago
        .Predict the future bandwidth with the function in predictor_function.
        Robust estimate corrects estimator as in the MPC implementation in https://github.com/hongzimao/pensieve
        :param consider_last_t_seconds:
        :param predictor_function:
        :param robust_estimate:
        """
        super().__init__(predictor_function, robust_estimate)
        self.consider_last_t_seconds = consider_last_t_seconds

    def copy(self):
        return TimeEstimator(self.consider_last_t_seconds, self.predictor_function,
                             self.robust_estimate)

    def select_valid(self, value_arr):
        _, most_recent = self.tput_history[-1]
        selected_value_arr = []
        for time, tput in value_arr:
            if (most_recent - time) < self.consider_last_t_seconds:
                selected_value_arr.append(tput)
        return selected_value_arr
