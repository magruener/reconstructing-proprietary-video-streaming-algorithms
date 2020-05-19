from abc import abstractmethod

import numpy as np
from pulp import LpVariable

ALLOWED_POLICIES = ['Approximation', 'Expert']


class ConvergenceException(Exception):
    """Raised when Optimization problem is not solvable """
    pass


class RewardFunction:

    @abstractmethod
    def return_reward_observation(self, enviroment_state):
        pass

    def return_objective_dataframe(self, enviroment_state):
        return self.return_reward_observation(enviroment_state)

    @abstractmethod
    def extract_coefficients_dictionary(self, evaluation_dictionary):
        pass

    @abstractmethod
    def extract_coefficients_dataframe(self, evaluation_dataframe):
        pass

    @abstractmethod
    def extract_lp_objective(self, evaluation_dataframe, lp_variables):
        pass

    @abstractmethod
    def extract_lp_variables(self):
        pass

    @abstractmethod
    def set_from_lp_variables(self, lp_variables):
        pass

    @abstractmethod
    def lp_variables_same(self, lp_variables):
        pass

    @abstractmethod
    def coefficient_names(self):
        pass

"""
class FunctionReward(RewardFunction):
    def __init__(self,observation_transformer):
        self.trajectory_dummy  = Trajectory()
        self.observation_transformer = observation_transformer

    def return_reward_observation(self, observation):
        last_level = observation['current_level'][-2]
        current_level = observation['current_level'][-1]
        streaming_enviroment = observation['streaming_environment']
        switch_mapper = list(np.arange(-streaming_enviroment.max_switch_allowed,streaming_enviroment.max_switch_allowed + 1))
        assert np.abs(current_level - last_level) <= streaming_enviroment.max_switch_allowed,observation['current_level']


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
        current_level_switch = to_categorical([switch_mapper.index(current_level - last_level)],num_classes=streaming_enviroment.max_switch_allowed * 2 + 1)
        action_prob = self.observation_transformer.predict([state_t, state_t_future,current_level_switch])
        action_prob = action_prob.flatten()
        #log(D(θ,φ(s,a,s′)))−log(1−D(θ,φ(s,a,s′)))
        return np.log(action_prob[1]) - np.log(1. - action_prob[1]) #We want to maximize the likelihood that it is part of the first class

    def extract_coefficients_dictionary(self, evaluation_dictionary):
        pass

    def extract_coefficients_dataframe(self, evaluation_dataframe):
        pass

    def extract_lp_objective(self, evaluation_dataframe, lp_variables):
        pass

    def extract_lp_variables(self):
        pass

    def set_from_lp_variables(self, lp_variables):
        pass

    def lp_variables_same(self, lp_variables):
        pass

    def coefficient_names(self):
        pass


class BufferFillingReward(RewardFunction):

    def coefficient_names(self):
        return [self.perceptual_key, 'download_time_deviation_penality']

    def lp_variables_same(self, lp_variables):
        lp_variables_same = self.download_time_deviation_penality == lp_variables['buffer_drainage_penality']
        lp_variables_same = lp_variables_same and (self.perceptual_weight == lp_variables[self.perceptual_key + '_weight'])
        return lp_variables_same

    def extract_lp_objective(self, evaluation_dataframe, lp_variables):
        coefficients = self.extract_coefficients_dataframe(evaluation_dataframe)
        perceptual_weight = lp_variables[self.perceptual_key + '_weight']
        buffer_drainage_penality = lp_variables['buffer_drainage_penality']
        return sum([perceptual_weight * perceptual_gain + buffer_drainage_penality * buffer_drainage for
                    perceptual_gain, buffer_drainage in coefficients.to_numpy()])

    def extract_lp_variables(self):
        lp_variables = {
            self.perceptual_key + '_weight': LpVariable(self.perceptual_key + '_weight', .1, 15),
            'buffer_drainage_penality': LpVariable("buffer_drainage_penality", .1, 15)
        }
        return lp_variables

    def set_from_lp_variables(self, lp_variables):
        self.download_time_deviation_penality = lp_variables['buffer_drainage_penality']
        self.perceptual_weight = lp_variables['perceptual_weight']

    def __init__(self, perceptual_weight=1.0, download_time_deviation_penality=1, perceptual_key='encoded_mbitrate'):
        self.download_time_deviation_penality = download_time_deviation_penality
        self.perceptual_weight = perceptual_weight
        self.perceptual_key = perceptual_key

    def return_reward_observation(self, enviroment_state):
        drainange = enviroment_state['download_time_s'][-1] - enviroment_state['segment_length_s'][-1]
        reward = self.perceptual_weight * enviroment_state[
            self.perceptual_key] - self.download_time_deviation_penality * drainange
        reward /= enviroment_state['segment_length_s'][-1]
        return reward

    def extract_coefficients_dictionary(self, evaluation_dictionary):
        reward_coefficient_list = []
        reward_coefficient_list += [evaluation_dictionary[self.perceptual_key]]
        reward_coefficient_list += [
            -(evaluation_dictionary['download_time_s'] - evaluation_dictionary['segment_length_s'])]
        reward_coefficient_list = np.array(reward_coefficient_list) / evaluation_dictionary['segment_length_s']
        return reward_coefficient_list

    def extract_coefficients_dataframe(self, evaluation_dataframe):
        reward_matrix_dataframe = evaluation_dataframe[[self.perceptual_key]].copy()
        reward_matrix_dataframe['buffer_drainage'] = -(evaluation_dataframe['download_time_s'] - evaluation_dataframe['segment_length_s'])
        reward_matrix_dataframe /= evaluation_dataframe[['segment_length_s']].values
        return reward_matrix_dataframe.dropna()
"""

class ClassicPerceptualReward(RewardFunction):
    """
    Simple perceptual reward metric
    """

    def lp_variables_same(self, lp_variables):
        lp_variables_same = (self.perceptual_weight == lp_variables[self.perceptual_key + '_weight'])
        lp_variables_same = lp_variables_same and (self.smoothing_penality == lp_variables['smoothing_penality'])
        lp_variables_same = lp_variables_same and (self.rebuffer_penalty == lp_variables['rebuffer_penalty'])
        return lp_variables_same

    def extract_lp_objective(self, evaluation_dataframe, lp_variables):
        coefficients = self.extract_coefficients_dataframe(evaluation_dataframe)
        perceptual_weight = lp_variables[self.perceptual_key + '_weight']
        rebuffer_penalty = lp_variables['rebuffer_penalty']
        smoothing_penality = lp_variables['smoothing_penality']
        return sum(
            [perceptual_weight * perceptual_gain + rebuffer_penalty * (
                    rebuffering + 1e-10) + smoothing_penality * smoothing for
             perceptual_gain, rebuffering, smoothing in coefficients.to_numpy()])

    def extract_lp_variables(self):
        lp_variables = {
            self.perceptual_key + '_weight': LpVariable(self.perceptual_key + '_weight', .1, 15),
            'rebuffer_penalty': LpVariable("rebuffer_penalty", .1, 15),
            'smoothing_penality': LpVariable("smoothing_penality", .1, 15)
        }
        return lp_variables

    def set_from_lp_variables(self, lp_variables):
        self.perceptual_weight = lp_variables[self.perceptual_key + '_weight']
        self.smoothing_penality = lp_variables['smoothing_penality']
        self.rebuffer_penalty = lp_variables['rebuffer_penalty']

    def __init__(self, perceptual_weight=1.0, rebuffer_penalty=4.3, smoothing_penality=1,
                 perceptual_key='single_mbitrate'):
        self.perceptual_weight = perceptual_weight
        self.smoothing_penality = smoothing_penality
        self.rebuffer_penalty = rebuffer_penalty
        self.perceptual_key = perceptual_key

    def return_reward_observation(self, enviroment_descriptor):
        reward = self.perceptual_weight * enviroment_descriptor[self.perceptual_key][-1] - self.rebuffer_penalty * \
                 enviroment_descriptor[
                     'rebuffer_time_s'][-1] - self.smoothing_penality * np.abs(
            enviroment_descriptor[self.perceptual_key][-1] - enviroment_descriptor[self.perceptual_key][-2])
        reward /= enviroment_descriptor['segment_length_s'][-1]
        return reward

    def return_reward_dataframe(self, enviroment_descriptor):
        reward = self.perceptual_weight * enviroment_descriptor[self.perceptual_key] - self.rebuffer_penalty * \
                 enviroment_descriptor[
                     'rebuffering_seconds'] - self.smoothing_penality * np.abs(
            enviroment_descriptor[self.perceptual_key] - enviroment_descriptor[self.perceptual_key].shift(1))
        reward /= enviroment_descriptor['segment_length_s']
        return reward

    def extract_coefficients_dictionary(self, evaluation_dictionary):
        reward_coefficient_list = []
        reward_coefficient_list += [evaluation_dictionary[self.perceptual_key]]
        reward_coefficient_list += [-evaluation_dictionary['rebuffer_time_s']]
        reward_coefficient_list += [
            -np.abs(evaluation_dictionary[self.perceptual_key] - evaluation_dictionary[self.perceptual_key][-1])]
        reward_coefficient_list = np.array(reward_coefficient_list) / evaluation_dictionary['segment_length_s']
        return reward_coefficient_list

    def extract_coefficients_dataframe(self, evaluation_dataframe):
        reward_matrix_dataframe = evaluation_dataframe[[self.perceptual_key]].copy()
        reward_matrix_dataframe['rebuffering_seconds'] = -evaluation_dataframe['rebuffering_seconds'].values
        reward_matrix_dataframe[self.perceptual_key + '_shift'] = -np.abs(
            evaluation_dataframe[self.perceptual_key].shift(1) - evaluation_dataframe[self.perceptual_key])
        reward_matrix_dataframe /= evaluation_dataframe[['segment_length_s']].values
        return reward_matrix_dataframe.dropna()

    def coefficient_names(self):
        return [self.perceptual_key, 'rebuffering_seconds', self.perceptual_key + '_shift']

"""
class RewardLearner(RewardFunction, ABC):

    def __init__(self, linear_reward_function: RewardFunction):
        self.linear_reward_function = linear_reward_function
        self.reward_function_coefficients_expert = []
        self.reward_function_coefficients_approximation = []
        self.contains_policies = set()

    def generate_reward_function_coefficients(self, shuffle_data=True):
        reward_function_coefficients = pd.concat(
            [self.reward_function_coefficients_expert, self.reward_function_coefficients_approximation])
        reward_generating_policies = ['Expert'] * len(self.reward_function_coefficients_expert) + [
            'Approximation'] * len(self.reward_function_coefficients_approximation)
        if shuffle_data:
            reward_function_coefficients, reward_generating_policies = shuffle(reward_function_coefficients,
                                                                               reward_generating_policies)
        return reward_function_coefficients, reward_generating_policies

    def extract_coefficients_dictionary(self, evaluation_dictionary):
        return self.linear_reward_function.extract_coefficients_dictionary(evaluation_dictionary)

    def extract_coefficients_dataframe(self, evaluation_dataframe):
        return self.linear_reward_function.extract_coefficients_dataframe(evaluation_dataframe)

    def extract_lp_objective(self, evaluation_dataframe, lp_variables):
        return self.linear_reward_function.extract_lp_objective(evaluation_dataframe, lp_variables)

    def extract_lp_variables(self):
        return self.linear_reward_function.extract_lp_variables()

    def set_from_lp_variables(self, lp_variables):
        self.linear_reward_function.set_from_lp_variables(lp_variables)

    def lp_variables_same(self, lp_variables):
        return self.linear_reward_function.lp_variables_same(lp_variables)

    def coefficient_names(self):
        return self.linear_reward_function.coefficient_names()

    def return_objective(self, evaluation):
        return self.return_reward_observation(evaluation)

    @abstractmethod
    def add_data(self, evaluation_dataframe, generating_policy, replace_data=False):
        pass


"""