import logging
import os

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import class_weight
from tensorflow.keras import backend as K, Input, Model
from tensorflow.keras.layers import GRU, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy.stats import entropy

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dill

from ABRPolicies.ABRPolicy import ABRPolicy
from BehaviourCloning.ActionCloning import KerasPolicy
from SimulationEnviroment.SimulatorEnviroment import Trajectory, TrajectoryVideoStreaming

LOGGING_LEVEL = logging.DEBUG
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
N_BINS_DIST = 15


def proximal_policy_optimization_loss(advantage, old_prediction, loss_clipping=0.2, entropy_loss=5e-3):
    """
    https://github.com/LuEE-C/PPO-Keras/blob/master/Main.py
    # Only implemented clipping for the surrogate loss, paper said it was best
    :param advantage:
    :param old_prediction:
    :param loss_clipping:
    :param entropy_loss:
    :return:
    """

    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)  # Multiply with the one hot encoded taken action
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(
            r, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage) + entropy_loss * -(
                prob * K.log(prob + 1e-10)))

    return loss


class KerasDiscriminator:

    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 action_dimension,
                 drop_prob=0.2):
        """
        General purpose keras GRU classifer
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param hidden_dim:
        :param action_dimension:
        :param drop_prob:
        """
        discriminator_past_input = Input(shape=(None, past_measurement_dimensions))
        discriminator_future_input = Input(shape=(None, future_measurements_dimensions))
        discriminator_action_input = Input(shape=(action_dimension,))

        self.discriminator_past_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)
        self.discriminator_future_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)
        self.discriminator_dense_1 = Dense(units=hidden_dim, activation="relu")
        self.discriminator_dense_2 = Dense(units=hidden_dim, activation="relu")
        self.discriminator_dense_final = Dense(activation="softmax", units=2)  # We predict two -> Is it real or not
        concatenated = concatenate(
            [self.discriminator_past_GRU(discriminator_past_input),
             self.discriminator_future_GRU(discriminator_future_input)])
        concatenated = self.discriminator_dense_1(concatenated)
        concatenated = concatenate([concatenated, discriminator_action_input])
        concatenated = self.discriminator_dense_2(concatenated)
        discriminator_likelihood = self.discriminator_dense_final(concatenated)
        self.model = Model(inputs=[discriminator_past_input, discriminator_future_input,
                                   discriminator_action_input], outputs=discriminator_likelihood)
        self.model.compile(loss="categorical_crossentropy", optimizer='adam')


class KerasValue:

    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 drop_prob=0.2):
        """
        V(S) function
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param hidden_dim:
        :param drop_prob:
        """
        discriminator_past_input = Input(shape=(None, past_measurement_dimensions))
        discriminator_future_input = Input(shape=(None, future_measurements_dimensions))

        self.discriminator_past_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)
        self.discriminator_future_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)
        self.discriminator_dense_1 = Dense(units=hidden_dim // 2, activation="relu")
        self.discriminator_dense_2 = Dense(units=hidden_dim // 2, activation="relu")
        self.discriminator_dense_3 = Dense(units=hidden_dim // 4, activation="relu")
        self.discriminator_dense_final = Dense(units=1)
        concatenated = concatenate(
            [self.discriminator_past_GRU(discriminator_past_input),
             self.discriminator_future_GRU(discriminator_future_input)])
        concatenated = self.discriminator_dense_1(concatenated)
        concatenated = self.discriminator_dense_2(concatenated)
        concatenated = self.discriminator_dense_3(concatenated)
        linear_output_layer = self.discriminator_dense_final(concatenated)
        self.model = Model(inputs=[discriminator_past_input, discriminator_future_input],
                           outputs=linear_output_layer)
        self.model.compile(loss="mse", optimizer='adam')


class KerasGAIL:
    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 action_dimension,
                 drop_prob=0.2):
        """
        Keras model for GAIL approach
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param hidden_dim:
        :param action_dimension:
        :param drop_prob:
        """
        self.policy_model = KerasPolicy(past_measurement_dimensions,
                                        future_measurements_dimensions,
                                        hidden_dim,
                                        action_dimension,
                                        drop_prob=drop_prob)
        advantage_input = Input(shape=(1,))
        likelihood_action_distributed_input = Input(shape=(action_dimension,))
        keras_inputs = []
        keras_inputs += [self.policy_model.policy_past_input, self.policy_model.policy_future_input]
        keras_inputs += [advantage_input, likelihood_action_distributed_input]
        self.gail_training_model = Model(inputs=keras_inputs, outputs=self.policy_model.policy_action_output)
        self.gail_training_model.compile(loss=proximal_policy_optimization_loss(advantage=advantage_input,
                                                                                old_prediction=likelihood_action_distributed_input),
                                         optimizer=Adam(lr=1e-4))


class KerasEmbedder:

    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 embedding_dimension,
                 drop_prob=0.2):
        self.policy_past_input = Input(shape=(None, past_measurement_dimensions))
        self.policy_past_GRU = GRU(units=hidden_dim,
                                   return_sequences=False, dropout=drop_prob)(self.policy_past_input)
        self.policy_future_input = Input(shape=(None, future_measurements_dimensions))
        self.policy_future_GRU = GRU(units=hidden_dim, return_sequences=False, dropout=drop_prob)(
            self.policy_future_input)
        self.policy_dense1 = Dense(units=hidden_dim, activation='relu')
        self.policy_dense2 = Dense(units=hidden_dim // 2, activation='relu')
        self.policy_dense3 = Dense(units=embedding_dimension)
        concatenated = concatenate([self.policy_past_GRU, self.policy_future_GRU])
        concatenated = self.policy_dense1(concatenated)
        concatenated = self.policy_dense2(concatenated)
        self.policy_action_output = self.policy_dense3(concatenated)
        self.model = Model(inputs=[self.policy_past_input, self.policy_future_input], outputs=self.policy_action_output)
        self.model.compile(loss="mse", optimizer='adam')


class RandomExpertDistillation(ABRPolicy):
    """
    Implementation from https://arxiv.org/pdf/1905.06750.pdf RED which was a poster at ICML2019

    Current work in Off-policy learning is not advantageous to us because we do not have the reward function
    """

    def reset(self):
        super().reset()

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
        action_prob = self.policy_network.policy_model.model.predict([state_t, state_t_future])
        self.likelihood_last_decision_val = action_prob
        if self.deterministic:
            next_quality_switch_idx = np.argmax(action_prob)
        else:
            probability = action_prob.flatten()
            next_quality_switch_idx = np.random.choice(np.arange(len(probability)), size=1, p=probability)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality_switch_idx], a_min=0,
                               a_max=streaming_enviroment.max_quality_level)
        return next_quality

    def copy(self):
        copy_ = RandomExpertDistillation(self.abr_name, self.max_quality_change, self.deterministic,
                                         self.past_measurement_dimensions,
                                         self.future_measurements_dimensions, self.cloning_epochs, self.drop_prob,
                                         self.hidden_dim,
                                         self.batch_size_cloning,
                                         self.validation_split, self.scaling_factor_sigma, self.future_reward_discount,
                                         self.model_iterations, self.cores_avail)
        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        self.policy_network.policy_model.model.save_weights(filepath=tmp_file_name)
        copy_.policy_network.policy_model.model.load_weights(tmp_file_name)
        os.remove(tmp_file_name)
        return copy_

    def __init__(self, abr_name, max_quality_change, deterministic, past_measurement_dimensions,
                 future_measurements_dimensions, cloning_epochs, drop_prob=0.1, hidden_dim=32, batch_size_cloning=64,
                 validation_split=0.2, scaling_factor_sigma=1, future_reward_discount=0.99, model_iterations=20,
                 cores_avail=1, rde_distill_epochs=20, pretrain=True, balanced=False):

        """
        :type validation_split: object
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param lookback:
        """

        super().__init__(abr_name, max_quality_change, deterministic)
        self.rde_distill_epochs = rde_distill_epochs
        self.cores_avail = cores_avail
        self.model_iterations = model_iterations
        self.future_measurements_dimensions = future_measurements_dimensions
        self.validation_split = validation_split
        self.future_reward_discount = future_reward_discount
        self.pretrain = pretrain
        self.balanced = balanced
        self.past_measurement_dimensions = past_measurement_dimensions
        self.n_actions = max_quality_change * 2 + 1
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.value_history = None
        self.policy_history = None
        self.pretrain_history = None

        self.value_history_last = None
        self.policy_history_last = None

        self.batch_size_cloning = batch_size_cloning
        self.cloning_epochs = cloning_epochs
        self.trajectory_dummy = Trajectory()
        self.policy_network = KerasGAIL(past_measurement_dimensions=self.past_measurement_dimensions,
                                        future_measurements_dimensions=self.future_measurements_dimensions,
                                        hidden_dim=self.hidden_dim,
                                        action_dimension=self.n_actions,
                                        drop_prob=self.drop_prob)
        self.scaling_factor_sigma = scaling_factor_sigma
        self.bc_cloning_network = KerasEmbedder(past_measurement_dimensions=self.past_measurement_dimensions,
                                                future_measurements_dimensions=self.future_measurements_dimensions,
                                                hidden_dim=hidden_dim,
                                                embedding_dimension=self.n_actions,
                                                drop_prob=drop_prob)
        self.rnd_cloning_network = KerasEmbedder(past_measurement_dimensions=self.past_measurement_dimensions,
                                                 future_measurements_dimensions=self.future_measurements_dimensions,
                                                 hidden_dim=hidden_dim,
                                                 embedding_dimension=self.n_actions,
                                                 drop_prob=drop_prob)
        self.value_model = KerasValue(past_measurement_dimensions=self.past_measurement_dimensions,
                                      future_measurements_dimensions=self.future_measurements_dimensions,
                                      hidden_dim=self.hidden_dim,
                                      drop_prob=self.drop_prob)

    def reset_learning(self):
        self.value_history = None
        self.policy_history = None
        self.pretrain_history = None

        self.value_history_last = None
        self.policy_history_last = None

        self.policy_network = KerasGAIL(past_measurement_dimensions=self.past_measurement_dimensions,
                                        future_measurements_dimensions=self.future_measurements_dimensions,
                                        hidden_dim=self.hidden_dim,
                                        action_dimension=self.n_actions,
                                        drop_prob=self.drop_prob)
        self.bc_cloning_network = KerasEmbedder(past_measurement_dimensions=self.past_measurement_dimensions,
                                                future_measurements_dimensions=self.future_measurements_dimensions,
                                                hidden_dim=self.hidden_dim,
                                                embedding_dimension=self.n_actions,
                                                drop_prob=self.drop_prob)
        self.rnd_cloning_network = KerasEmbedder(past_measurement_dimensions=self.past_measurement_dimensions,
                                                 future_measurements_dimensions=self.future_measurements_dimensions,
                                                 hidden_dim=self.hidden_dim,
                                                 embedding_dimension=self.n_actions,
                                                 drop_prob=self.drop_prob)
        self.value_model = KerasValue(past_measurement_dimensions=self.past_measurement_dimensions,
                                      future_measurements_dimensions=self.future_measurements_dimensions,
                                      hidden_dim=self.hidden_dim,
                                      drop_prob=self.drop_prob)

    def clone_from_trajectory(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
                              video_csv_list, log_steps=False):
        self.reset_learning()
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
        ###############################################################################################################
        #### Fit first network
        random_prediction_training = self.rnd_cloning_network.model.predict([state_t_training, state_t_future_training])
        random_prediction_testing = self.rnd_cloning_network.model.predict([state_t_testing, state_t_future_testing])

        testing_data = ([state_t_testing, state_t_future_testing], random_prediction_testing)
        self.pretrain_distill_history = self.bc_cloning_network.model.fit(
            [state_t_training, state_t_future_training],
            random_prediction_training,
            validation_data=testing_data,
            epochs=self.rde_distill_epochs, verbose=0, shuffle=True,
            callbacks=self.early_stopping).history
        trained_prediction_training = self.bc_cloning_network.model.predict([state_t_training, state_t_future_training])

        scaling_factor = np.random.random(size=100) * 100  # Pick the hyperparameter randomly
        rewards = [np.exp(-fact * (np.square(trained_prediction_training - random_prediction_training)).mean(
            axis=-1)).flatten().mean() for fact in scaling_factor]
        self.scaling_factor_sigma = scaling_factor[np.argmin(np.abs(np.array(rewards) - 1.0))]
        print('Choosen Scaling factor %.2f' % self.scaling_factor_sigma)

        red_trajectory_generator_training = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                     trace_list=trace_list[train_idx],
                                                                     video_csv_list=video_csv_list[train_idx])
        keras_class_weighting = None
        if self.balanced:
            keras_class_weighting = class_weight.compute_class_weight('balanced',
                                                                      np.unique(action_training.argmax(1)),
                                                                      action_training.argmax(1))

        weight_filepaths = []
        if self.pretrain:
            self.pretrain_bc_history = self.policy_network.policy_model.model.fit(
                [state_t_training, state_t_future_training],
                action_training,
                validation_data=([state_t_testing, state_t_future_testing], action_testing),
                epochs=self.rde_distill_epochs, verbose=0,
                callbacks=self.early_stopping, class_weight=keras_class_weighting).history

        for cloning_iteration in tqdm(range(self.cloning_epochs), desc='Cloning Epochs'):
            """
            Iterations of the RED algorithm
            """
            training_evaluation, training_trajectories = red_trajectory_generator_training.create_trajectories(
                random_action_probability=0)
            training_trajectories.convert_list()
            state_t_training_sampled = training_trajectories.trajectory_state_t_arr
            state_t_future_training_sampled = training_trajectories.trajectory_state_t_future
            action_sampled = to_categorical(training_trajectories.trajectory_action_t_arr, num_classes=self.n_actions)
            action_likelihood_sampled = training_trajectories.trajectory_likelihood

            bc_clone_prediction = self.bc_cloning_network.model.predict([state_t_training_sampled,
                                                                         state_t_future_training_sampled])
            random_prediction = self.rnd_cloning_network.model.predict([state_t_training_sampled,
                                                                        state_t_future_training_sampled])
            # report_var(·) = exp(−σ1‖fˆθ(·)−fθ(·)‖22)
            reward = np.exp(-self.scaling_factor_sigma * (np.square(bc_clone_prediction - random_prediction)).mean(
                axis=-1)).flatten()  # Scales to 1.0 as recommended
            # Train the value net
            future_reward_obtained = []
            i_start = 0
            i_end = 0
            for evaluation_dataframe in training_evaluation:
                i_end += len(evaluation_dataframe.streaming_session_evaluation)
                reward_transform = list(reward[i_start:i_end])
                # We ignore the last reward obtained as we don't have a corresponding state
                for i in range(1, len(reward_transform))[::-1]:
                    exponent = (len(reward_transform) - i)
                    reward_transform[i - 1] += reward_transform[i] * self.future_reward_discount ** exponent
                future_reward_obtained += reward_transform
                i_start = i_end
            future_reward_obtained = np.array(future_reward_obtained).reshape((-1, 1))
            future_reward_predicted = self.value_model.model.predict(
                [state_t_training_sampled, state_t_future_training_sampled])
            history = self.value_model.model.fit([state_t_training_sampled, state_t_future_training_sampled],
                                                 future_reward_obtained,
                                                 validation_split=0.2, epochs=self.model_iterations,
                                                 verbose=0,
                                                 shuffle=True).history  # Repeated early stopping callback introduce errors
            self.value_history_last = history.copy()
            history = self.keep_last_entry(history)

            if self.value_history is None:
                self.value_history = history
            else:
                for k, v in history.items():
                    self.value_history[k] += history[k]

            estimated_advantage = future_reward_obtained - future_reward_predicted
            estimated_advantage = estimated_advantage

            history = self.policy_network.gail_training_model.fit(
                [state_t_training_sampled, state_t_future_training_sampled,
                 estimated_advantage, action_likelihood_sampled],
                action_sampled,
                validation_split=self.validation_split, epochs=self.model_iterations,
                verbose=0, shuffle=True).history
            self.policy_history_last = history.copy()
            history = self.keep_last_entry(history)

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
            self.policy_network.policy_model.model.save_weights(filepath=weight_filepath)
            weight_filepaths.append(weight_filepath)

        best_iteration = self.opt_policy_opt_operator(self.policy_history[self.opt_policy_value_name])
        self.policy_network.policy_model.model.load_weights(weight_filepaths[best_iteration])
        for path in weight_filepaths:
            os.remove(path)

    def save_model(self, weight_filepath):
        self.policy_network.policy_model.model.save_weights(filepath=weight_filepath)

    def load_model(self, weight_filepath):
        self.policy_network.policy_model.model.load_weights(weight_filepath)

    def score(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
              video_csv_list, add_data=False):
        expert_trajectory.convert_list()
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list,
                                                                               video_csv_list=video_csv_list)
        state_t_testing = expert_trajectory.trajectory_state_t_arr
        state_t_future_testing = expert_trajectory.trajectory_state_t_future
        approx_action = self.policy_network.policy_model.model.predict([state_t_testing, state_t_future_testing]).argmax(-1)
        expert_action = expert_trajectory.trajectory_action_t_arr

        behavioural_cloning_evaluation, behavioural_cloning_evaluation_trajectory = behavioural_cloning_trace_generator_testing.create_trajectories(
            random_action_probability=0, cores_avail=1)
        return self.score_comparison(expert_evaluation=expert_evaluation,
                                     expert_trajectory=expert_trajectory,
                                     expert_action=expert_action,
                                     approx_evaluation=behavioural_cloning_evaluation,
                                     approx_trajectory=behavioural_cloning_evaluation_trajectory,
                                     approx_action=approx_action, add_data=add_data)



class KerasGAILDIFF:

    def __init__(self,
                 past_measurement_dimensions,
                 future_measurements_dimensions,
                 hidden_dim,
                 action_dimension,
                 drop_prob=0.2):
        """
        https://arxiv.org/abs/1606.03476
        :param past_measurement_dimensions:
        :param future_measurements_dimensions:
        :param hidden_dim:
        :param action_dimension:
        :param drop_prob:
        """
        self.discriminator_model = KerasDiscriminator(past_measurement_dimensions,
                                                      future_measurements_dimensions,
                                                      hidden_dim,
                                                      action_dimension,
                                                      drop_prob=drop_prob)

        self.policy_model = KerasPolicy(past_measurement_dimensions,
                                        future_measurements_dimensions,
                                        hidden_dim,
                                        action_dimension,
                                        drop_prob=drop_prob)
        # ---------------------------------------------------------------------------------------------------------------------------------
        # Build GAIL Model
        self.discriminator_model.discriminator_past_GRU.trainable = False
        self.discriminator_model.discriminator_future_GRU.trainable = False
        self.discriminator_model.discriminator_dense_1.trainable = False
        self.discriminator_model.discriminator_dense_2.trainable = False
        self.discriminator_model.discriminator_dense_final.trainable = False

        concatenated = concatenate(
            [self.discriminator_model.discriminator_past_GRU(self.policy_model.policy_past_input),
             self.discriminator_model.discriminator_future_GRU(
                 self.policy_model.policy_future_input)])
        concatenated = self.discriminator_model.discriminator_dense_1(concatenated)
        concatenated = concatenate([concatenated, self.policy_model.policy_action_output])
        concatenated = self.discriminator_model.discriminator_dense_2(concatenated)
        gail_likelihood = self.discriminator_model.discriminator_dense_final(concatenated)
        self.gail_model = Model(inputs=[self.policy_model.policy_past_input, self.policy_model.policy_future_input],
                                outputs=gail_likelihood)
        self.gail_model.compile(loss="categorical_crossentropy", optimizer='adam')
        self.discriminator_model = self.discriminator_model.model
        self.policy_model = self.policy_model.model


class GAILDifferentiable(ABRPolicy):

    def __init__(self, abr_name, max_quality_change, deterministic, past_measurement_dimension,
                 future_measurements_dimensions, cloning_epochs, drop_prob=0.1,
                 hidden_dim=32, batch_size_cloning=64, validation_split=0.2, pretrain=False, pretrain_max_epochs=20,
                 random_action_probability=0.9, random_action_probability_decay=0.75, adverserial_max_epochs=20,
                 cores_avail=1, balanced=False):
        """
        :type validation_split: object
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param lookback:
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.cores_avail = cores_avail
        self.random_action_probability_decay = random_action_probability_decay
        self.random_action_probability = random_action_probability
        self.pretrain = pretrain
        self.past_measurement_dimension = past_measurement_dimension
        self.future_measurements_dimensions = future_measurements_dimensions

        self.n_actions = max_quality_change * 2 + 1
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.gail_network = KerasGAILDIFF(past_measurement_dimensions=self.past_measurement_dimension,
                                          future_measurements_dimensions=self.future_measurements_dimensions,
                                          hidden_dim=hidden_dim,
                                          action_dimension=self.n_actions,
                                          drop_prob=drop_prob)
        self.pretrain_history = None
        self.discriminator_history = None
        self.value_history = None
        self.policy_history = None

        self.pretrain_history_last = None
        self.discriminator_history_last = None
        self.value_history_last = None
        self.policy_history_last = None

        self.pretrain_max_epochs = pretrain_max_epochs

        self.batch_size_cloning = batch_size_cloning
        self.cloning_epochs = cloning_epochs
        self.trajectory_dummy = Trajectory()
        self.validation_split = validation_split
        self.adverserial_max_epochs = adverserial_max_epochs
        self.balanced = balanced

    def copy(self):
        copy_ = GAILDifferentiable(self.abr_name, self.max_quality_change, self.deterministic,
                                   self.past_measurement_dimension,
                                   self.future_measurements_dimensions, self.cloning_epochs, self.drop_prob,
                                   self.hidden_dim, self.batch_size_cloning, self.validation_split, self.pretrain,
                                   self.pretrain_max_epochs,
                                   self.random_action_probability, self.random_action_probability_decay,
                                   self.adverserial_max_epochs, self.cores_avail)

        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        self.gail_network.gail_model.save_weights(filepath=tmp_file_name)
        copy_.gail_network.gail_model.load_weights(tmp_file_name)
        os.remove(tmp_file_name)
        return copy_

    def reset_learning(self):
        self.pretrain_history = None
        self.discriminator_history = None
        self.value_history = None
        self.policy_history = None

        self.pretrain_history_last = None
        self.discriminator_history_last = None
        self.value_history_last = None
        self.policy_history_last = None

        self.gail_network = KerasGAILDIFF(past_measurement_dimensions=self.past_measurement_dimension,
                                          future_measurements_dimensions=self.future_measurements_dimensions,
                                          hidden_dim=self.hidden_dim,
                                          action_dimension=self.n_actions,
                                          drop_prob=self.drop_prob)

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

        action_prob = self.gail_network.policy_model.predict([state_t, state_t_future])
        self.likelihood_last_decision_val = max(action_prob)
        if self.deterministic:
            next_quality_switch_idx = np.argmax(action_prob)
        else:
            probability = action_prob.flatten()
            next_quality_switch_idx = np.random.choice(np.arange(len(probability)), size=1, p=probability)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality_switch_idx], a_min=0,
                               a_max=streaming_enviroment.max_quality_level)
        return next_quality

    def likelihood_last_decision(self):
        return self.likelihood_last_decision_val

    def reset(self):
        pass

    def split_input_data(self, expert_evaluation, expert_trajectory):
        train_idx, test_test2_idx = train_test_split(np.arange(len(expert_evaluation)),
                                                     test_size=self.validation_split * 2.)
        test_idx, test2_idx = train_test_split(test_test2_idx, test_size=self.validation_split)

        state_train_idx = np.array(
            [expert_trajectory.trajectory_sample_association[expert_evaluation[idx].name] for idx in
             train_idx]).flatten()
        state_test_idx = np.array(
            [expert_trajectory.trajectory_sample_association[expert_evaluation[idx].name] for idx in
             test_idx]).flatten()
        return train_idx, test_idx, test2_idx, state_train_idx, state_test_idx

    def calculate_reference_reward(self, expert_evaluation, test_idx):
        return [frame.reward.mean() for frame in [expert_evaluation[i].streaming_session_evaluation for i in test_idx]]

    def clone_from_trajectory(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment,
                              trace_list,
                              video_csv_list, log_steps=False):
        self.reset_learning()
        self.fit_clustering_scorer(expert_trajectory)

        # Select the training/validation traces
        trace_list = np.array(trace_list)
        video_csv_list = np.array(video_csv_list)
        expert_trajectory.convert_list()
        train_idx, test_idx, test2_idx, state_train_idx, state_test_idx = self.split_input_data(expert_evaluation,
                                                                                                expert_trajectory)
        state_t_expert = expert_trajectory.trajectory_state_t_arr
        state_t_future_expert = expert_trajectory.trajectory_state_t_future
        expert_action = expert_trajectory.trajectory_action_t_arr
        expert_action = to_categorical(expert_action, num_classes=self.n_actions)
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list[test2_idx],
                                                                               video_csv_list=video_csv_list[test2_idx])
        behavioural_cloning_trace_generator_training = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                                trace_list=trace_list[train_idx],
                                                                                video_csv_list=video_csv_list[
                                                                                    train_idx])

        mean_reward_expert_run_test = self.calculate_reference_reward(expert_evaluation, test2_idx)
        validation_data = (
            [state_t_expert[state_test_idx], state_t_future_expert[state_test_idx]], expert_action[state_test_idx])

        keras_class_weighting = None
        if self.balanced :
            keras_class_weighting = class_weight.compute_class_weight('balanced',
                                                                  np.unique(expert_action[state_train_idx].argmax(1)),
                                                                  expert_action[state_train_idx].argmax(1))


        if self.pretrain:
            history = self.gail_network.policy_model.fit(
                [state_t_expert[state_train_idx], state_t_future_expert[state_train_idx]],
                expert_action[state_train_idx],
                validation_data=validation_data, epochs=self.pretrain_max_epochs, verbose=0,
                shuffle=True,
                callbacks=self.early_stopping,
                class_weight=keras_class_weighting).history
            self.pretrain_history_last = history.copy()
            self.pretrain_history = self.keep_last_entry(history)

        current_random_action_probability = self.random_action_probability
        weight_filepaths = []
        for cloning_iteration in tqdm(range(self.cloning_epochs), desc='Cloning Epochs'):
            # --------------------------------------------------------------------------------------------------
            # Train Discriminator
            _, behavioural_cloning_trajectory = behavioural_cloning_trace_generator_training.create_trajectories(
                random_action_probability=0)
            state_t, state_t_future, state_t1, state_t1_future, approximation_actions = behavioural_cloning_trajectory.sample(
                behavioural_cloning_trajectory.n_trajectories)
            train_idx_clone, test_idx_clone = train_test_split(np.arange(len(state_t)), test_size=self.validation_split)

            approximation_actions = to_categorical(approximation_actions, num_classes=self.n_actions)

            state_t_train = np.vstack([state_t[train_idx_clone], state_t_expert[state_train_idx]])
            state_t_future_train = np.vstack([state_t_future[train_idx_clone], state_t_future_expert[state_train_idx]])
            action_train = np.vstack([approximation_actions[train_idx_clone], expert_action[state_train_idx]])
            target_label_train = to_categorical(np.vstack([0] * len(train_idx_clone) + [1] * len(state_train_idx)),
                                                num_classes=2)

            state_t_validation = np.vstack([state_t[test_idx_clone], state_t_expert[state_test_idx]])
            state_t_future_validation = np.vstack(
                [state_t_future[test_idx_clone], state_t_future_expert[state_test_idx]])
            action_validation = np.vstack([approximation_actions[test_idx_clone], expert_action[state_test_idx]])
            target_label_validation = to_categorical(np.vstack([0] * len(test_idx_clone) + [1] * len(state_test_idx)),
                                                     num_classes=2)

            validation_data_discriminator = (
                [state_t_validation, state_t_future_validation, action_validation], target_label_validation)

            data_train = [state_t_train, state_t_future_train, action_train]

            history = self.gail_network.discriminator_model.fit(data_train, target_label_train,
                                                                validation_data=validation_data_discriminator,
                                                                epochs=self.adverserial_max_epochs,
                                                                verbose=0, callbacks=self.early_stopping).history
            self.discriminator_history_last = history.copy()
            history = self.keep_last_entry(history)

            if self.discriminator_history is None:
                self.discriminator_history = history
            else:
                for k, v in history.items():
                    self.discriminator_history[k] += history[k]

            # Train the policy to fool the the discriminator
            fooling_label = to_categorical([1] * len(state_t), num_classes=2)
            history = self.gail_network.gail_model.fit([state_t, state_t_future], fooling_label,
                                                       validation_split=self.validation_split,
                                                       epochs=self.adverserial_max_epochs,
                                                       verbose=0, callbacks=self.early_stopping).history
            self.policy_history_last = history.copy()
            history = self.keep_last_entry(history)

            if self.policy_history is None:
                self.policy_history = history
                self.policy_history['enviroment_reward_approximation'] = []
                self.policy_history['enviroment_reward_expert'] = []
                self.policy_history['enviroment_reward_difference'] = []
            else:
                for k, v in history.items():
                    self.policy_history[k] += history[k]
            behavioural_cloning_evaluation, _ = behavioural_cloning_trace_generator_testing.create_trajectories(
                random_action_probability=0)
            clone_list = [frame.reward.mean() for frame in behavioural_cloning_evaluation]
            reward_difference = np.mean(clone_list) - np.mean(mean_reward_expert_run_test)
            self.policy_history['enviroment_reward_approximation'] += [np.mean(clone_list)]
            self.policy_history['enviroment_reward_expert'] += [np.mean(mean_reward_expert_run_test)]
            self.policy_history['enviroment_reward_difference'] += [reward_difference]
            current_random_action_probability *= self.random_action_probability_decay
            weight_filepath = self.rnd_id + '_policy_network_iteration_%d.h5' % cloning_iteration
            self.gail_network.gail_model.save_weights(filepath=weight_filepath)
            weight_filepaths.append(weight_filepath)
        best_iteration = self.opt_policy_opt_operator(self.policy_history[self.opt_policy_value_name])
        self.gail_network.gail_model.load_weights(weight_filepaths[best_iteration])
        logger.info('Restoring best iteration %d' % best_iteration)
        for path in weight_filepaths:
            os.remove(path)

    def save_model(self, weight_filepath):
        self.gail_network.gail_model.save_weights(filepath=weight_filepath)

    def load_model(self, weight_filepath):
        self.gail_network.gail_model.load_weights(weight_filepath)


class GAILPPO(ABRPolicy):
    def __init__(self, abr_name, max_quality_change, deterministic,
                 past_measurement_dimensions,
                 future_measurements_dimensions, cloning_epochs, drop_prob=0.1,
                 hidden_dim=32, batch_size_cloning=64, validation_split=0.2, pretrain=False, pretrain_max_epochs=20,
                 random_action_probability=0.9, random_action_probability_decay=0.75, future_reward_discount=0.99,
                 adverserial_max_epochs=20, cores_avail=1, balanced=False):
        """
        https://arxiv.org/abs/1606.03476 with PPO as reward learning function
        :type validation_split: object
        :param abr_name:
        :param max_quality_change:
        :param deterministic:
        :param lookback:
        """
        super().__init__(abr_name, max_quality_change, deterministic)
        self.cores_avail = cores_avail
        self.future_reward_discount = future_reward_discount
        self.random_action_probability_decay = random_action_probability_decay
        self.random_action_probability = random_action_probability
        self.pretrain = pretrain
        self.past_measurement_dimensions = past_measurement_dimensions
        self.n_actions = max_quality_change * 2 + 1
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.future_measurements_dimensions = future_measurements_dimensions
        self.discriminator = KerasDiscriminator(past_measurement_dimensions=self.past_measurement_dimensions,
                                                future_measurements_dimensions=self.future_measurements_dimensions,
                                                hidden_dim=self.hidden_dim,
                                                action_dimension=self.n_actions,
                                                drop_prob=self.drop_prob)
        self.gail_model = KerasGAIL(past_measurement_dimensions=self.past_measurement_dimensions,
                                    future_measurements_dimensions=self.future_measurements_dimensions,
                                    hidden_dim=self.hidden_dim,
                                    action_dimension=self.n_actions,
                                    drop_prob=self.drop_prob)
        self.value_model = KerasValue(past_measurement_dimensions=self.past_measurement_dimensions,
                                      future_measurements_dimensions=self.future_measurements_dimensions,
                                      hidden_dim=self.hidden_dim,
                                      drop_prob=self.drop_prob)

        self.pretrain_history = None
        self.discriminator_history = None
        self.value_history = None
        self.policy_history = None

        self.pretrain_history_last = None
        self.discriminator_history_last = None
        self.value_history_last = None
        self.policy_history_last = None

        self.pretrain_max_epochs = pretrain_max_epochs
        self.adverserial_max_epochs = adverserial_max_epochs

        self.batch_size_cloning = batch_size_cloning
        self.cloning_epochs = cloning_epochs
        self.trajectory_dummy = Trajectory()
        self.validation_split = validation_split
        self.balanced = balanced

    def copy(self):
        copy_ = GAILPPO(self.abr_name, self.max_quality_change, self.deterministic,
                        self.past_measurement_dimensions,
                        self.future_measurements_dimensions, self.cloning_epochs, self.drop_prob,
                        self.hidden_dim, self.batch_size_cloning, self.validation_split, self.pretrain,
                        self.pretrain_max_epochs,
                        self.random_action_probability, self.random_action_probability_decay,
                        self.adverserial_max_epochs, self.cores_avail)

        tmp_file_name = self.randomString(self.rnd_string_length) + 'tmp_id'
        self.gail_model.policy_model.model.save_weights(filepath=tmp_file_name)
        copy_.gail_model.policy_model.model.load_weights(tmp_file_name)
        os.remove(tmp_file_name)
        return copy_

    def reset_learning(self):
        self.pretrain_history = None
        self.discriminator_history = None
        self.value_history = None
        self.policy_history = None

        self.pretrain_history_last = None
        self.discriminator_history_last = None
        self.value_history_last = None
        self.policy_history_last = None

        self.discriminator = KerasDiscriminator(past_measurement_dimensions=self.past_measurement_dimensions,
                                                future_measurements_dimensions=self.future_measurements_dimensions,
                                                hidden_dim=self.hidden_dim,
                                                action_dimension=self.n_actions,
                                                drop_prob=self.drop_prob)
        self.gail_model = KerasGAIL(past_measurement_dimensions=self.past_measurement_dimensions,
                                    future_measurements_dimensions=self.future_measurements_dimensions,
                                    hidden_dim=self.hidden_dim,
                                    action_dimension=self.n_actions,
                                    drop_prob=self.drop_prob)
        self.value_model = KerasValue(past_measurement_dimensions=self.past_measurement_dimensions,
                                      future_measurements_dimensions=self.future_measurements_dimensions,
                                      hidden_dim=self.hidden_dim,
                                      drop_prob=self.drop_prob)

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
        action_prob = self.gail_model.policy_model.model.predict([state_t, state_t_future])
        self.likelihood_last_decision_val = action_prob
        if self.deterministic:
            next_quality_switch_idx = np.argmax(action_prob)
        else:
            probability = action_prob.flatten()
            next_quality_switch_idx = np.random.choice(np.arange(len(probability)), size=1, p=probability)
        next_quality = np.clip(current_level + self.quality_change_arr[next_quality_switch_idx], a_min=0,
                               a_max=streaming_enviroment.max_quality_level)
        return next_quality

    def likelihood_last_decision(self):
        return self.likelihood_last_decision_val

    def reset(self):
        pass

    def split_input_data(self, expert_evaluation, expert_trajectory):
        train_idx, test_test2_idx = train_test_split(np.arange(len(expert_evaluation)),
                                                     test_size=self.validation_split * 2.)
        test_idx, test2_idx = train_test_split(test_test2_idx, test_size=self.validation_split)

        state_train_idx = np.array(
            [expert_trajectory.trajectory_sample_association[expert_evaluation[idx].name] for idx in
             train_idx]).flatten()
        state_test_idx = np.array(
            [expert_trajectory.trajectory_sample_association[expert_evaluation[idx].name] for idx in
             test_idx]).flatten()
        return train_idx, test_idx, test2_idx, state_train_idx, state_test_idx

    def calculate_reference_reward(self, expert_evaluation, test_idx):
        return [frame.reward.mean() for frame in [expert_evaluation[i].streaming_session_evaluation for i in test_idx]]

    def clone_from_trajectory(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment,
                              trace_list,
                              video_csv_list, log_steps=False):
        self.reset_learning()
        self.fit_clustering_scorer(expert_trajectory)

        # Select the training/validation traces
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
        weight_filepaths = []

        behavioural_cloning_trace_generator_training = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                                trace_list=trace_list[train_idx],
                                                                                video_csv_list=video_csv_list[
                                                                                    train_idx])

        keras_class_weighting = None
        if self.balanced:
            keras_class_weighting = class_weight.compute_class_weight('balanced',
                                                                      np.unique(action_training.argmax(1)),
                                                                      action_training.argmax(1))

        if self.pretrain:
            history = self.gail_model.policy_model.model.fit(
                [state_t_training, state_t_future_training],
                action_training,
                validation_data=([state_t_testing, state_t_future_testing], action_testing),
                epochs=self.pretrain_max_epochs, verbose=0,
                callbacks=self.early_stopping,class_weight = keras_class_weighting).history
            self.pretrain_history_last = history.copy()
            self.pretrain_history = self.keep_last_entry(history)

        for cloning_iteration in tqdm(range(self.cloning_epochs), desc='Cloning Epochs'):
            # --------------------------------------------------------------------------------------------------
            # Train Discriminator
            behavioural_cloning_training_evaluation, behavioural_cloning_training_trajectory = behavioural_cloning_trace_generator_training.create_trajectories(
                random_action_probability=0)

            behavioural_cloning_training_trajectory.convert_list()
            training_trajectory_state_t = behavioural_cloning_training_trajectory.trajectory_state_t_arr
            training_trajectory_state_t_future = behavioural_cloning_training_trajectory.trajectory_state_t_future
            behavioural_action = behavioural_cloning_training_trajectory.trajectory_action_t_arr
            behavioural_action_likelihood = behavioural_cloning_training_trajectory.trajectory_likelihood

            train_idx_clone, test_idx_clone = train_test_split(np.arange(len(training_trajectory_state_t)),
                                                               test_size=self.validation_split)

            behavioral_action = to_categorical(behavioural_action, num_classes=self.n_actions)

            state_t_train = np.vstack([training_trajectory_state_t[train_idx_clone], state_t_training])
            state_t_future_train = np.vstack(
                [training_trajectory_state_t_future[train_idx_clone], state_t_future_training])
            action_train = np.vstack([behavioral_action[train_idx_clone], action_training])
            target_label_train = to_categorical(np.vstack([0] * len(train_idx_clone) + [1] * len(action_training)),
                                                num_classes=2)

            state_t_validation = np.vstack([training_trajectory_state_t[test_idx_clone], state_t_testing])
            state_t_future_validation = np.vstack(
                [training_trajectory_state_t_future[test_idx_clone], state_t_future_testing])
            action_validation = np.vstack([behavioral_action[test_idx_clone], action_testing])
            target_label_validation = to_categorical(np.vstack([0] * len(test_idx_clone) + [1] * len(action_testing)),
                                                     num_classes=2)

            validation_data_discriminator = (
                [state_t_validation, state_t_future_validation, action_validation], target_label_validation)

            data_train = [state_t_train, state_t_future_train, action_train]

            history = self.discriminator.model.fit(data_train, target_label_train,
                                                   validation_data=validation_data_discriminator,
                                                   epochs=self.adverserial_max_epochs,
                                                   verbose=0).history  # Repeated early stopping callback introduce errors
            self.discriminator_history_last = history.copy()
            history = self.keep_last_entry(history)
            if self.discriminator_history is None:
                self.discriminator_history = history
            else:
                for k, v in history.items():
                    self.discriminator_history[k] += history[k]

            data_predict_discriminator = [training_trajectory_state_t,
                                          training_trajectory_state_t_future,
                                          behavioral_action]

            discriminator_prediction = self.discriminator.model.predict(data_predict_discriminator)[:, 1]

            reward = np.log(discriminator_prediction)  # Scales to 1.0 as recommended
            # Train the value net
            future_reward_obtained = []
            i_start = 0
            i_end = 0
            for evaluation_dataframe in behavioural_cloning_training_evaluation:
                i_end += len(evaluation_dataframe.streaming_session_evaluation)
                reward_transform = list(reward[i_start:i_end])
                # We ignore the last reward obtained as we don't have a corresponding state
                for i in range(1, len(reward_transform))[::-1]:
                    exponent = (len(reward_transform) - i)
                    reward_transform[i - 1] += reward_transform[i] * self.future_reward_discount ** exponent
                future_reward_obtained += reward_transform
                i_start = i_end
            future_reward_obtained = np.array(future_reward_obtained).reshape((-1, 1))
            future_reward_predicted = self.value_model.model.predict(
                [training_trajectory_state_t, training_trajectory_state_t_future])
            history = self.value_model.model.fit([training_trajectory_state_t, training_trajectory_state_t_future],
                                                 future_reward_obtained,
                                                 validation_split=0.2, epochs=self.adverserial_max_epochs,
                                                 verbose=0).history
            self.value_history_last = history.copy()
            history = self.keep_last_entry(history)

            if self.value_history is None:
                self.value_history = history
            else:
                for k, v in history.items():
                    self.value_history[k] += history[k]

            estimated_advantage = future_reward_obtained - future_reward_predicted
            estimated_advantage = estimated_advantage
            # --------------------------------------------------------------------------------------------------------
            # Fit with the PPO loss
            # print(np.mean(self.gail_model.policy_model.concatenate_informations.get_weights()))
            # print('---------' * 10)
            # print('---------' * 10)

            history = self.gail_model.gail_training_model.fit(
                [training_trajectory_state_t, training_trajectory_state_t_future,
                 estimated_advantage, behavioural_action_likelihood],
                behavioral_action,
                validation_split=self.validation_split, epochs=self.adverserial_max_epochs,
                verbose=0, shuffle=True).history
            # print(np.mean(self.gail_model.policy_model.concatenate_informations.get_weights()))
            # print('=========' * 10)
            # print('=========' * 10)
            self.policy_history_last = history.copy()
            history = self.keep_last_entry(history)

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
            self.gail_model.policy_model.model.save_weights(filepath=weight_filepath)
            weight_filepaths.append(weight_filepath)
        best_iteration = self.opt_policy_opt_operator(self.policy_history[self.opt_policy_value_name])
        self.gail_model.policy_model.model.load_weights(weight_filepaths[best_iteration])
        logger.info('Restoring best iteration %d' % best_iteration)
        for path in weight_filepaths:
            os.remove(path)

    def save_model(self, weight_filepath):
        self.gail_model.policy_model.model.save_weights(filepath=weight_filepath)

    def load_model(self, weight_filepath):
        self.gail_model.policy_model.model.load_weights(weight_filepath)

    def score(self, expert_evaluation, expert_trajectory: Trajectory, streaming_enviroment, trace_list,
              video_csv_list, add_data=False):
        expert_trajectory.convert_list()
        behavioural_cloning_trace_generator_testing = TrajectoryVideoStreaming(self, streaming_enviroment,
                                                                               trace_list=trace_list,
                                                                               video_csv_list=video_csv_list)
        state_t_testing = expert_trajectory.trajectory_state_t_arr
        state_t_future_testing = expert_trajectory.trajectory_state_t_future
        approx_action = self.gail_model.policy_model.model.predict([state_t_testing, state_t_future_testing]).argmax(-1)
        expert_action = expert_trajectory.trajectory_action_t_arr

        behavioural_cloning_evaluation, behavioural_cloning_evaluation_trajectory = behavioural_cloning_trace_generator_testing.create_trajectories(
            random_action_probability=0, cores_avail=1)
        return self.score_comparison(expert_evaluation=expert_evaluation,
                                     expert_trajectory=expert_trajectory,
                                     expert_action=expert_action,
                                     approx_evaluation=behavioural_cloning_evaluation,
                                     approx_trajectory=behavioural_cloning_evaluation_trajectory,
                                     approx_action=approx_action, add_data=add_data)

