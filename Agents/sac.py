import joblib
import tensorflow as tf
from typing import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, multiply
import numpy as np

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.Transition_Frame import TransitionFrame

tf.keras.backend.set_floatx('float64')


def soft_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable], tau: float) -> None:
    """Move each source variable by a factor of tau towards the corresponding target variable.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
        tau {float} -- How much to change to source var, between 0 and 1.
    """
    if len(source_vars) != len(target_vars):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source_vars, target_vars):
        # target = tf.cast(target, float)
        # source = tf.cast(source, float)
        # print(source.dtype)
        target.assign((1.0 - tau) * target + tau * source)


def hard_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable]):
    """Copy source variables to target variables.
       Arguments:
           source_vars {Sequence[tf.Variable]} -- Source variables to copy from
           target_vars {Sequence[tf.Variable]} -- Variables to copy data to
       """
    # Tau of 1, so get everything from source and keep nothing from target
    soft_update(source_vars, target_vars, 1.0)


class SAC(modelFreeAgent.ModelFreeAgent):
    displayName = 'SAC'

    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True,
                                                             "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True,
                                                             "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True,
                                                             "The distance in timesteps between target model updates"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Tau', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Temperature', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update")
                     ]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):

        # Initializing model parameters
        paramLen = len(SAC.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval, self.tau, self.temperature = [int(arg) for arg in args[-paramLen:]]
        self.logprob_epsilon = 1e-6  # For numerical stability when computing tf.log
        self.polyak_coef = 0.01
        self.total_steps = 0

        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size,
                                                    TransitionFrame(empty_state, -1, 0, empty_state, False))
        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.actor_network = self.actorModel()

        self.softq_network = self.q_network()
        self.softq_target_network = self.q_network()

        self.softq_network2 = self.q_network()
        self.softq_target_network2 = self.q_network()

        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        input2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)

        self.softq_network([input1, input2])
        self.softq_target_network([input1, input2])
        hard_update(self.softq_network.variables, self.softq_target_network.variables)

        self.softq_network2([input1, input2])
        self.softq_target_network2([input1, input2])
        hard_update(self.softq_network2.variables, self.softq_target_network2.variables)

        # Optimizers for the networks
        self.softq_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.softq_optimizer2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)

        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)

    def callqFunc(self):
        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        input2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)
        return self.softq_target_network([input1, input2])

    def callqFunc2(self):
        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        input2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)
        return self.softq_network([input1, input2])

    def callqFunc3(self):
        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        input2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)
        return self.softq_network2([input1, input2])

    def actorModel(self):
        w_bound = 3e-3

        input_shape = self.state_size
        inputA = Input(input_shape)
        x = Flatten()(inputA)
        x = Dense(24, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        mean = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound), bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))(x)
        log_std = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound))(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tf.compat.v1.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)

        return squashed_actions, logprob


    """def actorModel1(self, state):
        w_bound = 3e-3

        input_shape = self.state_size
        inputA = Input(input_shape)
        x = Flatten()(inputA)
        x = Dense(24, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        mean = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound), bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))(x)
        log_std = Dense(self.action_size, kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound))(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tf.compat.v1.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)

        return logprob"""

    def q_network(self):
        # Generate critic network model
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        inputB = Input(shape=(self.action_size,))
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = multiply([x, inputB])
        model = Model(inputs=[inputA, inputB], outputs=outputs)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model

    def value_network(self):
        # Generate critic network model
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        x = Flatten()(inputA)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=[inputA], outputs=x)
        model.compile(loss='mse', optimizer=self.critic_optimizer)
        return model

    def ou_noise(self, a, p=0.15, mu=0, differential=1e-1, sigma=0.2, dim=1):
        # Exploration noise generation
        return a + p * (mu - a) * differential + sigma * np.sqrt(differential) * np.random.normal(size=dim)

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def softq_value2(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network2(states, actions)

    # def actions(self, states: np.ndarray) -> np.ndarray:
       #  """Get the actions for a batch of states."""
       # return self.actor_network(states)[0]

    def choose_action(self, state):
        """Get the action for a single state."""
        bg_noise = np.zeros(self.action_size)
        bg_noise = self.ou_noise(bg_noise, dim=self.action_size)
        u, v = self.predict(state, False)
        sampled_actions = u.np.ndarray
        sampled_actions = np.squeeze(sampled_actions)
        sampled_actions = sampled_actions + bg_noise
        # Clipping action between bounds -0.3 and 0.3
        legal_action = np.clip(sampled_actions, -0.3, 0.3)[0]
        legal_action = np.squeeze(legal_action)
        action_returned = legal_action.astype(int)
        print("action chosen")
        return action_returned

    # def step(self, obs):
        # return self.actor_network(obs)[0]

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):

        self.addToMemory(state, action, reward, new_state, done)
        loss = 0

        if len(self.memory) < 2*self.batch_size:
            # print("loss = 0")
            return loss
        mini_batch = self.sample()
        states, actions, next_states, rewards, dones = self.learn(mini_batch)
        states = states.astype(float)
        next_states = next_states.astype(float)

        # Computing action and a_tilde
        action = self.actorModel()
        action_logprob2 = self.actor_network(states)
        print(states.shape)
        print(actions.shape)
        print(self.allBatchMask.shape)

        input1 = tf.keras.Input(shape=self.state_size, dtype=tf.float64)
        input2 = tf.keras.Input(shape=self.action_size, dtype=tf.float64)

        value_target1 = self.softq_network([input1, input2])
        value_target2 = self.softq_network2([input1, input2])

        # Taking the minimum of the q-functions values
        next_value_batch = tf.math.minimum(value_target1, value_target2) - self.temperature * action_logprob2

        # Computing target for q-functions
        softq_targets = rewards + self.gamma * (1 - dones) * tf.reshape(next_value_batch, [-1])
        # softq_targets = tf.reshape(softq_targets, [self.batch_size, 1])

        # Gradient descent for the first q-function
        with tf.GradientTape() as softq_tape:
            softq = self.softq_network([input1, input2])
            softq_loss = tf.reduce_mean(tf.square(softq - softq_targets))

        # Gradient descent for the second q-function
        with tf.GradientTape() as softq_tape2:
            softq2 = self.softq_network2([input1, input2])
            softq_loss2 = tf.reduce_mean(tf.square(softq2 - softq_targets))

        # Gradient ascent for the policy (actor)
        with tf.GradientTape() as actor_tape:
            # actions = self.actorModel()
            actions, action_logprob = self.actor_network(states)
            new_softq = tf.math.minimum(self.softq_network([input1, input2]), self.softq_network2([input1, input2]))

            # Loss implementation from the pseudocode -> works worse
            # actor_loss = tf.reduce_mean(action_logprob - new_softq)

            # New actor_loss -> works better
            advantage = tf.stop_gradient(action_logprob - new_softq)
            actor_loss = tf.reduce_mean(action_logprob * advantage)

        # print(self.actorModel.layers[0].trainable_weights)

        # Computing the gradients with the tapes and applying them
        softq_gradients = softq_tape.gradient(softq_loss, self.softq_network.trainable_weights)
        softq_gradients2 = softq_tape2.gradient(softq_loss2, self.softq_network2.trainable_weights)
        actor_gradients = actor_tape.gradient(actor_loss, self.actor_network.trainable_variables)

        # Minimize gradients wrt weights
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))
        self.softq_optimizer.apply_gradients(zip(softq_gradients, self.softq_network.trainable_weights))
        self.softq_optimizer2.apply_gradients(zip(softq_gradients2, self.softq_network2.trainable_weights))

        # Computing mean and variance of soft-q function
        # softq_mean, softq_variance = tf.nn.moments(softq, axes=[0])

        self.updateTarget()
        return softq_loss, actor_loss
        # tf.reduce_mean(action_logprob), softq_mean[0], tf.sqrt(softq_variance[0]),

    def updateTarget(self):
        if self.total_steps >= 2 * self.batch_size and self.total_steps % self.target_update_interval == 0:
            # Update the weights of the soft q-function target networks
            soft_update(self.softq_network.variables, self.softq_target_network.variables, self.polyak_coef)
            soft_update(self.softq_network2.variables, self.softq_target_network2.variables, self.polyak_coef)

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros(vector_length)
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def learn(self, mini_batch):

        # X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        states = (np.zeros((self.batch_size,) + self.state_size))
        actions = np.zeros((self.batch_size,) + (self.action_size,))
        next_states = (np.zeros((self.batch_size,) + self.state_size))
        rewards = np.zeros((self.batch_size,) + (self.action_size,))
        dones = np.zeros((self.batch_size,) + (self.action_size,))

        for index_rep, transition in enumerate(mini_batch):
            states[index_rep] = transition.state
            actions[index_rep] = self.create_one_hot(self.action_size, transition.action)
            next_states[index_rep] = transition.next_state
            rewards[index_rep] = transition.reward
            dones[index_rep] = transition.is_done

        return states, actions, next_states, rewards, dones

    def predict(self, state, isTarget):

        shape = (-1,) + self.state_size
        state = np.reshape(state, shape)
        # state = state(float)

        if isTarget:
            print("Target achieved")
            # result = self.q_network([state])
        else:
            result1 = self.actor_network(state)

        return result1

    def save(self, filename):
        mem = self.actor_network.get_weights()
        joblib.dump((SAC.displayName, mem), filename)
        print('Model saved')

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != SAC.displayName:
            print('load failed')
        else:
            self.actor_network.set_weights(mem)
            # self.target.set_weights(mem)

    def memsave(self):
        return self.actor_network.get_weights()

    def memload(self, mem):
        self.actor_network.set_weights(mem)
        # self.target.set_weights(mem)

    def reset(self):
        pass






