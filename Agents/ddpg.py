import joblib
import numpy as np
import numpy.random as nr
import random

import tensorflow
import tf
from keras.backend.load_backend import k
from keras.optimizers import Adam
from tensorflow_core.python.training.tracking.util import keras_backend

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

class OUNoise():
    """Generate Noise using Ornstein Uhlenbeck Process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG(modelFreeAgent.ModelFreeAgent):
    displayName = 'DDPG'
    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True, "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True, "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True, "The distance in timesteps between target model updates")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(DDPG.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.target_update_interval = [int(arg) for arg in args[-paramLen:]]
        # self.actor = self.actor_network()
        # self.critic = self.critic_network()
        tensorflow_session = self._tensorflow_session

        empty_state = self.get_empty_state()

        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size,
                                                    TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.gamma = 0.001
        self.total_steps = 0
        self.tau = 0.001
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)
        # self.state_size = state_size
        # self.action_size = action_size
        self.critic, self.state_input, self.action_input = self.critic_network()
        # Generate carbon copy of the model so that we avoid divergence
        self.target_critic, self.target_c_weights, self.target_c_state = self.critic_network()

        # Generate the main model
        self.actor, self.actor_weights, self.actor_input = self.actor_network()
        # Generate carbon copy of the model so that we avoid divergence
        self.target_actor, self.target_a_weights, self.target_a_state = self.actor_network()

        # gradients for policy update
        self._action_gradients = tensorflow.gradients(self.critic.output,
                                                      self.action_input)

        # actor gradients
        self.action_gradients = tensorflow.placeholder(tensorflow.float32, [None, self.action_size])
        self._parameter_gradients = tensorflow.gradients(self.actor.output,
                                                         self.actor_weights,
                                                         -self.action_gradients)
        self._gradients = zip(self._parameter_gradients, self.target_a_weights)

        # Define the optimisation function
        self._optimize = tensorflow.train.AdamOptimizer(self.gamma).apply_gradients(self._gradients)
        # Let tensorflow and keras work together
        keras_backend.set_session(tensorflow_session)

        # tensorflow_session.run(tensorflow.initialize_all_variables())

    def _generate_tensorflow_session(self):
        """ Generate tensorflow session"""
        config = tf.compat.v1.ConfigProto
        config.gpu_options.allow_growth = True
        return tensorflow.Session(config=config)

    def critic_network(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten, multiply

        input_a = Input(shape=self.state_size)
        input_b = Input(shape=(self.action_size,))
        x = Flatten()(input_a)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = multiply([x, input_b])
        critic_model = Model(inputs=[input_a, input_b], outputs=outputs)
        critic_model.compile(loss='mse', optimizer=Adam(lr=0.001))
        target_critic_model = Model(inputs=[input_a, input_b], outputs=outputs)
        target_critic_model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return critic_model, target_critic_model

    def actor_network(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten

        input_a = Input(shape=self.state_size)
        x = Flatten()(input_a)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = x
        actor_model = Model(inputs=[input_a], outputs=outputs)
        actor_model.compile(loss='mse', optimizer=Adam(lr=0.001))
        target_actor_model = Model(inputs=[input_a], outputs=outputs)
        target_actor_model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return actor_model, target_actor_model

    def sample(self):
        return self.memory.sample(self.batch_size)

    def choose_action(self, state):
        qval = self.predict(state, False)
        epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * self.time_steps)
        # TODO: Put epsilon at a level near this
        if random.random() > epsilon:
            action = np.argmax(qval)
        else:
            action = self.state_size.sample()
        noise = OUNoise
        policy_action = action + noise
        return policy_action

    def train(self):
        """Train DDPG agent from current memory"""
        if len(self.memory) > self.batch_size:
            self._train()

    def _train(self):
        """Sample and train actor and critic"""
        # batch = self.remember()
        states, actions, rewards, done, next_states = self.sample()
        self._train_critic(states, actions, next_states, done, rewards)
        self.train_actor(states)
        self._update_target_models()

    def train_actor(self, states):
        """
        Trains the actor network using the calculated deterministic policy gradients.
        """
        gradients = self._get_gradients(states)
        self.actor._train_actor(states, gradients)

    def _train_actor(self, states, action_gradients):
        # todo better explanation of the inputs to this method
        """
        Update weights of the main network
        """
        self.tensorflow_session.run(self._optimize,
                                     feed_dict={self.states: states, self._action_gradients: action_gradients
                                                })

    def _train_critic(self, states, actions, next_states, done, rewards):
        """
        Trains the critic network
        """
        q_targets = self._get_q_targets(next_states, done, rewards)
        self.critic.train(states, actions, q_targets)

    def _get_q_targets(self, next_states, done, rewards):
        """
        Calculates the q targets with the following formula
        q_value = reward + gamma * next_q_value
        """
        next_actions = self.critic.predict(next_states)
        next_q_values = self.target_critic.predict(next_states, next_actions)
        q_targets = [reward if this_done else reward + next_q_value
                     for (reward, next_q_value, this_done)
                     in zip(rewards, next_q_values, done)]
        return q_targets

    def train_target_model(self):
        """
        Updates the weights of the target network to slowly track the critic
        network.
            target_weights = tau * main_weights + (1 - tau) * target_weights
        """
        main_weight = self.critic.get_weights()
        target_weight = self.target_critic.get_weights()
        target_weights = self.tau * main_weight + (1 - self.tau) * target_weight
        self.target_critic.set_weights(target_weights)

    def train_target_model_actor(self):
        """
        Updates the weights of the target network to slowly track the main
        network.
            target_weights = tau * main_weights + (1 - tau) * target_weights
        """
        main_weight = self.actor.get_weights()
        target_weight = self.target_actor.get_weights()
        target_weights = self.tau * main_weight + (1 - self.tau) * target_weight
        self.target_actor.set_weights(target_weights)

    def _get_gradients(self, states):
        """
        Calculate the Deterministic Policy Gradient for Actor training
        """
        action_for_gradients = self.actor.predict(states)
        gr = self.critic.get_gradients(states, action_for_gradients)
        return gr

    def _update_target_models(self):
        """
        Updates the target models to slowly track the main models
        """
        self.critic.train_target_model()
        self.actor.train_target_model_actor()

    def get_gradients(self, states, actions):
        """Returns the gradients."""

        return self.tensorflow_session.run(self._action_gradients, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })[0]

    def addToMemory(self, state, action, reward, new_state, done):
        """
        Add transitions to memory
        """
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):
        """
        Store transitions, calculate and minimize loss
        """
        self.addToMemory(state, action, reward, new_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        mini_batch = self.sample()

        states, actions, rewards, done, next_states = mini_batch

        with tf.GradientTape() as tape:
            q_target_val = self._get_q_targets(next_states, done, rewards)
            q_val = self.predict(state, done)

            loss = tf.reduce_mean((np.subtract(q_target_val, q_val)) ** 2)
            # Minimize the loss
            self._optimize.zero_grad()
            loss.backward()
            # grads_q = tape.gradient(critic_loss, self.q_mu.trainable_variables)
        return loss

        # loss = self.model.train_on_batch(X_train, Y_train)
        # self.updateTarget()
        # return loss

    def save(self, filename):
        mem1 = self.target_actor.get_weights()
        mem2 = self.target_critic.get_weights()
        joblib.dump((DDPG.displayName, mem1, mem2), filename)

    def load(self, filename):
        name, mem = joblib.load(filename)
        if name != DDPG.displayName:
            print('load failed')
        else:
            self.actor.set_weights(mem)
            self.target_actor.set_weights(mem)
            self.critic.set_weights(mem)
            self.target_critic.set_weights(mem)

    def memsave(self):
        return self.actor.get_weights(), self.critic.get_weights()

    def memload(self, mem):
        self.actor.set_weights(mem)
        self.target_actor.set_weights(mem)
        self.critic.set_weights(mem)
        self.target_critic.set_weights(mem)

    def reset(self):
        self.DDPG.clear()