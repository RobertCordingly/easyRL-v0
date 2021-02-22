import numpy as np
import numpy.random as nr
import random

import tf
from keras.optimizers import Adam
from tensorflow_core.python.eager import tape

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.Transition_Frame import TransitionFrame

class OUNoise():
    """docstring for OUNoise"""
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

    def __init__(self, *args, state_size, action_size):
        paramLen = len(DDPG.newParameters)
        super().__init__(*args[:-paramLen])
        self.batch_size, self.memory_size, self.gamma, self.target_update_interval = [int(arg) for arg in args[-paramLen:]]
        self.actor = self.actor_network()
        self.critic = self.critic_network()
        self.target_actor = self.update_target_actor()
        self.target_critic = self.update_target_critic()
        # self.target_actor_network = self.create_actorNetwork()
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size,
                                                    TransitionFrame(empty_state, -1, 0, empty_state, False))
        self.total_steps = 0
        self.tau = 0.999
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)
        self.s_dim = state_size
        self.a_dim = action_size

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
        critic = Model(inputs=[input_a, input_b], outputs=outputs)
        target_critic = Model(inputs=[input_a, input_b], outputs=outputs)
        return critic, target_critic

    def actor_network(self):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input, Flatten

        input_a = Input(shape=self.state_size)
        x = Flatten()(input_a)
        x = Dense(24, input_dim=self.state_size, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size, activation='linear')(x)
        outputs = x
        actor = Model(inputs=[input_a], outputs=outputs)
        target_actor = Model(inputs=[input_a], outputs=outputs)
        return actor, target_actor

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

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):
        self.addToMemory(state, action, reward, new_state, done)
        loss = 0
        if len(self.memory) < 2*self.batch_size:
            return loss
        mini_batch = self.sample()

        a_train, b_train = self.calculate_target_actor(mini_batch)
        a_loss = self.target_actor.train_on_batch(a_train, b_train)
        c_train, d_train = self.calculate_target_critic(mini_batch)
        c_loss = self.target_critic.train_on_batch(c_train, d_train)
        # self.target_actor.compile(loss='mse', optimizer=Adam(lr=0.001))
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        grads_q = tape.gradient(c_loss, self.actor.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.critic_optimizer.step()
        self.updateTargetActor()
        self.updateTargetCritic()
        return a_loss, c_loss

    def updateTargetActor(self):

        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.target_actor.set_weights(self.target_actor.get_weights())
            print("target updated")
        self.total_steps += 1

    def updateTargetCritic(self):
        if self.total_steps >= 2*self.batch_size and self.total_steps % self.target_update_interval == 0:
            self.target.set_weights(self.model.get_weights())
            print("target updated")
        self.total_steps += 1


    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros(vector_length)
        if hot_index != -1:
            output[hot_index] = 1
        return output

    def calculate_target_actor(self, mini_batch):
        X_train = [np.zeros((self.batch_size,) + self.state_size)]
        next_states = np.zeros((self.batch_size,) + self.state_size)

        for index_rep, transition in enumerate(mini_batch):
            X_train[0][index_rep] = transition.state
            next_states[index_rep] = transition.next_state

        Y_train = np.zeros((self.batch_size,))
        anext = self.target_actor.predict([next_states, self.allBatchMask])
        anext = np.amax(anext, 1)

        for index_rep, transition in enumerate(mini_batch):
            if transition.is_done:
                Y_train[index_rep][transition.action] = transition.reward
            else:
                Y_train[index_rep][transition.action] = transition.reward + anext[index_rep] * self.gamma
        return X_train, Y_train

    def calculate_target_critic(self, mini_batch):

        with tf.GradientTape() as tape:
            X_train = [np.zeros((self.batch_size,) + self.state_size),
                       np.zeros((self.batch_size,) + (self.action_size,))]
            next_states = np.zeros((self.batch_size,) + self.state_size)

            for index_rep, transition in enumerate(mini_batch):
                X_train[0][index_rep] = transition.state
                X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
                next_states[index_rep] = transition.next_state

            Y_train = np.zeros((self.batch_size,) + (self.action_size,))
            qnext = self.target_critic.predict([next_states, self.allBatchMask])
            qnext = np.amax(qnext, 1)

            for index_rep, transition in enumerate(mini_batch):
                if transition.is_done:
                    Y_train[index_rep][transition.action] = transition.reward
                else:
                    Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
            return X_train, Y_train

    def predict(self, state, isTarget):

        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        if isTarget:
            result_a = self.target_actor.predict([state, self.allMask])
            result_c = self.target_actor.predict([state, self.allMask])
        else:
            result_a = self.actor.predict([state, self.allMask])
            result_c = self.critic.predict([state, self.allMask])
        return result_a, result_c

    def update_target_critic(self):


    def update_target_actor(self):










