import joblib
import tensorflow as tf
import numpy as np

# from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, multiply

from Agents import modelFreeAgent, deepQ
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

# Discount factor for future rewards
gamma = 0.99

tf.keras.backend.set_floatx('float64')

class OUNoise():
    """Generate Noise using Ornstein Uhlenbeck Process"""
    def __init__(self, act_size, mu=0, theta=0.15, sigma=0.3, *args):

        # super().__init__(*args)
        self.action_dimension = act_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.state = np.ones(self.action_dimension) * self.mu

        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu


    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.randn(len(x))
        self.state = x + dx
        return self.state


class Ag(modelFreeAgent.ModelFreeAgent):
    displayName = 'DDPG'

    newParameters = [modelFreeAgent.ModelFreeAgent.Parameter('Batch Size', 1, 256, 1, 32, True, True,
                                                             "The number of transitions to consider simultaneously when updating the agent"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Memory Size', 1, 655360, 1, 1000, True, True,
                                                             "The maximum number of timestep transitions to keep stored"),
                     modelFreeAgent.ModelFreeAgent.Parameter('Target Update Interval', 1, 100000, 1, 200, True, True,
                                                             "The distance in timesteps between target model updates"),
                     deepQ.DeepQ.Parameter('History Length', 0, 20, 1, 10, True, True, "The number of recent timesteps to use as input")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(Ag.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval, self.historylength = [int(arg) for arg in args[-paramLen:]]
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))
        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.ou_noise = OUNoise(self.action_size)

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        """self.states = np.zeros(self.state_size)
        self.actions = np.zeros(self.action_size)
        self.rewards = np.zeros(1)
        self.next_states = np.zeros(self.state_size)
        self.dones = np.zeros(1)"""

        self.total_steps = 0
        self.allMask = np.full((1, self.action_size), 1)
        self.allBatchMask = np.full((self.batch_size, self.action_size), 1)

        # Used to update target networks
        self.tau = 0.005



    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        input_shape = self.state_size
        inputA = Input(shape=input_shape)
        x = Flatten()(inputA)
        x = Dense(24, activation='relu')(x)  # fully connected
        x = Dense(24, activation='relu')(x)
        # x = Dense(1, activation='tanh')(x)
        x = Dense(1, activation='linear', kernel_initializer=last_init)(x)
        # outputs = multiply([x, inputA])
        model = Model(inputs=inputA, outputs=x)
        model.compile(loss='mse', optimizer=self.actor_optimizer)


        # inputs = Input(shape=self.state_size)
        # out = Flatten()(inputs)
        # out = Dense(24, input_dim=self.state_size, activation="relu")(out)
        """ 
        out = Dense(24, activation="relu")(out)
        # outputs = Dense(self.state_size, activation='linear')(out)
        out = Dense(1, activation="linear", kernel_initializer=last_init)(out)
        outputs = multiply([out, inputs])
        # upper_bound = 2.0
        # Our upper bound is 2.0 for Pendulum.
        # outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss='mse', optimizer=self.actor_optimizer)
         model  """
        return model

    def get_critic(self):

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

    def choose_action(self, state1):
        #state = state1.eval(session=tf.compat.v1.Session())
        state = np.asarray(state1)
        state = np.reshape(*state, [1, self.state_size])
        x = self.actor_model(state)
        sampled_actions = tf.squeeze(x)
        # sampled_actions = self.actor_model(state)
        noise = OUNoise(self.action_size)
        # noise = tf.squeeze(noise)
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, -0.3, 0.3)

        return [np.squeeze(legal_action)]

    def sample(self):
        return self.memory.sample(self.batch_size)

    def addToMemory(self, state, action, reward, new_state, done):
        self.memory.append_frame(TransitionFrame(state, action, reward, new_state, done))

    def remember(self, state, action, reward, new_state, done=False):

        self.addToMemory(state, action, reward, new_state, done)
        loss = 0

        if len(self.memory) < 2*self.batch_size:
            return loss
        mini_batch = self.sample()
        self.learn(mini_batch)

    """def predict(self, state, isTarget):

        shape = (1,) + self.state_size
        state = np.reshape(state, shape)
        state = tf.cast(state, dtype=tf.float32)
        if isTarget:
            result = self.target_critic.predict([state, self.allMask])
        else:
            result = self.critic_model.predict([state, self.allMask])
        return result"""

    def create_one_hot(self, vector_length, hot_index):
        output = np.zeros(vector_length)
        if hot_index != -1:
            output[hot_index] = 1
        return output



    def learn(self, mini_batch):

        X_train = [np.zeros((self.batch_size,) + self.state_size), np.zeros((self.batch_size,) + (self.action_size,))]
        # X_train = tf.convert_to_tensor(X_train)
        next_states = (np.zeros((self.batch_size,) + self.state_size))

        # actions = np.zeros((self.batch_size,) + self.action_size)

        for index_rep, transition in enumerate(mini_batch):
            X_train[0][index_rep] = transition.state
            X_train[1][index_rep] = self.create_one_hot(self.action_size, transition.action)
            # X_train = tf.convert_to_tensor(X_train)
            #
            # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            next_states[index_rep] = transition.next_state
            # next_states = tf.convert_to_tensor(next_states)
            #actions[index_rep] = transition.actions
            # shape = (1,) + self.state_size
            # next_states = next_states, shape

        Y_train = np.zeros((self.batch_size,) + (self.action_size,))
        # Y_train = tf.convert_to_tensor(Y_train)
        # self.allBatchMask = tf.convert_to_tensor(self.allBatchMask)
        # next_states = tf.convert_to_tensor(next_states)

        # with tf.GradientTape() as tape:
        actions = self.actor_model(X_train, training=True)
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.critic_model([X_train, actions])
            q_values = tf.squeeze(q_values)
        # return tape.gradient(q_values, actions)

            # grads = self.q_grads(X_train, actions)
        # grads = np.array(s_grads).reshape((-1, self.action_size))
            # critic_value = self.critic_model([X_train, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
        with tf.GradientTape() as tape:
            grads = tape.gradient(q_values, actions)
        # grads = -grads
        # actor_loss = -tf.math.reduce_mean(q_values)
            # actor_loss = self.actor_model.train_on_batch(X_train)
            # print(actor_loss)
            # print(actor_loss.dtype)
            # actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        with tf.GradientTape() as tape:
            # grads = tape.gradient(q_values, actions)
            actor_grad = tape.gradient(self.actor_model(X_train), self.actor_model.trainable_variables, grads)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))


        with tf.GradientTape() as tape:
            print(next_states.dtype)
            print(self.allBatchMask.dtype)
            self.allBatchMask = self.allBatchMask.astype(float)
            # bt_mask = tf.convert_to_tensor(self.allBatchMask)
            qnext = self.target_critic([next_states, self.allBatchMask])

            qnext = np.amax(qnext, 1)

            for index_rep, transition in enumerate(mini_batch):
                if transition.is_done:
                    Y_train[index_rep][transition.action] = transition.reward
                else:
                    Y_train[index_rep][transition.action] = transition.reward + qnext[index_rep] * self.gamma
            # X_train = tf.ragged.constant(X_train)
            # print(X_train.dtype)
            # Y_train = tf.convert_to_tensor(Y_train)
            # Y_train = tf.ragged.constant(Y_train)
            print(Y_train.dtype)
            critic_value = self.critic_model(X_train, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(Y_train - critic_value))
            # critic_loss = tf.reduce_mean((tf.abs(tf.subtract(X))))
            # critic_loss = self.critic_model.train_on_batch(X_train, Y_train)
        # critic_loss = tf.ragged.constant(critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        print(critic_loss.dtype)

        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

    def q_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.critic_model([states, actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

        """for x in range(32):
            states = batch[x].state
            actions = batch[x].action
            rewards = batch[x].reward
            next_states = batch[x].next_state
            dones = batch[x].is_done"""

        """with tf.GradientTape() as tape:
                target_actions = self.target_actor(next_states, training=True)
                y = rewards + self.gamma * self.target_critic([next_states, target_actions], training=True)
                critic_value = self.critic_model([states, actions], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

            with tf.GradientTape() as tape:
                actions = self.actor_model(states, training=True)
                critic_value = self.critic_model([states, actions], training=True)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))"""



    def save(self, filename):
        mem = self.critic_model.get_weights()
        joblib.dump((Ag.displayName, mem), filename)

        mem1 = self.actor_model.get_weights()
        joblib.dump((Ag.displayName, mem1), filename)

    def load(self, filename):
        name, mem, mem1 = joblib.load(filename)
        if name != Ag.displayName:
            print('load failed')
        else:
            self.critic_model.set_weights(mem)
            self.target_critic.set_weights(mem)
            self.actor_model.set_weights(mem1)
            self.target_actor.set_weights(mem1)

    def memsave(self):
        x = self.critic_model.get_weights()
        y = self.actor_model.get_weights()
        return x, y

    def memload(self, mem, mem1):
        self.critic_model.set_weights(mem)
        self.target_critic.set_weights(mem)
        self.actor_model.set_weights(mem1)
        self.target_actor.set_weights(mem1)

    def reset(self):
        pass

