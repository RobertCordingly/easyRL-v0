import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from Agents import modelFreeAgent
from Agents.Collections import ExperienceReplay
from Agents.Collections.TransitionFrame import TransitionFrame

# Discount factor for future rewards
gamma = 0.99


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
                                                             "The distance in timesteps between target model updates")]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):
        paramLen = len(Ag.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval = [int(arg) for arg in args[-paramLen:]]
        empty_state = self.get_empty_state()
        self.memory = ExperienceReplay.ReplayBuffer(self, self.memory_size, TransitionFrame(empty_state, -1, 0, empty_state, False))

        self.ou_noise = OUNoise(self.action_size)

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001


        # Used to update target networks
        self.tau = 0.005

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_size,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        # outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.state_size)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for given state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def choose_action(self, state):
        # torch.from_numpy(state).float().to(device)
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = OUNoise(self.action_size)
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

        if len(self.memory) > self.batch_size:
            return loss
        mini_batch = self.sample()
        self.learn(mini_batch, self.gamma)

    def learn(self, mini_batch, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = mini_batch

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states, training=True)
            y = rewards + gamma * self.target_critic([next_states, target_actions], training=True)
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
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

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



        """

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_model(states, actions)
        critic_loss = tensorflow.reduce_mean((np.subtract(Q_targets, Q_expected)) ** 2)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.remember()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_model(states)
        actor_loss = -self.critic_model(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.remember()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_model, self.target_critic, self.tau)
        self.soft_update(self.actor_model, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """"""Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter """"""
        """
        """for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    """