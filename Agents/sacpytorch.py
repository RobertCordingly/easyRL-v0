import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Agents import modelFreeAgent

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(self.state_size, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(self.state_size() + self.action_size, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(self.state_size + self.action_size, 24)
        self.linear5 = nn.Linear(24, 24)
        self.linear6 = nn.Linear(24, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(self.state_size, 24)
        self.linear2 = nn.Linear(24, 24)

        self.mean_linear = nn.Linear(24, self.action_size)
        self.log_std_linear = nn.Linear(24, self.action_size)

        self.apply(weights_init_)

        # action rescaling
        if self.action_size is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.tensor((self.action_size.high - self.action_size.low) / 2.)
            self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.logprob_epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(self.state_size, 24)
        self.linear2 = nn.Linear(24, 24)

        self.mean = nn.Linear(24, self.action_size)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

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
                     modelFreeAgent.ModelFreeAgent.Parameter('Entropy', 0.00, 1.00, 0.001, 0.97, True, True,
                                                             "The rate at which target models update")
                     ]
    parameters = modelFreeAgent.ModelFreeAgent.parameters + newParameters

    def __init__(self, *args):

        # Initializing model parameters
        paramLen = len(SAC.newParameters)
        super().__init__(*args[:-paramLen])

        self.batch_size, self.memory_size, self.target_update_interval, self.tau = [int(arg) for arg in args[-paramLen:]]
        self.logprob_epsilon = 1e-6  # For numerical stability when computing tf.log





