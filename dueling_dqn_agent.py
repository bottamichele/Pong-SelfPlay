import torch as tc
import copy

from torch.nn.functional import mse_loss
from torch.optim import Adam

from dqn_algorithms.policy import e_greedy_policy
from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.memory import PropPriorMemory
from dqn_algorithms.q_methods import compute_q_ddqn

from agent import AgentTrain

class DuelingDQNAgentTrain(AgentTrain):
    def __init__(self, target_total_step, mem_size, batch_size, update_target_steps, lr=10**-4, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=9.9*10**-6, device=tc.device("cuda" if tc.cuda.is_available() else "cpu")):
        """Create new training Dueling DQN agent.
        
        Parameters
        --------------------
        target_total_step: int
            number total step for training

        mem_size: int
            memory replay size

        batch_size: int
            batch size

        update_target_steps: int
            target network is upadeted every 'update_target_steps'

        lr: float, optional
            learning rate

        gamma: float, optional
            discount factor

        eps_init: float, optional
            initial epsilon value
            
        eps_min: float, optional
            minimum epsilon value
            
        eps_decay: float, optional
            epsilon decay
            
        device: torch.device
            device this agent is trained on"""

        assert target_total_step > 0
        assert mem_size > 0
        assert batch_size > 0
        assert update_target_steps > 0
        assert lr > 0
        assert gamma > 0.0 and gamma < 1.0
        assert eps_init <= 1.0 and eps_init >= eps_min
        assert eps_min <= eps_init and eps_min >= 0.0
        assert eps_decay > 0

        #Attributes.
        self.device = device
        self.target_total_step = target_total_step
        self.total_states = 0
        self.batch_size = batch_size
        self.memory = PropPriorMemory(mem_size, 12, device=self.device)
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.beta_decay = (1.0 - self.memory.beta) / target_total_step
        self.gamma = gamma
        self.update_target_steps = update_target_steps

        #Model.
        self.model = DuelingDQN(12, 3, 256, 1, 1, 1).to(self.device)
        self.target_model = DuelingDQN(12, 3, 256, 1, 1, 1).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.update_target_model()
        self.model.train()
        self.target_model.eval()


    def choose_action(self, obs):
        assert obs.shape == (12,), "Dueling DQN agent allows only observations with shape (12,)."

        x = tc.from_numpy(obs)
        return e_greedy_policy(self.model, self.epsilon, tc.unsqueeze(x, 0).to(self.device))
    
    def train(self, observation, next_observation, action, reward, termination, truncation, info):
        observation = tc.from_numpy(observation).to(self.device)
        next_observation = tc.from_numpy(next_observation).to(self.device)

        self.memory.store_transiction(observation, action, reward, next_observation, termination or truncation)

        if len(self.memory) >= self.batch_size:
            #Sample mini-batch.
            obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, weight_b = self.memory.sample_batch(self.batch_size)
            # obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self.memory.sample_batch(self.batch_size)

            #Convertion.
            action_b = action_b.to(dtype=tc.int32)
            next_obs_done_b = next_obs_done_b.to(dtype=tc.int32)

            #Computet q values.
            q, q_target = compute_q_ddqn(self.model, self.target_model, self.gamma, obs_b, action_b, reward_b, next_obs_b, next_obs_done_b)

            #Compute loss and gradient.
            loss = tc.mean((q_target - q).pow(2) * weight_b)
            # loss = mse_loss(q, q_target).to(self.device)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Update priorities.
            td_errors = tc.clamp(q_target - q, -1.0, 1.0)
            self.memory.update_priorities(td_errors)

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        self.memory.beta += self.beta_decay

        if self.total_states % self.update_target_steps == 0:
            self.update_target_model()
    
    def update_target_model(self):
        """Update target network's weights."""

        self.target_model.load_state_dict(self.model.state_dict())

    def get_model(self):
        return copy.deepcopy(self.model)
    
    def needs_to_learn(self):
        return self.episode <= self.target_total_step