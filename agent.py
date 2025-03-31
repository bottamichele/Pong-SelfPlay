import torch as tc

from abc import ABC, abstractmethod

from dqn_algorithms.policy import greedy_policy

# ========================================
# ============= CLASS Agent ==============
# ========================================

class Agent:
    """An DQN agent which plays Pong."""

    def __init__(self, model, device=tc.device("cpu")):
        """Create new Pong's agent.
        
        Parameter
        --------------------
        model: DQN-like
            a DQN neural network"""

        self._model = model.to(device=device)
        self._device = device

        self._model.eval()

    def choose_action(self, obs):
        """Choose an action from an observation.
        
        Parameter
        --------------------
        obs: numpy.ndarray
            an observation"""
        
        return greedy_policy(self._model, tc.from_numpy(obs).to(dtype=tc.float32, device=self._device))

# ========================================
# ========= CLASS Training Agent =========
# ========================================

class AgentTrain(ABC):
    """A DQN agent which learns to play Pong."""

    episode = 1
    states = 0
    total_states = 0
    device = None

    @abstractmethod
    def train(self, observation, next_observation, action, reward, termination, truncation, info):
        """Do a train step."""

        pass

    @abstractmethod
    def needs_to_learn(self):
        """Check if agent still needs to learn to play.
        
        Return
        --------------------
        check: bool
            True if agent still needs to learn, False otherwise"""
        
        pass

    @abstractmethod
    def get_model(self):
        """Return its current model.
        
        Return
        --------------------
        model: DQN-like
            its current copy model"""
        
        pass