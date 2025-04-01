from agent import Agent
from self_play import train
from run_agent import run_agent

from dqn_agent import DQNAgentTrain, load_dueling_dqn_model, load_dqn_model

USE_DUELING_DQN = True
TRAIN = False

if __name__ == "__main__":
    if USE_DUELING_DQN and TRAIN:
        train(DQNAgentTrain(5_000_000, 750000, 64, 5000, eps_decay=3.96*10**-6), 6, 20, 10, 0.5, 20, 10)
    elif USE_DUELING_DQN and not TRAIN:
        model = load_dueling_dqn_model("./models/dueling_dqn.pth")
        agent = Agent(model)
        run_agent(agent, 50)