from self_play import train
from dueling_dqn_agent import DuelingDQNAgentTrain

if __name__ == "__main__":
    # train(DuelingDQNAgentTrain(5_000_000, 750000, 64, 5000, eps_decay=3.96*10**-6), 6, 20, 10, 0.5, 1, 1)
    train(DuelingDQNAgentTrain(2_000_000, 750000, 64, 5000, eps_decay=3.96*10**-6), 6, 20, 10, 0.8, 20, 15)