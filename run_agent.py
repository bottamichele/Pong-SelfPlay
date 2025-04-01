import gymnasium as gym
import numpy as np
import pong_gym

from pong_gym.wrappers import NormalizeObservationPong

def run_agent(agent, n_test):
    """Test training agent's models.
    
    Parameters
    --------------------
    agent: Agent
        a DQN agent
        
    n_test: int
        number of episodes model is tested"""

    #Create envoriment for testing.
    env = gym.make("pong_gym/Pong-v0", render_mode="human")
    env = NormalizeObservationPong(env)

    #Testing phase.
    scores = []
    for i in range(n_test):
        obs, info = env.reset()
        episode_done = False

        while not episode_done:
            #A action is chosen.
            action = agent.choose_action(obs)

            #The action chosen is performed.
            next_obs, _, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated

            #Next observation.
            obs = next_obs

        scores.append(info["agent_score"] - info["bot_score"])
        print(f"-{i+1}) score = {scores[-1]}")

    #Print stats.
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print("- avg score = {:.2f}; std score = {:.2f}".format(mean_score, std_score))

    env.close()