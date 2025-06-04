import gymnasium as gym
import numpy as np
import torch as tc
import datetime
import os
import copy
import pong_gym

from pong_gym.wrappers import NormalizeObservationPong

from pong_pz import pong_v0
from pong_pz.wrappers import normalize_observation_pong

from dqn_algorithms.agent import DQNAgent
from dqn_algorithms.nn import DuelingDQN
from dqn_algorithms.policy import greedy_policy

from collections import deque

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 5000000
MEMORY_SIZE = 500000
UPDATE_TARGET_STEP = 5000
GAMMA = 0.99
LEARNING_RATE = 10**-4
BATCH_SIZE = 64
EPSILON_INIT = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 3.96 * 10**-6
N_LATEST_POLICY = 10
CHANGE_POLICY_GAME = 5
COPY_POLICY_GAME = 10
LATEST_POLICY_PROB = 0.5
DEVICE = tc.device("cpu")
DEBUG_TEST = True

# ========================================
# ================= TRAIN ================
# ========================================

def train():
    training_path = os.path.join("./dueling_dqn/", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

    #Create the enviroment.
    env = pong_v0.env()
    env = normalize_observation_pong(env)
    agent_name = env.possible_agents[0]
    opponent_name = env.possible_agents[1]

    #Create the agent.
    agent = DQNAgent(DuelingDQN(env.observation_space(agent_name).shape[0], env.action_space(agent_name).n, 256, 2, 1, 1),
                     {"type": "uniform_memory", "mem_size": MEMORY_SIZE, "obs_size": env.observation_space(agent_name).shape[0]},
                     UPDATE_TARGET_STEP,
                     BATCH_SIZE,
                     LEARNING_RATE,
                     GAMMA,
                     EPSILON_INIT,
                     EPSILON_END,
                     EPSILON_DECAY,
                     DEVICE)
    
    #Stack of the n latest copied policies.
    latest_policies = deque(maxlen=N_LATEST_POLICY)
    latest_policies.append(copy.deepcopy(agent.model).to(DEVICE))
    
    #Training phase.
    rng = np.random.default_rng()
    total_states = 0
    episode = 1

    os.makedirs(training_path)
    print("Training is started")

    opponent = latest_policies[-1]
    opponent.eval()

    while total_states <= TARGET_TOTAL_STEPS:
        #The opponent's policy is changed.
        if episode % CHANGE_POLICY_GAME == 0:
            if rng.uniform() <= LATEST_POLICY_PROB:
                idx = -1
            else:
                idx = rng.integers(0, len(latest_policies))
            opponent = latest_policies[idx]
            opponent.eval()

        #Episode.
        observations, infos = env.reset()
        states = 0
        episode_done = False

        while not episode_done:
            #The players choose an action to perform.
            agent_action    = agent.choose_action(tc.Tensor(observations[agent_name]).to(DEVICE))
            opponent_action = greedy_policy(opponent, tc.Tensor(observations[opponent_name] * np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)).to(DEVICE)) #The opponent's observation becomes as the left paddle's.

            #The actions chosen are performed
            next_observations, rewards, termimations, truncations, infos = env.step({agent_name: agent_action, opponent_name: opponent_action})
            episode_done = (termimations[agent_name] or truncations[agent_name]) or (termimations[opponent_name] or truncations[opponent_name])

            #Store the current transiction.
            agent.remember(tc.Tensor(observations[agent_name]).to(DEVICE), agent_action, rewards[agent_name], tc.Tensor(next_observations[agent_name]).to(DEVICE), episode_done)

            #Train step.
            agent.train()

            #Update target model.
            if total_states % agent.update_target_step == 0:
                agent.update_target_model()
            
            #Next observation.
            states += 1
            total_states += 1
            observations = next_observations

        #Current agent's policy is copied.
        if episode % COPY_POLICY_GAME == 0:
            latest_policies.append(copy.deepcopy(agent.model).to(DEVICE))

        #Print the current episode stats.
        print("- Episode {:>3d}: {:2d} - {:2d}; touch = {:>2d} - {:>2d}; total states = {:>7d}; states = {:>4d}; epsilon = {:.2f}".format(
                        episode,
                        infos[agent_name]["score"],
                        infos[opponent_name]["score"],
                        infos[agent_name]["ball_touched"],
                        infos[opponent_name]["ball_touched"],
                        total_states,
                        states,
                        agent.epsilon))
        
        if total_states >= 1200000 and test_policy(copy.deepcopy(agent.model)):
            tc.save(agent.model.state_dict(), os.path.join(training_path, f"model_{total_states}.pth"))
        
        #Next episode.
        episode += 1

    agent.save_model(training_path)
    env.close()

def test_policy(model):
    #Create the enviroment.
    env = gym.make("pong_gym/Pong-v0")
    env = NormalizeObservationPong(env)

    #Test the policy.
    obs, info = env.reset()
    done = False
    model.eval()

    while not done:
        #Choose action.
        action = greedy_policy(model, tc.Tensor(obs))

        #Perform action chosen.
        next_obs, _, terminated, truncation, info = env.step(action)
        done = terminated or truncation

        #Next observation.
        obs = next_obs

    if DEBUG_TEST:
        print("- TEST POLICY: {:2d} - {:2d}".format(info["agent_score"], info["bot_score"]))

    env.close()

    return info["agent_score"] - info["bot_score"] > 0

# ========================================
# ========== TEST_TRAINED_MODEL ==========
# ========================================

def test_trained_model(trained_model_path, n_runs=20):
    #Create the enviroment.
    env = gym.make("pong_gym/Pong-v0")
    env = NormalizeObservationPong(env)

    #Create the Dueling DQN model.
    model = DuelingDQN(env.observation_space.shape[0], env.action_space.n, 256, 2, 1, 1)
    model.eval()

    files = os.listdir(trained_model_path)
    for file in files:
        #Load a saved model.
        model.load_state_dict(tc.load(os.path.join(trained_model_path, file)))

        #Test the model.
        scores = []
        for _ in range(n_runs):
            obs, info = env.reset()
            done = False

            while not done:
                #Choose action.
                action = greedy_policy(model, tc.Tensor(obs))

                #Perform action chosen.
                next_obs, _, terminated, truncation, info = env.step(action)
                done = terminated or truncation

                #Next observation.
                obs = next_obs
        
            scores.append(info["agent_score"] - info["bot_score"])

        print("- {}: mean score = {:.2f}; std score = {:.2f}; scores = {}".format(file, np.mean(scores), np.std(scores), scores))

    env.close()

# ========================================
# ================= MAIN =================
# ========================================

if __name__ == "__main__":
    train()