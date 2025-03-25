import multiprocessing as mp
import numpy as np
import gymnasium as gym
import torch as tc
import datetime
import pong_gym
import os

from collections import deque

from pong_gym.wrappers import NormalizeObservationPong

from pong_pz import pong_v0
from pong_pz.wrappers import normalize_observation_pong, point_reward

from agent import Agent

def test_agent(test_models, is_training, n_test, path_training):
    """Test training agent's models.
    
    Parameters
    --------------------
    test_models: multiprocessing.Queue
        queue of agent models which need to be tested
    
    is_training: multiprocessing.Value
        whether the agent is still being trained 
        
    n_test: int
        number of episodes each agent's model is tested
        
    path_training: str
        a path training"""
    
    #Create envoriment for testing.
    env = gym.make("pong_gym/Pong-v0")
    env = NormalizeObservationPong(env)

    #Main loop.
    best_model = None

    while is_training.value:
        #A model is tested.
        while not test_models.empty():
            a_model, epis = test_models.get()
            test_agent = Agent(a_model)

            #Testing phase of a policy.
            scores = []
            for _ in range(n_test):
                obs, info = env.reset()
                episode_done = False

                while not episode_done:
                    #A action is chosen.
                    action = test_agent.choose_action(obs)

                    #The action chosen is performed.
                    next_obs, _, terminated, truncated, info = env.step(action)
                    episode_done = terminated or truncated

                    #Next observation.
                    obs = next_obs

                scores.append(info["agent_score"] - info["bot_score"])

            #Print test stats.
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            print("-----------------------------------------------------------------")
            print("TEST MODEL EP. {} STATS: avg score = {:.2f}; std score = {:.2f}".format(epis, mean_score, std_score))
            print("-----------------------------------------------------------------")

            #Update best policy.
            if best_model is None or mean_score >= best_model[1]:
                best_model = [a_model, mean_score]
                tc.save(a_model.state_dict(), os.path.join(path_training, f"model_{epis}.pth"))

    env.close()


def train(agent, last_n_policies, copy_policy_games, change_policy_games, last_policy_prob, a_model_games, n_test_per_policy, use_pr_wrap=False):
    """Train an agent against itself to play Pong.
    
    Parameters
    --------------------
    agent: AgentTrain
        an agent to train
        
    last_n_policies: int
        stack the most recent N agent policies

    copy_policy_games: int
        current agent's policy is copied every 'copy_policy_games' episodes    

    change_policy_games: int
        opponenent's policy is changed every 'change_policy_games' episodes

    last_policy_prob: float
        probabilty of a training agent which plays against itself with last policy stored

    a_model_games: int
        current agent policy is tested every 'a_model_games' episodes

    n_test_per_policy: int
        number of test done for each policy

    use_pr_wrap: bool, optional
        whether or not to use point reward wrapper
    """

    assert last_n_policies > 0
    assert copy_policy_games > 0
    assert change_policy_games > 0
    assert last_policy_prob >= 0 and last_policy_prob <= 1.0
    assert a_model_games > 0
    assert n_test_per_policy > 0

    rng = np.random.default_rng()
    
    #Create path training.
    path_training = "./train_"+datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.makedirs(path_training)

    #Create the enviroment.
    env = pong_v0.env()
    env = normalize_observation_pong(env)
    if use_pr_wrap:     env = point_reward(env)

    #Player names.
    agent_name = env.possible_agents[0]
    opponent_name = env.possible_agents[1]

    #Player elos.
    agent_elo = 500
    opponent_elo = 500

    #Create new opponent agent as skilled as training agent.
    opponent = Agent(agent.get_model())

    #Stack the most recent agent policies.
    last_policies = deque(maxlen=last_n_policies)
    last_policies.append([Agent(agent.get_model()), agent_elo])

    #Multiprocessing stuff.
    test_models = mp.Queue()
    is_training = mp.Value("i", 1)
    test_agent_process = mp.Process(target=test_agent, args=(test_models, is_training, n_test_per_policy, path_training))

    #Training phase.
    test_agent_process.start()

    while agent.needs_to_learn():
        #New episode.
        observations, infos = env.reset()
        agent.states = 0
        episode_done = False

        #Opponent's policy is changed if needed
        if agent.episode % change_policy_games == 0:
            if rng.uniform() <= last_policy_prob:
                idx = -1
            else:
                idx = rng.integers(0, len(last_policies))
            opponent = last_policies[idx][0]
            opponent_elo = last_policies[idx][1]

        #Episode.
        while not episode_done:
            #The players choose an action to perform.
            agent_action    = agent.choose_action(observations[agent_name])
            opponent_action = opponent.choose_action(observations[opponent_name] * np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)) #This becomes as opponent agent would play with the left paddle.

            #The actions chosen are performed
            next_observations, rewards, termimations, truncations, infos = env.step({agent_name: agent_action, opponent_name: opponent_action})
            episode_done = (termimations[agent_name] or truncations[agent_name]) or (termimations[opponent_name] or truncations[opponent_name])

            #A train step is done.
            agent.train(observations[agent_name], 
                        next_observations[agent_name],
                        agent_action, 
                        rewards[agent_name], 
                        termimations[agent_name], 
                        truncations[agent_name],
                        infos[agent_name])
            
            #----------
            agent.states += 1
            agent.total_states += 1
            observations = next_observations

        #agent elo is updated.
        agent_expectation = 1 / (1 + 10**((opponent_elo - agent_elo) / 400))
        agent_point = 1.0 if infos[agent_name]["score"] > infos[opponent_name]["score"] else 0.0
        agent_elo += 16 * (agent_point - agent_expectation)

        #Current agent policy is copied.
        if agent.episode % copy_policy_games == 0:
            last_policies.append([Agent(agent.get_model()), agent_elo])

        #Current agent policy is delegated on another process which will test it.
        if agent.episode % a_model_games == 0:
            test_models.put((agent.get_model(), agent.episode))

        #Print current episode stats.
        print("- Episode {:>3d}: {:2d} - {:2d}; touch = {:>2d} - {:>2d}; ELO = {:>4.1f} - {:>4.1f}; total states = {:>7d}; states = {:>4d}; e = {:.2f}".format(
                        agent.episode,
                        infos[agent_name]["score"],
                        infos[opponent_name]["score"],
                        infos[agent_name]["ball_touched"],
                        infos[opponent_name]["ball_touched"],
                        agent_elo,
                        opponent_elo,
                        agent.total_states,
                        agent.states,
                        agent.epsilon))
        
        #Next episode.
        agent.episode += 1

    #Close enviroment.
    env.close()

    #Save model.
    tc.save(agent.get_model().state_dict(), os.path.join(path_training, "model.pth"))

    #Wait the other process which is testing agent policies left.
    is_training.value = 0
    test_agent_process.join()