import gymnasium as gym
import numpy as np
import torch as tc
import datetime
import os
import copy
import pong_gym

from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from pong_gym.wrappers import NormalizeObservationPong

from pong_pz import pong_v0
from pong_pz.wrappers import normalize_observation_pong

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net.nn import NNActorCriticDiscrete
from ppo_algorithm.agent import PPOAgent

from torch.utils.tensorboard import SummaryWriter

# ========================================
# ============ HYPERPARAMETERS ===========
# ========================================

TARGET_TOTAL_STEPS = 5000000
N_ENVS = 4
N_STEPS = 1024
LEARNING_RATE = 10**-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 64
N_EPOCHS = 8
CLIP_RANGE = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.0
TEST_EVERY_TRAINSTEP = 5
DEVICE = tc.device("cpu")
LOGGING = False
DEBUG_TEST = True

# ========================================
# ================= TRAIN ================
# ========================================

def train():
    training_path = os.path.join("./ppo/", datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    training_model_path = os.path.join(training_path, "models")

    #Create the enviroment.
    env = pong_v0.env()
    env = normalize_observation_pong(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, N_ENVS, num_cpus=0, base_class="gymnasium")

    #Create the agent.
    agent = PPOAgent(NNActorCriticDiscrete(env.observation_space.shape[0], env.action_space.n, 256, 2, 1, 1),
                     Rollout(N_STEPS, N_ENVS * 2, env.observation_space.shape, (), act_dtype=tc.int32, device=DEVICE),
                     LEARNING_RATE,
                     BATCH_SIZE,
                     N_EPOCHS,
                     DEVICE,
                     gamma=GAMMA,
                     gae_coeff=GAE_LAMBDA,
                     clip_range=CLIP_RANGE,
                     value_coeff=VALUE_COEFFICIENT,
                     entr_coeff=ENTROPY_COEFFICIENT)

    #Tensorboard logger.
    if LOGGING:
        summary = SummaryWriter(os.path.join(training_path, "log"))

    #Training phase.
    total_states = 0
    n_trainstep = 1

    os.makedirs(training_model_path)
    print("Training is started")

    obs, infos = env.reset()
    done = np.zeros(N_ENVS * 2, dtype=np.int32)
    states = np.zeros(N_ENVS * 2, dtype=np.int32)
    while total_states <= TARGET_TOTAL_STEPS:
        for _ in range(N_STEPS):
            #Choose action.
            obs[np.arange(1, N_ENVS * 2, step=2, dtype=np.int32)] *= np.array([-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
            action, value, log_prob = agent.choose_action(tc.Tensor(obs).to(DEVICE))

            #Perform action chosen.
            next_obs, reward, terminated, truncation, next_infos = env.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncation)

            total_states += 1
            states += 1

            #Store one step infos into rollout.
            agent.remember(tc.Tensor(obs).to(DEVICE), 
                           action, 
                           log_prob, 
                           tc.Tensor(reward).to(DEVICE), 
                           tc.Tensor(done).to(DEVICE), 
                           value.reshape(-1))

            #Print episode infos.
            for i in range(N_ENVS * 2):
                if next_done[i]:
                    agent_score = infos[i]["score"]
                    agent_touch = infos[i]["ball_touched"]
                    opponent_score = infos[i + 1]["score"]          if i % 2 == 0 else infos[i - 1]["score"]
                    opponent_touch = infos[i + 1]["ball_touched"]   if i % 2 == 0 else infos[i - 1]["ball_touched"]

                    print("- total state = {:>7d}; score = {:2d} - {:2d}; touch = {:>2d} - {:>2d}; states = {:>4d}; agent = {}".format(total_states, agent_score, opponent_score, agent_touch, opponent_touch, states[i], i))

                    states[i] = 0

            #Next observation.
            obs = next_obs
            done = next_done
            infos = next_infos

        #Train step.
        train_infos = agent.train(tc.Tensor(obs).to(DEVICE), tc.Tensor(done).to(DEVICE))

        if LOGGING:
            summary.add_scalar("train/surrogate_loss", train_infos["surrogate_loss"], total_states)
            summary.add_scalar("train/value_loss", train_infos["value_loss"], total_states)
            summary.add_scalar("train/entropy_loss", train_infos["entropy_loss"], total_states)
            summary.add_scalar("train/total_loss", train_infos["total_loss"], total_states)
            summary.add_scalar("train/clip_fraction", train_infos["clip_fraction"], total_states)
            summary.add_scalar("train/approximate_kl_div", train_infos["approx_kl"], total_states)
            summary.add_scalar("train/explained_variance", train_infos["explained_variance"], total_states)

        if n_trainstep % TEST_EVERY_TRAINSTEP == 0 and test_policy(copy.deepcopy(agent.model)):
            tc.save(agent.model.state_dict(), os.path.join(training_model_path, f"model_{total_states}.pth"))
        n_trainstep += 1

    env.close()
    if LOGGING:
        summary.close() 
    agent.save_model(training_model_path)

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
        action, value, _, _ = model.action_and_value(tc.Tensor(obs).unsqueeze(0))

        #Perform action chosen.
        next_obs, _, terminated, truncation, info = env.step(action.cpu().item())
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

    #Create the PPO model.
    model = NNActorCriticDiscrete(env.observation_space.shape[0], env.action_space.n, 256, 2, 1, 1)
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
                action, value, _, _ = model.action_and_value(tc.Tensor(obs).unsqueeze(0))
                action = action.cpu().item()

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
