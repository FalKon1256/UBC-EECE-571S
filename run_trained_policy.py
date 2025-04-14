import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common import utils
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from utils.custom_evaluation import custom_evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import FeedForward32Policy
from imitation.util.util import make_vec_env


class ActionFloat32Wrapper(gym.ActionWrapper):
    """Force changing the type of the action to a list of Python float to fix the Box2D type error."""
    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return [float(a) for a in action]


# Define the control parameters for experiments. 
IS_CONTINUOUS = False
SEED = 1000                 # Train:0 / Eval:1000
EVAL_EPISODES = 5           # 20

# Evaluation mode selection (please set this to FALSE if not using the best cnn models to test). 
SEE_BEST_MODEL_ACTIONS = True

# Policy selection. 
ppo_policy = True
bc_policy = False
dagger_policy = False

# Model selection (only changes for BC or DAgger). 
mlp_model = False
cnn_model = False



# Define the Car Racing environment for evaluation. 
rng = np.random.default_rng(SEED)

if IS_CONTINUOUS:
    # Continuous action space. 
    env = make_vec_env(
        "CarRacing-v2", 
        rng=rng, 
        n_envs=1, 
        post_wrappers=[
            lambda env, _: ActionFloat32Wrapper(env), 
            lambda env, _: RolloutInfoWrapper(env), 
        ], 
        env_make_kwargs={
            "render_mode": "human", 
            "continuous": True, 
        }, 
    )
else:
    # Discrete action space. 
    env = make_vec_env(
        "CarRacing-v2", 
        rng=rng, 
        n_envs=1, 
        post_wrappers=[
            lambda env, _: RolloutInfoWrapper(env), 
        ], 
        env_make_kwargs={
            "render_mode": "human", 
            "continuous": False, 
        }, 
    )
env = VecTransposeImage(env)


###################################### PPO Model #####################################

if ppo_policy:
    if IS_CONTINUOUS:
        model_path = "models/ppo/continuous/ppo_timesteps_1000000/checkpoints/ppo_continuous_1000000_steps.zip"
    else:
        model_path = "models/ppo/discrete/ppo_timesteps_1000000/checkpoints/ppo_discrete_1000000_steps.zip"
    model = PPO.load(model_path)

################################# BC or DAgger Model #################################

if dagger_policy or bc_policy:

    if bc_policy:
        if IS_CONTINUOUS:
            if mlp_model:
                model_path = "models/bc/bc-mlp-continuous-traj_20.zip"  # for mlp
            else:
                model_path = "models/bc/bc-continuous-traj_20.zip"      # for action test (better model) 
                #model_path = "models/bc/bc-continuous-traj_1.zip"      # for custom change
        else:
            if mlp_model:
                model_path = "models/bc/bc-mlp-discrete-traj_20.zip"    # for mlp
            else:
                model_path = "models/bc/bc-discrete-traj_20.zip"        # for action test (better model)
                #model_path = "models/bc/bc-discrete-traj_1.zip"        # for custom change

    elif dagger_policy:
        if IS_CONTINUOUS:
            if mlp_model:
                model_path = "models/dagger/dagger-mlp-continuous-totalsteps_10000.zip" # for mlp
            else:
                model_path = "models/dagger/dagger-continuous-totalsteps_20000.zip"     # for action test (better model)
                #model_path = "models/dagger/dagger-continuous-totalsteps_1000.zip"     # for custom change
        else:
            if mlp_model:
                model_path = "models/dagger/dagger-mlp-discrete-totalsteps_10000.zip"   # for mlp
            else:
                model_path = "models/dagger/dagger-discrete-totalsteps_20000.zip"       # for action test (better model)
                #model_path = "models/dagger/dagger-discrete-totalsteps_1000.zip"       # for custom change
    else:
        raise ValueError("Invalid policy selected, please check again!")

    # Create the policy instance. 
    if cnn_model:
        # CNN case. 
        model = CnnPolicy(
            observation_space=env.observation_space, 
            action_space=env.action_space, 
            lr_schedule=lambda _: 3e-4, 
        )
    elif mlp_model:
        # MLP case. 
        model = FeedForward32Policy(
            observation_space=env.observation_space, 
            action_space=env.action_space, 
            lr_schedule=lambda _: 3e-4, 
        )
    else:
        raise ValueError("Invalid model selected, please check again!")

    # Get an instance of the loaded policy. 
    loaded = torch.load(model_path, map_location=utils.get_device("auto"), weights_only=False)

    # Extract the state_dict (saved weights) from the instance. 
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
    else:
        state_dict = loaded

    # Load the saved weights to the policy instance. 
    model.load_state_dict(state_dict)

######################################################################################

# Evaluate the loaded policy after training. 
print(f"Start the evaluation of the loaded policy from the path: {model_path}, ...")

# Choose to use which evaluation mode. 
if SEE_BEST_MODEL_ACTIONS:
    # Special mode, will be able to see the actions information. 
    print("[Notice] Chose to use CUSTOM evaluation model, will see action per step for best models!")

    # MLP strutures are not available for special mode. 
    if mlp_model:
        raise ValueError("Mlp models are invalid for this evaluation mode, please use normal mode!")

    # Name the model for the evaluation function. 
    if ppo_policy:
        model_name = "ppo_trained_100k"
    elif bc_policy:
        model_name = "bc_traj_20"
    elif dagger_policy:
        model_name = "dagger_totalsteps_20000"
    else:
        raise ValueError("Invalid policy selected, please check again!")

    # Start the evaluation. 
    episode_rewards, episode_lengths = custom_evaluate_policy(
                                            model=model, 
                                            env=env, 
                                            n_eval_episodes=EVAL_EPISODES,
                                            deterministic=True,
                                            return_episode_rewards=True,
                                            model_name=model_name,
                                            is_continuous=IS_CONTINUOUS
    )
else:
    # Normal mode, will not be able to see the actions information. 
    print("[Notice] Chose to use NORMAL evaluatation mode!")

    # Start the evaluation. 
    episode_rewards, episode_lengths = evaluate_policy(
                                            model=model, 
                                            env=env, 
                                            n_eval_episodes=EVAL_EPISODES,
                                            deterministic=True,
                                            return_episode_rewards=True,
    )

# Compute the mean and standard deviation of the reward. 
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)

# Compute the mean and standard deviation of the episode length. 
mean_length = np.mean(episode_lengths)
std_length = np.std(episode_lengths)

# Show the results. 
print(f"Average reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Average episode length: {mean_length:.2f} +/- {std_length:.2f}")
print(f"Reward of each episode: {episode_rewards}")
print(f"Episode Length of each episode: {episode_lengths}")
