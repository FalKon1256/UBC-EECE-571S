import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from imitation.util import logger as imit_logger
from utils.custom_flatten_traj import flatten_trajectories_with_transpose


class ActionFloat32Wrapper(gym.ActionWrapper):
    """Force changing the type of the action to a list of Python float to fix the Box2D type error."""
    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return [float(a) for a in action]


# Define the control parameters for experiments. 
TRAJ_NUM = 20           #20
IS_CONTINUOUS = True
IS_MLP = False

SEED = 0                #0
EVAL_EPISODES = 20      #20
TRAIN_EPOCHS = 50       #50


# Set the loading path of the expert model trained by PPO. 
if IS_CONTINUOUS:
    expert_model_path = "models/ppo/continuous/ppo_timesteps_1000000/checkpoints/ppo_continuous_1000000_steps.zip"
else:
    expert_model_path = "models/ppo/discrete/ppo_timesteps_1000000/checkpoints/ppo_discrete_1000000_steps.zip"

# Set the saving path for the log data.  
if IS_CONTINUOUS:
    if IS_MLP:
        log_dir = f"logs/bc_log/continuous/mlp-traj_{TRAJ_NUM}"
    else:
        log_dir = f"logs/bc_log/continuous/traj_{TRAJ_NUM}"
else:
    if IS_MLP:
        log_dir = f"logs/bc_log/discrete/mlp-traj_{TRAJ_NUM}"
    else:
        log_dir = f"logs/bc_log/discrete/traj_{TRAJ_NUM}"

# Set new logger instance for Tensorboard. 
logger = imit_logger.configure(
    folder=log_dir, 
    format_strs=["stdout", "csv", "tensorboard"]
)


# Define the Car Racing environment for training. 
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
            #"render_mode": "human",        # Uncomment if want to observe the training process from the environment. 
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
            #"render_mode": "human",
            "continuous": False,
        }, 
    )
env = VecTransposeImage(env)

# Show the observation and action space to check whether the training environment setting is correct. 
print(f"Training environment observation space: {env.observation_space}")
print(f"Training environment action space: {env.action_space}")


# Load the trained PPO expert policy. 
print("Prepare to load expert policy...")
expert_model = load_policy(
    "ppo",
    path=expert_model_path,
    venv=env,
)
print("Expert policy loaded!")

# Use the rollout function to collect the demonstrations. 
print("Using expert policy to generate demonstrations...")
rollouts = rollout.rollout(
    policy=expert_model, 
    venv=env, 
    sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=TRAJ_NUM), 
    rng=rng, 
    #deterministic_policy=True, 
)

# Flattens the trajectories to transitions. 
transitions = flatten_trajectories_with_transpose(rollouts)
print("Finished to genertate demonstrations!")

# Check the format of transitions. 
print(f"Trajectory dataset: {transitions.obs.shape[0]}")
print(f"Observation shape: {transitions.obs.shape}")
print(f"Action Sample: {transitions.acts[0]}")



# Create the CNN instance. 
cnn_instance = CnnPolicy(
    observation_space=env.observation_space, 
    action_space=env.action_space, 
    lr_schedule=lambda _: 3e-4, 
)

# Initialize the BC trainer instance. 
if IS_MLP:
    # MLP case, use the FeedForward32Policy. 
    bc_trainer = bc.BC(
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        demonstrations=transitions, 
        rng=rng, 
        custom_logger=logger, 
        #l2_weight=0.0001,           # Optional for L2 loss. 
    )
else:
    # CNN case, use the CnnPolicy. 
    bc_trainer = bc.BC(
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        demonstrations=transitions, 
        rng=rng, 
        policy=cnn_instance, 
        custom_logger=logger, 
        #l2_weight=0.0001, 
    )

# Evaluate the policy before training. 
print("Evaluate the BC policy before training...")
mean_reward_before_training, std_reward_before_training = evaluate_policy(
                                                                model=bc_trainer.policy, 
                                                                env=env, 
                                                                n_eval_episodes=EVAL_EPISODES, 
                                                                deterministic=True, 
                                                                return_episode_rewards=False,
)

# Start BC training. 
print("Start training the BC policy...")
bc_trainer.train(n_epochs=TRAIN_EPOCHS)


# Save the trained BC policy. 
os.makedirs(f"models/bc", exist_ok=True)

if IS_CONTINUOUS:
    if IS_MLP:
        bc_trainer.policy.save(f"models/bc/bc-mlp-continuous-traj_{TRAJ_NUM}.zip")
    else:
        bc_trainer.policy.save(f"models/bc/bc-continuous-traj_{TRAJ_NUM}.zip")
else:
    if IS_MLP:
        bc_trainer.policy.save(f"models/bc/bc-mlp-discrete-traj_{TRAJ_NUM}.zip")
    else:
        bc_trainer.policy.save(f"models/bc/bc-discrete-traj_{TRAJ_NUM}.zip")
print("Trained BC policy has been save to path: models/bc/")

# Evaluate the policy after training. 
print("Evaluate the BC policy after training...")
mean_reward_after_training, std_reward_after_training = evaluate_policy(
                                                                model=bc_trainer.policy, 
                                                                env=env, 
                                                                n_eval_episodes=EVAL_EPISODES, 
                                                                deterministic=True, 
                                                                return_episode_rewards=False,
)


# Show the results. 
print("===================== Results ==========================")
print(f"Trajectory Number: {TRAJ_NUM}")
print(f"Trajectory Dataset: {transitions.obs.shape[0]}")
print(f"Reward before training 50 epochs: {mean_reward_before_training:.2f} +/- {std_reward_before_training:.2f}")
print(f"Reward after training 50 epochs: {mean_reward_after_training:.2f} +/- {std_reward_after_training:.2f}")
