import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from imitation.util import logger as imit_logger


class ActionFloat32Wrapper(gym.ActionWrapper):
    """Force changing the type of the action to a list of Python float to fix the Box2D type error."""
    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return [float(a) for a in action]


# Define the control parameters for experiments. 
TOTAL_STEPS = 1000      #10000
IS_CONTINUOUS = True
IS_MLP = False
SEED = 0                #0
EVAL_EPISODES = 20      #20


# Set the loading path of the expert model trained by PPO. 
if IS_CONTINUOUS:
    expert_model_path = "models/ppo/continuous/ppo_timesteps_1000000/checkpoints/ppo_continuous_1000000_steps.zip"
else:
    expert_model_path = "models/ppo/discrete/ppo_timesteps_1000000/checkpoints/ppo_discrete_1000000_steps.zip"

# Set the saving path for the log data.  
if IS_CONTINUOUS:
    if IS_MLP:
        log_dir = f"logs/dagger_log/continuous/mlp-totalsteps_{TOTAL_STEPS}"
        traj_dir = f"models/dagger/rollout/continuous/traj_rollout-mlp-totalsteps_{TOTAL_STEPS}"
    else:
        log_dir = f"logs/dagger_log/continuous/totalsteps_{TOTAL_STEPS}"
        traj_dir = f"models/dagger/rollout/continuous/traj_rollout-totalsteps_{TOTAL_STEPS}"
else:
    if IS_MLP:
        log_dir = f"logs/dagger_log/discrete/mlp-totalsteps_{TOTAL_STEPS}"
        traj_dir = f"models/dagger/rollout/discrete/traj_rollout-mlp-totalsteps_{TOTAL_STEPS}"
    else:
        log_dir = f"logs/dagger_log/discrete/totalsteps_{TOTAL_STEPS}"
        traj_dir = f"models/dagger/rollout/discrete/traj_rollout-totalsteps_{TOTAL_STEPS}"

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
            #"render_mode": "human",
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
expert = load_policy(
    "ppo",
    path=expert_model_path,
    venv=env,
)
print("Expert policy loaded!")



# Create the CNN instance. 
cnn_instance = CnnPolicy(
    observation_space=env.observation_space, 
    action_space=env.action_space, 
    lr_schedule=lambda _: 3e-4, 
)

# Initialize the BC trainer instance for DAgger training. 
if IS_MLP:
    # MLP case, use the FeedForward32Policy. 
    bc_trainer = bc.BC(
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        rng=rng,                                # Randomly shuffle the expert dataset. 
        custom_logger=logger, 
    )
else:
    # CNN case, use the CnnPolicy. 
    bc_trainer = bc.BC(
        observation_space=env.observation_space, 
        action_space=env.action_space, 
        rng=rng, 
        policy=cnn_instance, 
        custom_logger=logger, 
    )

# Create the DAgger trainer instance. 
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=traj_dir, 
    expert_policy=expert,
    rng=rng,                    # Randomly deciding whether the expert policy interacts with the environment. 
    bc_trainer=bc_trainer,
)

# Evaluate the policy before training. 
print("Evaluate the DAgger policy before training...")
mean_reward_before_training, std_reward_before_training = evaluate_policy(
                                                                model=dagger_trainer.policy, 
                                                                env=env, 
                                                                n_eval_episodes=EVAL_EPISODES, 
                                                                deterministic=True, 
                                                                return_episode_rewards=False,
)

# Start DAgger training. 
print("Start training the DAgger policy...")
dagger_trainer.train(
    total_timesteps=TOTAL_STEPS,            # Collects how many steps in total for the dataset. 
    rollout_round_min_episodes=1,           # At least run 1 map for each update. 
    bc_train_kwargs={"n_epochs": 5}         # Train 5 epoches after adding new trajectories for each round. 
    #rollout_round_min_timesteps=1000,      # Optional: A lower bound on how many time steps each round has to run. 
)

# Save the trained DAgger policy. 
os.makedirs(f"models/dagger", exist_ok=True)

if IS_CONTINUOUS:
    if IS_MLP:
        dagger_trainer.policy.save(f"models/dagger/dagger-mlp-continuous-totalsteps_{TOTAL_STEPS}.zip")
    else:
        dagger_trainer.policy.save(f"models/dagger/dagger-continuous-totalsteps_{TOTAL_STEPS}.zip")
else:
    if IS_MLP:
        dagger_trainer.policy.save(f"models/dagger/dagger-mlp-discrete-totalsteps_{TOTAL_STEPS}.zip")
    else:
        dagger_trainer.policy.save(f"models/dagger/dagger-discrete-totalsteps_{TOTAL_STEPS}.zip")
print("Trained DAgger policy has been save to path: models/dagger/")

# Evaluate the policy after training. 
print("Evaluate the DAgger policy after training...")
mean_reward_after_training, std_reward_after_training = evaluate_policy(
                                                                model=dagger_trainer.policy, 
                                                                env=env, 
                                                                n_eval_episodes=EVAL_EPISODES, 
                                                                deterministic=True, 
                                                                return_episode_rewards=False,
)

# Show the results. 
print("===================== Results ==========================")
print(f"total_steps: {TOTAL_STEPS}")
print(f"Reward before training: {mean_reward_before_training:.2f} +/- {std_reward_before_training:.2f}")
print(f"Reward after training: {mean_reward_after_training:.2f} +/- {std_reward_after_training:.2f}")
