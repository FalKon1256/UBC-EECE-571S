import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env


class ActionFloat32Wrapper(gym.ActionWrapper):
    """Force changing the type of the action to a list of Python float to fix the Box2D type error."""
    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return [float(a) for a in action]


# Defining the parameters and paths. 
IS_CONTINUOUS = True
TOTAL_TRAIN_TIMESTEPS= 1_000_000     #1_000_000
SEED = 0

if IS_CONTINUOUS:
    log_dir = "logs/ppo_log/continuous"
    checkpoint_name_prefix = "ppo_continuous"
    checkpoint_save_dir = f"models/ppo/continuous/ppo_timesteps_{TOTAL_TRAIN_TIMESTEPS}/checkpoints"
    final_policy_save_dir = "models/ppo/continuous/ppo_final_policy"
else:
    log_dir = "logs/ppo_log/discrete"
    checkpoint_name_prefix = "ppo_discrete"
    checkpoint_save_dir = f"models/ppo/discrete/ppo_timesteps_{TOTAL_TRAIN_TIMESTEPS}/checkpoints"
    final_policy_save_dir = "models/ppo/discrete/ppo_final_policy"


# Define the Car Racing environment for training. 
rng = np.random.default_rng(SEED)

if IS_CONTINUOUS:
    # Continuous action space. 
    env = make_vec_env(
        "CarRacing-v2", 
        rng=rng, 
        n_envs=1, 
        post_wrappers=[
            lambda env, _: ActionFloat32Wrapper(env),       # Needs to place first for the post wrappers, not required if discrete action space. 
            lambda env, _: RolloutInfoWrapper(env),
        ],
        env_make_kwargs={
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
            "continuous": False,
        },
    )
env = VecTransposeImage(env)

# Show the observation and action space to check whether the training environment setting is correct. 
print(f"Training environment observation space: {env.observation_space}")
print(f"Training environment action space: {env.action_space}")


# Function for PPO training. 
def train_expert():
    """This is the training function for the PPO policy to train an expert policy."""

    print("Start PPO training for the expert policy ...")

    # Create the evaluation environment for the callback function. 
    if IS_CONTINUOUS:
        # Continuous action space. 
        eval_env = make_vec_env(
            "CarRacing-v2",
            rng=rng,
            n_envs=1,
            post_wrappers=[
                lambda env, _: ActionFloat32Wrapper(env),
                lambda env, _: RolloutInfoWrapper(env),
            ],
            env_make_kwargs={
                "continuous": True,
            },
        )
    else:
        # Discrete action space. 
        eval_env = make_vec_env(
            "CarRacing-v2",
            rng=rng,
            n_envs=1,
            post_wrappers=[
                lambda env, _: RolloutInfoWrapper(env),
            ],
            env_make_kwargs={
                "continuous": False,
            },
        )
    eval_env = VecTransposeImage(eval_env)

    # Show the observation and action space to check whether the evaluation environment setting is correct. 
    print("Evaluation environment observation space: ", eval_env.observation_space)
    print("Evaluation environment action space: ", eval_env.action_space)
    
    # Save the checkpoint for every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=checkpoint_save_dir, 
        name_prefix=checkpoint_name_prefix, 
    )
    
    # Conduct a fast evaluation for every 10k steps, while updating the best model. 
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Create the evaluation callback instance for the training process. 
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Create a PPO policy instance for training. 
    expert = PPO(
        policy=CnnPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,         #0.005
        learning_rate=3e-4,
        n_epochs=10,
        n_steps=2048,
        verbose=1,
        device="cuda",
        tensorboard_log=log_dir,
    )

    # Start PPO training. 
    expert.learn(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name="ppo", callback=callbacks)
    return expert

if __name__ == "__main__":

    # Train the PPO policy as an expert policy. 
    expert = train_expert()

    # Save the policy models. 
    expert.save(final_policy_save_dir)
    print("Final PPO policy saved!")
