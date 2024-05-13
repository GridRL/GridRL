#!/usr/bin/env python3

"""SB3 training example."""

from typing import Union, Callable, Any
import warnings
import sys
import os

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import numpy as np

    try:
        from sbx import PPO
        from sbx.ppo.ppo import PPOPolicy
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecNormalize,
            VecMonitor,
        )
        from stable_baselines3.common.callbacks import CallbackList
    except ModuleNotFoundError as exc:
        print(
            f"Couldn't load [sbx] package. Try >> pip install sbx-rl.\n\tError: {exc}"
        )
        raise

from gridrl.envs import make_env
from example_environment import ExplorationWorld1ExampleEnv

sys.dont_write_bytecode = True

#__all__=[]

def load_sb3_model(
    env: Any,
    model_checkpoint_filename: Union[str, None],
    num_envs: int,
    model_class: Any,
    policy_class: Any,
    **model_args,
):
    """Builds the model loading an existing checkpoint."""
    fallback = True
    if model_checkpoint_filename is not None and os.path.isfile(
        model_checkpoint_filename
    ):
        print(f"Loading from checkpoint [{model_checkpoint_filename}].")
        custom_objects = None
        try:
            model = model_class.load(
                model_checkpoint_filename, env=env, custom_objects=custom_objects
            )
            if num_envs >= 2:
                model.n_steps = model_args.get("n_steps", 2**11)
                model.n_envs = num_envs
                model.rollout_buffer.buffer_size = model_args.get("n_steps", 2**11)
                model.rollout_buffer.n_envs = num_envs
                model.rollout_buffer.reset()
            fallback = False
        except AttributeError:
            pass
    if fallback:
        model = model_class(policy_class, env, verbose=1, **model_args)
    return model


def train_sb3(
    env_class: Callable,
    env_config: dict,
    train_config: dict,
    model_class: Any,
    model_args: dict,
    policy_class: Any,
    callbacks: list,
    seed: int = 0,
):
    """Training routine."""
    train_config["episode_length"] = max(1, train_config.get("episode_length", 2**14))
    train_config["num_envs"] = max(1, int(train_config.get("num_envs", 1)))
    env_config["max_steps"] = train_config["episode_length"]
    env_config.update({"flat_obs": True, "flat_obs_dtype": np.float32})
    vec_class = DummyVecEnv if train_config["num_envs"] < 2 else SubprocVecEnv
    env = vec_class(
        [
            (make_env(env_class=env_class, rank=i, env_config=env_config, seed=seed))
            for i in range(train_config["num_envs"])
        ]
    )
    if train_config.get("monitor", True):
        env = VecMonitor(env)
    if train_config.get("normalize", False):
        env = VecNormalize(env)
    model = load_sb3_model(
        env,
        train_config.get("model_init_checkpoint_filename", None),
        train_config["num_envs"],
        model_class,
        policy_class,
        **model_args,
    )
    if train_config["num_envs"] < 2:
        print(
            "Train_config [num_envs] is 1, running in sequential mode."
            " Increase this field to speed-up training."
        )
    try:
        params_count = int(
            np.sum([p.numel() for p in model.policy.parameters() if p.requires_grad])
        )
        print(
            f"Env instances: {train_config['num_envs']:d}\tModel parameters: {params_count:d}"
            f"\nLR: {model.learning_rate}"
            f"\tOptimizerLR: {None if not hasattr(model.policy,'optimizer') else model.policy.optimizer.param_groups[0].get('lr', None)}"
            f"\tent_coef: {model.ent_coef}"
            f"\nAct space: {env.action_space}\tObs space: {env.observation_space}"
        )
    except AttributeError:
        pass
    callbacks = []
    learn_steps = (
        train_config["episode_length"]
        * train_config["num_envs"]
        * max(1, int(train_config.get("episode_per_env_repeats", 10)))
    )
    learn_repeats = max(1, int(train_config.get("episode_per_env_repeats", 1)))
    for _ in range(learn_repeats):
        should_break = False
        try:
            model.learn(
                total_timesteps=learn_steps,
                callback=CallbackList(callbacks),
                tb_log_name="train_model",
            )
        except (KeyboardInterrupt, SystemExit):
            should_break = True
        if train_config.get("model_save_checkpoint_filename", None) is not None:
            try:
                model.save(train_config["model_save_checkpoint_filename"])
            except (TypeError, FileNotFoundError, OSError):
                pass
        if should_break:
            break


if __name__ == "__main__":
    USED_NUM_ENVS = 1
    UsedModelClass = PPO
    UsedPolicyClass = PPOPolicy
    train_sb3(
        env_class=ExplorationWorld1ExampleEnv,
        env_config={
            "screen_observation_type": 3,
            "action_complexity": 3,
            "action_nop": False,
            "auto_screen_obs": True,
            "grayscale_screen": True,
            "grayscale_single_channel": True,
            "flat_obs": False,
            "flat_obs_dtype": np.float32,
        },
        train_config={
            "episode_length": 2**14,
            "num_envs": USED_NUM_ENVS,
            "episode_per_env_repeats": 10,
            "learn_repeats": 5,
            "monitor": True,
            "normalize": False,
            "model_init_checkpoint_filename": f"{sys.path[0]}sb3_jax_flat_init_checkpoint.zip",
            "model_save_checkpoint_filename": f"{sys.path[0]}sb3_jax_flat_save_checkpoint.zip",
        },
        model_class=UsedModelClass,
        policy_class=UsedPolicyClass,
        model_args={
            "learning_rate": 2e-4,
            "n_steps": 2**11,
            "batch_size": 2**9,
            "n_epochs": 3,
            "gamma": 0.995,
            "clip_range": 0.1,
            "target_kl": 0.05,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
        callbacks=[],
        seed=0,
    )
