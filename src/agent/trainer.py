import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from src.env import FuturesTradingEnv
import json

# Attempt to load RecurrentPPO for LSTM policy. Fallback to standard PPO if sb3-contrib is unavailable.
try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
    print("[Agent Trainer] sb3-contrib found! Using RecurrentPPO (PPO-LSTM).")
except ImportError:
    from stable_baselines3 import PPO
    USING_RECURRENT = False
    print("[Agent Trainer] sb3-contrib not found. Falling back to standard Feed-Forward PPO.")

# Check if TensorBoard is installed to prevent crashes during SB3 log initialization
try:
    import tensorboard
    HAS_TENSORBOARD = True
    print("[Agent Trainer] TensorBoard is installed. Logging enabled.")
except ImportError:
    HAS_TENSORBOARD = False
    print("[Agent Trainer] TensorBoard not found. Disabling TensorBoard logging to prevent crashes.")


class ProgressCallback(BaseCallback):
    """Callback for tracking training progress and saving to a JSON file."""
    def __init__(self, model_name: str, total_timesteps: int, check_stop_fn=None, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.progress_file = os.path.join("models", f"progress_{model_name}.json")
        self.check_stop_fn = check_stop_fn
        self.last_write_step = 0
        self.start_timesteps = 0
        self.was_aborted = False
        
    def _on_training_start(self) -> None:
        self.start_timesteps = self.model.num_timesteps
        try:
            with open(self.progress_file, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "current_step": 0,
                    "total_steps": self.total_timesteps,
                    "percentage": 0.0,
                    "status": "training"
                }, f)
        except Exception:
            pass
        
    def _on_step(self) -> bool:
        # Check if external stop request was triggered
        if self.check_stop_fn is not None and self.check_stop_fn():
            print(f"[Agent Trainer] Stop request detected. Halting training for {self.model_name}...")
            self.was_aborted = True
            steps_trained = self.num_timesteps - self.start_timesteps
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({
                        "model_name": self.model_name,
                        "current_step": steps_trained,
                        "total_steps": self.total_timesteps,
                        "percentage": round(min(100.0, (steps_trained / self.total_timesteps) * 100.0), 2),
                        "status": "stopped"
                    }, f)
            except Exception:
                pass
            return False # Returning False stops stable-baselines3 learning loop

        steps_trained = self.num_timesteps - self.start_timesteps
        # Throttling disk I/O progress updates (write only once every 500 steps)
        # to prevent high-frequency write locks and resolve Windows file access conflicts in UI
        if steps_trained - self.last_write_step >= 500 or steps_trained == 1:
            self.last_write_step = steps_trained
            pct = min(100.0, (steps_trained / self.total_timesteps) * 100.0)
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({
                        "model_name": self.model_name,
                        "current_step": steps_trained,
                        "total_steps": self.total_timesteps,
                        "percentage": round(pct, 2),
                        "status": "training"
                    }, f)
            except Exception as e:
                pass
        return True


    def _on_training_end(self) -> None:
        try:
            with open(self.progress_file, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "current_step": self.total_timesteps,
                    "total_steps": self.total_timesteps,
                    "percentage": 100.0,
                    "status": "completed"
                }, f)
        except Exception as e:
            pass


def lr_schedule(progress_remaining: float) -> float:
    """Linear learning rate schedule from 3e-4 to 5e-5."""
    initial_lr = 3e-4
    final_lr = 5e-5
    return final_lr + (initial_lr - final_lr) * progress_remaining


def train_agent(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    total_timesteps: int = 100000,
    model_save_dir: str = "models",
    tb_log_dir: str = "tb_logs",
    model_name: str = "ppo_volume_bars_child",
    check_stop_fn = None,
    resume: bool = False,
    learning_rate_val = None
):
    """
    Trains a PPO or RecurrentPPO agent on the custom futures trading environment.
    Supports resuming existing models and custom learning rates.
    Uses Vector Normalization for observation stability.
    """
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Extract clean symbol from model_name
    symbol_clean = model_name.replace("ppo_volume_bars_child_", "").upper()

    # Setup custom learning rate schedule
    if learning_rate_val is None:
        lr_input = lr_schedule
    elif isinstance(learning_rate_val, float):
        lr_input = learning_rate_val
    elif isinstance(learning_rate_val, str):
        val_str = learning_rate_val.lower().strip()
        if val_str.startswith("linear_"):
            try:
                start_lr = float(val_str.split("_")[1])
                end_lr = start_lr / 6.0
                def make_linear_decay(s_lr, e_lr):
                    return lambda progress_remaining: e_lr + (s_lr - e_lr) * progress_remaining
                lr_input = make_linear_decay(start_lr, end_lr)
            except Exception as e:
                print(f"[Agent Trainer] Error parsing linear LR: {e}. Falling back to default lr_schedule.")
                lr_input = lr_schedule
        elif val_str.startswith("constant_"):
            try:
                lr_input = float(val_str.split("_")[1])
            except Exception as e:
                print(f"[Agent Trainer] Error parsing constant LR: {e}. Falling back to default lr_schedule.")
                lr_input = lr_schedule
        else:
            try:
                lr_input = float(val_str)
            except ValueError:
                lr_input = lr_schedule
    else:
        lr_input = learning_rate_val

    model_path = None
    final_path = os.path.join(model_save_dir, f"{model_name}_final.zip")
    best_path = os.path.join(model_save_dir, f"{model_name}_best.zip")
    stats_path = os.path.join(model_save_dir, f"{model_name}_vec_normalize.pkl")
    
    if resume:
        if os.path.exists(final_path):
            model_path = final_path
        elif os.path.exists(best_path):
            model_path = best_path
            
        if model_path:
            print(f"[Agent Trainer] Resuming training. Loading existing model from {model_path}...")
        else:
            print(f"[Agent Trainer] Warning: Resume is set to True but no existing model was found at {final_path} or {best_path}. Starting training from scratch.")
            resume = False

    # 1. Initialize vectorized environments
    def make_train_env():
        return FuturesTradingEnv(train_df, symbol=symbol_clean)
    
    def make_val_env():
        return FuturesTradingEnv(val_df, symbol=symbol_clean)
        
    train_env = DummyVecEnv([make_train_env])
    val_env = DummyVecEnv([make_val_env])
    
    if resume and os.path.exists(stats_path):
        print(f"[Agent Trainer] Loading VecNormalize statistics from {stats_path}...")
        train_env = VecNormalize.load(stats_path, train_env)
        # Ensure training is True to continue updating stats
        train_env.training = True
        
        val_env = VecNormalize.load(stats_path, val_env)
        val_env.training = False # Don't update validation stats
    else:
        # Normalize observations
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        # Synchronize stats
        val_env.obs_rms = train_env.obs_rms
        val_env.training = False # Turn off updates for evaluation env
    
    # 2. Setup agent policy & hyperparameters
    # Optimized parameters for high-frequency trading with long-term trend awareness (gamma=0.98)
    hyperparams = {
        "learning_rate": lr_input,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01, # encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": tb_log_dir if HAS_TENSORBOARD else None
    }
    
    model = None
    if resume and model_path:
        custom_objects = {}
        if lr_input is not None:
            custom_objects["learning_rate"] = lr_input
            
        try:
            if USING_RECURRENT:
                model = RecurrentPPO.load(model_path, env=train_env, custom_objects=custom_objects)
            else:
                model = PPO.load(model_path, env=train_env, custom_objects=custom_objects)
            print(f"[Agent Trainer] Model successfully loaded for resume.")
        except Exception as e:
            print(f"[Agent Trainer] Error loading model: {e}. Falling back to starting from scratch.")
            resume = False

    if model is None:
        if USING_RECURRENT:
            # LSTM specific policy network with shared memory & 128 units
            policy = "MlpLstmPolicy"
            policy_kwargs = dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                shared_lstm=True,
                enable_critic_lstm=False,
                net_arch=dict(pi=[64, 64], vf=[64, 64])
            )
            model = RecurrentPPO(
                policy,
                train_env,
                policy_kwargs=policy_kwargs,
                **hyperparams
            )
        else:
            policy = "MlpPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64])
            )
            model = PPO(
                policy,
                train_env,
                policy_kwargs=policy_kwargs,
                **hyperparams
            )
        
    # 3. Setup Eval Callback
    # Monitor validation performance and save the best model
    # Evaluates every 60,000 steps as requested by the research team
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=model_save_dir,
        log_path=tb_log_dir if HAS_TENSORBOARD else None,
        eval_freq=60000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # 4. Train the model
    print(f"[Agent Trainer] Starting training for {total_timesteps} steps...")
    progress_callback = ProgressCallback(model_name=model_name, total_timesteps=total_timesteps, check_stop_fn=check_stop_fn)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, progress_callback],
        tb_log_name=model_name,
        reset_num_timesteps=not resume
    )
    
    # If training was aborted, do NOT save final weights to prevent corruption of previous fully trained files
    if progress_callback.was_aborted:
        print(f"[Agent Trainer] Training was ABORTED early by user request. Skipping final model file save to preserve previous fully trained models.")
        return None, train_env

    # 5. Save the final model and vec normalization statistics
    final_model_path = os.path.join(model_save_dir, f"{model_name}_final")
    model.save(final_model_path)
    
    # If the evaluation callback saved a best model, copy/rename it to model_name + "_best.zip"
    best_temp_path = os.path.join(model_save_dir, "best_model.zip")
    best_target_path = os.path.join(model_save_dir, f"{model_name}_best")
    if os.path.exists(best_temp_path):
        import shutil
        shutil.move(best_temp_path, best_target_path + ".zip")
        print(f"[Agent Trainer] Best evaluation model renamed to {best_target_path}.zip")
    
    train_env.save(stats_path)
    
    print(f"[Agent Trainer] Training finished! Model saved to {final_model_path}")
    print(f"[Agent Trainer] VecNormalize statistics saved to {stats_path}")
    
    return model, train_env

