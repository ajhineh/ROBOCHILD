import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
import json
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3 import SAC, TD3
from src.env import FuturesTradingEnv

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
    """Callback for tracking training progress and saving to a unified JSON progress file."""
    def __init__(self, progress_file: str, model_name: str, phase_name: str, total_timesteps: int, phase_weight: float, phase_offset: float, check_stop_fn=None, verbose=0):
        super().__init__(verbose)
        self.progress_file = progress_file
        self.model_name = model_name
        self.phase_name = phase_name
        self.total_timesteps = total_timesteps
        self.phase_weight = phase_weight
        self.phase_offset = phase_offset
        self.check_stop_fn = check_stop_fn
        self.last_write_step = 0
        self.start_timesteps = 0
        self.was_aborted = False
        
    def _on_training_start(self) -> None:
        self.start_timesteps = self.model.num_timesteps
        
    def _on_step(self) -> bool:
        if self.check_stop_fn is not None and self.check_stop_fn():
            print(f"[Agent Trainer] Stop request detected. Halting training for phase {self.phase_name}...")
            self.was_aborted = True
            steps_trained = self.num_timesteps - self.start_timesteps
            phase_pct = (steps_trained / self.total_timesteps) * 100.0
            overall_pct = self.phase_offset + (phase_pct * self.phase_weight)
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({
                        "model_name": self.model_name,
                        "current_step": steps_trained,
                        "total_steps": self.total_timesteps,
                        "percentage": round(min(100.0, overall_pct), 2),
                        "status": f"stopped ({self.phase_name})"
                    }, f)
            except Exception:
                pass
            return False # Stops learning loop

        steps_trained = self.num_timesteps - self.start_timesteps
        
        # Smart alerts triggering
        try:
            import wandb
            if wandb.run is not None:
                # 1. PPO Low Explained Variance Alert after 50k steps
                if self.phase_name.upper() == "PPO" and steps_trained >= 50000:
                    if hasattr(self.model, "logger") and self.model.logger is not None:
                        ev = self.model.logger.name_to_value.get("train/explained_variance")
                        if ev is not None and ev < 0.4:
                            if not hasattr(self, "_alert_ev_sent"):
                                self._alert_ev_sent = True
                                wandb.alert(
                                    title="PPO Low Explained Variance Warning",
                                    text=f"Warning: Explained variance is {ev:.4f} after {steps_trained} steps (target >= 0.4). Training is unstable.",
                                    level="WARN"
                                )
                
                # 2. Negative Mean Reward Alert after 100k steps
                if steps_trained >= 100000:
                    if hasattr(self.model, "logger") and self.model.logger is not None:
                        reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean") or self.model.logger.name_to_value.get("eval/mean_reward")
                        if reward is not None and reward < 0:
                            if not hasattr(self, "_alert_reward_sent"):
                                self._alert_reward_sent = True
                                wandb.alert(
                                    title="Negative Mean Reward Alert",
                                    text=f"Warning: Mean reward is {reward:.4f} after {steps_trained} steps. Model fails to yield positive results.",
                                    level="WARN"
                                )
                
                # 3. High KL Divergence Alert
                if hasattr(self.model, "logger") and self.model.logger is not None:
                    kl = self.model.logger.name_to_value.get("train/approx_kl")
                    if kl is not None and kl > 0.05:
                        if not hasattr(self, "_alert_kl_sent"):
                            self._alert_kl_sent = True
                            wandb.alert(
                                title="High KL Divergence Alert",
                                text=f"Warning: Approximate KL Divergence is {kl:.4f} indicating policy update instability.",
                                level="WARN"
                            )
        except Exception:
            pass

        if steps_trained - self.last_write_step >= 500 or steps_trained == 1:
            self.last_write_step = steps_trained
            phase_pct = (steps_trained / self.total_timesteps) * 100.0
            overall_pct = self.phase_offset + (phase_pct * self.phase_weight)
            try:
                with open(self.progress_file, "w") as f:
                    json.dump({
                        "model_name": self.model_name,
                        "current_step": steps_trained,
                        "total_steps": self.total_timesteps,
                        "percentage": round(min(100.0, overall_pct), 2),
                        "status": f"training ({self.phase_name})"
                    }, f)
            except Exception as e:
                pass
        return True


def lr_schedule(progress_remaining: float) -> float:
    """Linear learning rate schedule from 1.5e-4 to 2e-5."""
    initial_lr = 1.5e-4
    final_lr = 2e-5
    return final_lr + (initial_lr - final_lr) * progress_remaining


def purge_and_embargo_data(train_df: pd.DataFrame, val_df: pd.DataFrame, max_holding_steps: int = 48, embargo_candles: int = 576) -> pd.DataFrame:
    """
    Purges and embargoes train_df relative to val_df to prevent data leakage.
    Assumes train_df is chronologically before val_df.
    """
    if len(train_df) == 0 or len(val_df) == 0:
        return train_df
        
    # Purging: Drop the last max_holding_steps of train_df to prevent labels from extending into val_df
    # Embargoing: To prevent autoregressive feature leakage, add a gap of embargo_candles between train and val
    total_drop = max_holding_steps + embargo_candles
    
    if len(train_df) > total_drop * 2:
        print(f"[Agent Trainer] Purging & Embargoing: dropping the last {total_drop} candles from train_df.")
        train_df = train_df.iloc[:-total_drop]
    else:
        # Fallback if train_df is too short
        drop_len = min(len(train_df) // 4, total_drop)
        print(f"[Agent Trainer] train_df is short. Dropping last {drop_len} candles.")
        train_df = train_df.iloc[:-drop_len]
        
    return train_df


def train_agent(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    total_timesteps: int = 100000,
    model_save_dir: str = "models",
    tb_log_dir: str = "tb_logs",
    model_name: str = "ppo_volume_bars_child",
    check_stop_fn = None,
    resume: bool = False,
    learning_rate_val = None,
    override_hyperparams = None
):
    """
    Trains an Ensemble of agents: PPO-LSTM, SAC, and TD3 sequentially.
    """
    # Apply Purging & Embargoing to prevent data leakage (de Prado Pitfall #6)
    train_df = purge_and_embargo_data(train_df, val_df, max_holding_steps=48, embargo_candles=576)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    symbol_clean = model_name.replace("ppo_volume_bars_child_", "").upper()
    unified_progress_file = os.path.join(model_save_dir, f"progress_{model_name}.json")

    # Load dynamic config early to extract learning rate or other parameters
    config_path = os.path.join(model_save_dir, f"config_{symbol_clean.lower()}.json")
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            print(f"[Agent Trainer] Dynamic config pre-loaded from {config_path}")
        except Exception as e:
            print(f"[Agent Trainer] Error pre-loading config: {e}")

    # Setup custom learning rate schedule
    if learning_rate_val is None:
        learning_rate_val = config_data.get("learning_rate")

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

    # 1.5. Dynamic frame_stack shape checking for resume to prevent SB3 crashes
    override_stack = None
    if resume:
        # Check first phase (PPO) model shape to keep observation space consistent
        ppo_final = os.path.join(model_save_dir, f"{model_name}_ppo_final.zip")
        ppo_best = os.path.join(model_save_dir, f"{model_name}_ppo_best.zip")
        model_path_to_check = ppo_final if os.path.exists(ppo_final) else (ppo_best if os.path.exists(ppo_best) else None)
        
        if model_path_to_check:
            try:
                from stable_baselines3 import PPO
                # Load metadata without environment to check observation space shape
                tmp_model = PPO.load(model_path_to_check, device="cpu")
                obs_shape = tmp_model.observation_space.shape
                if obs_shape and len(obs_shape) > 0:
                    features_dim = obs_shape[0]
                    detected_stack = int(features_dim / 12)
                    print(f"[Agent Trainer] Detected existing PPO model observation space shape: {obs_shape}. Forcing frame_stack = {detected_stack} to prevent shape mismatch.")
                    override_stack = detected_stack
            except Exception as e:
                print(f"[Agent Trainer] Warning inspecting PPO model shape: {e}. Trying RecurrentPPO load...")
                try:
                    if USING_RECURRENT:
                        tmp_model = RecurrentPPO.load(model_path_to_check, device="cpu")
                        obs_shape = tmp_model.observation_space.shape
                        if obs_shape and len(obs_shape) > 0:
                            features_dim = obs_shape[0]
                            detected_stack = int(features_dim / 12)
                            print(f"[Agent Trainer] Detected existing RecurrentPPO model observation space shape: {obs_shape}. Forcing frame_stack = {detected_stack} to prevent shape mismatch.")
                            override_stack = detected_stack
                except Exception as e2:
                    print(f"[Agent Trainer] Failed to inspect model shape via RecurrentPPO: {e2}")

    # Define environment creators
    def make_train_env():
        return FuturesTradingEnv(train_df, symbol=symbol_clean, override_num_stack=override_stack)
    
    def make_val_env():
        return FuturesTradingEnv(val_df, symbol=symbol_clean, override_num_stack=override_stack)

    # We will train 3 algorithms sequentially: PPO, SAC, TD3
    phases = [
        {"name": "PPO", "algo": "ppo", "weight": 0.34, "offset": 0.0},
        {"name": "SAC", "algo": "sac", "weight": 0.33, "offset": 34.0},
        {"name": "TD3", "algo": "td3", "weight": 0.33, "offset": 67.0}
    ]

    for phase in phases:
        algo_name = phase["algo"]
        print(f"\n==================================================")
        print(f"[Agent Trainer] Starting Phase: {phase['name']} Training")
        print(f"==================================================")

        # 1. Initialize vectorized environments (individual normalization stats per algorithm)
        train_env = DummyVecEnv([make_train_env])
        val_env = DummyVecEnv([make_val_env])

        # Load or create VecNormalize
        stats_path = os.path.join(model_save_dir, f"{model_name}_{algo_name}_vec_normalize.pkl")
        if resume and os.path.exists(stats_path):
            print(f"[Agent Trainer] Loading VecNormalize statistics for {algo_name} from {stats_path}...")
            train_env = VecNormalize.load(stats_path, train_env)
            train_env.training = True
            val_env = VecNormalize.load(stats_path, val_env)
            val_env.training = False
        else:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            val_env.obs_rms = train_env.obs_rms
            val_env.training = False

        # تلاش برای بارگذاری تنظیمات هایپرپارامترها از فایل پیکربندی نماد
        config_path = os.path.join(model_save_dir, f"config_{symbol_clean.lower()}.json")
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                print(f"[Agent Trainer] Dynamic config loaded from {config_path}")
            except Exception as e:
                print(f"[Agent Trainer] Warning loading config file: {e}")
        
        if override_hyperparams:
            config_data.update(override_hyperparams)
            print(f"[Agent Trainer] Overrode hyperparameters: {override_hyperparams}")

        # 2. Setup Policy kwargs & Hyperparams
        eval_freq = config_data.get("eval_freq", max(2000, total_timesteps // 5))
        patience = config_data.get("early_stopping_patience", 3)
        
        stop_callback = None
        if patience > 0:
            try:
                from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
                stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=patience, min_evals=1, verbose=1)
                print(f"[Agent Trainer] Early Stopping enabled with patience = {patience} evaluations.")
            except ImportError:
                print("[Agent Trainer] StopTrainingOnNoModelImprovement not available in this SB3 version.")

        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=model_save_dir,
            log_path=tb_log_dir if HAS_TENSORBOARD else None,
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback
        )

        progress_callback = ProgressCallback(
            progress_file=unified_progress_file,
            model_name=model_name,
            phase_name=phase["name"],
            total_timesteps=total_timesteps,
            phase_weight=phase["weight"],
            phase_offset=phase["offset"],
            check_stop_fn=check_stop_fn
        )

        model = None
        model_path = None
        final_path = os.path.join(model_save_dir, f"{model_name}_{algo_name}_final.zip")
        best_path = os.path.join(model_save_dir, f"{model_name}_{algo_name}_best.zip")

        if resume:
            if os.path.exists(final_path):
                model_path = final_path
            elif os.path.exists(best_path):
                model_path = best_path

            if model_path:
                print(f"[Agent Trainer] Resuming {algo_name.upper()} model from {model_path}...")
                custom_objects = {"learning_rate": lr_input}
                if algo_name == "ppo":
                    # اورراید کردن پارامترهای آموزش در فاز از سرگیری آموزش از روی تنظیمات
                    if config_data:
                        custom_objects["vf_coef"] = config_data.get("vf_coef", 0.8)
                        custom_objects["clip_range"] = config_data.get("clip_range", 0.25)
                        custom_objects["ent_coef"] = config_data.get("ent_coef", 0.015)
                        custom_objects["gamma"] = config_data.get("gamma", 0.98)
                        custom_objects["gae_lambda"] = config_data.get("gae_lambda", 0.95)
                        custom_objects["n_steps"] = config_data.get("n_steps", 2048)
                        custom_objects["batch_size"] = config_data.get("batch_size", 256)
                        print(f"[Agent Trainer] Resuming PPO with dynamic hyperparameters from config.")
                
                try:
                    if algo_name == "ppo":
                        if USING_RECURRENT:
                            model = RecurrentPPO.load(model_path, env=train_env, custom_objects=custom_objects)
                        else:
                            model = PPO.load(model_path, env=train_env, custom_objects=custom_objects)
                    elif algo_name == "sac":
                        model = SAC.load(model_path, env=train_env, custom_objects=custom_objects)
                    elif algo_name == "td3":
                        model = TD3.load(model_path, env=train_env, custom_objects=custom_objects)
                except Exception as e:
                    print(f"[Agent Trainer] Error loading {algo_name.upper()}: {e}. Training from scratch.")
                    model = None

        # Parse net_arch_size and lstm_hidden_size
        net_arch_size = config_data.get("net_arch_size", "small")
        if net_arch_size == "large":
            net_arch_ppo = dict(pi=[256, 256], vf=[256, 256])
            net_arch_sac_td3 = dict(pi=[256, 256], qf=[256, 256])
        elif net_arch_size == "medium":
            net_arch_ppo = dict(pi=[128, 128], vf=[128, 128])
            net_arch_sac_td3 = dict(pi=[128, 128], qf=[128, 128])
        else: # small
            net_arch_ppo = dict(pi=[64, 64], vf=[64, 64])
            net_arch_sac_td3 = dict(pi=[64, 64], qf=[64, 64])
            
        lstm_hidden_size = int(config_data.get("lstm_hidden_size", 128))

        if model is None:
            print(f"[Agent Trainer] Creating {algo_name.upper()} model from scratch with net_arch_size={net_arch_size}...")
            if algo_name == "ppo":
                hyperparams = {
                    "learning_rate": lr_input,
                    "n_steps": config_data.get("n_steps", 2048),
                    "batch_size": config_data.get("batch_size", 256),
                    "n_epochs": 4,
                    "gamma": config_data.get("gamma", 0.98),
                    "gae_lambda": config_data.get("gae_lambda", 0.95),
                    "clip_range": config_data.get("clip_range", 0.25),
                    "ent_coef": config_data.get("ent_coef", 0.015),
                    "vf_coef": config_data.get("vf_coef", 0.8),
                    "max_grad_norm": 0.5,
                    "verbose": 1,
                    "tensorboard_log": tb_log_dir if HAS_TENSORBOARD else None
                }
                if USING_RECURRENT:
                    policy = "MlpLstmPolicy"
                    policy_kwargs = dict(
                        lstm_hidden_size=lstm_hidden_size,
                        n_lstm_layers=1,
                        shared_lstm=True,
                        enable_critic_lstm=False,
                        net_arch=net_arch_ppo
                    )
                    model = RecurrentPPO(policy, train_env, policy_kwargs=policy_kwargs, **hyperparams)
                else:
                    policy = "MlpPolicy"
                    policy_kwargs = dict(net_arch=net_arch_ppo)
                    model = PPO(policy, train_env, policy_kwargs=policy_kwargs, **hyperparams)
            elif algo_name == "sac":
                hyperparams = {
                    "learning_rate": lr_input,
                    "buffer_size": 50000,  # محدود به حافظه VPS (1.8GB کل)
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.98,
                    "ent_coef": "auto",
                    "verbose": 1,
                    "tensorboard_log": tb_log_dir if HAS_TENSORBOARD else None
                }
                policy_kwargs = dict(net_arch=net_arch_sac_td3)
                model = SAC("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **hyperparams)
            elif algo_name == "td3":
                hyperparams = {
                    "learning_rate": lr_input,
                    "buffer_size": 50000,  # محدود به حافظه VPS (1.8GB کل)
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.98,
                    "policy_delay": 2,
                    "verbose": 1,
                    "tensorboard_log": tb_log_dir if HAS_TENSORBOARD else None
                }
                policy_kwargs = dict(net_arch=net_arch_sac_td3)
                model = TD3("MlpPolicy", train_env, policy_kwargs=policy_kwargs, **hyperparams)

        # 3. Setup WandB Logging and Callback
        callbacks_list = [eval_callback, progress_callback]
        
        use_wandb = os.getenv("USE_WANDB", "false").lower() in ["true", "1"]
        wandb_run = None
        
        if use_wandb:
            try:
                import wandb
                from wandb.integration.sb3 import WandbCallback
                import inspect
                
                tags = ["ensemble", "volume-bars", symbol_clean.lower()]
                run_config = hyperparams if 'hyperparams' in locals() else {}
                
                # Check if we are running under a sweep
                is_sweep = wandb.run is not None and (wandb.run.sweep_id is not None or os.getenv("WANDB_SWEEP_ID") is not None)
                
                if is_sweep:
                    print(f"[Agent Trainer] Active WandB Sweep detected. Reusing existing run: {wandb.run.name}")
                    sig = inspect.signature(WandbCallback.__init__)
                    callback_kwargs = {
                        "model_save_path": model_save_dir,
                        "model_save_freq": 10000,
                        "verbose": 2
                    }
                    if "save_model" in sig.parameters:
                        callback_kwargs["save_model"] = True
                    wandb_callback = WandbCallback(**callback_kwargs)
                    callbacks_list.append(wandb_callback)
                else:
                    if wandb.run is not None:
                        wandb.run.finish()
                    
                    wandb_run = wandb.init(
                        entity=os.getenv("WANDB_ENTITY", "ROBOCHILD"),
                        project=os.getenv("WANDB_PROJECT", f"robochild-{symbol_clean.lower()}"),
                        name=f"{model_name}_{algo_name}",
                        config=run_config,
                        sync_tensorboard=True,
                        monitor_gym=True,
                        save_code=True,
                        tags=tags,
                        notes=f"Sequential training of {algo_name.upper()} model for {symbol_clean}."
                    )
                    
                    sig = inspect.signature(WandbCallback.__init__)
                    callback_kwargs = {
                        "model_save_path": model_save_dir,
                        "model_save_freq": 10000,
                        "verbose": 2
                    }
                    if "save_model" in sig.parameters:
                        callback_kwargs["save_model"] = True
                    
                    wandb_callback = WandbCallback(**callback_kwargs)
                    callbacks_list.append(wandb_callback)
                    print(f"[Agent Trainer] Weights & Biases (WandB) run initialized for phase {algo_name}.")
            except Exception as w_err:
                print(f"[Agent Trainer] Failed to initialize WandB run: {w_err}. Proceeding without WandB callback.")

        # 4. Train Phase
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks_list,
            tb_log_name=f"{model_name}_{algo_name}",
            reset_num_timesteps=not resume
        )

        if progress_callback.was_aborted:
            if wandb_run is not None:
                try:
                    import wandb
                    wandb.run.finish()
                except Exception:
                    pass
            print(f"[Agent Trainer] Training was ABORTED during phase {phase['name']}. Preserving previous models.")
            return None, train_env

        # 4. Save best and final models
        model.save(final_path)
        
        best_temp_path = os.path.join(model_save_dir, "best_model.zip")
        if os.path.exists(best_temp_path):
            shutil.move(best_temp_path, best_path)
            print(f"[Agent Trainer] Best model saved to {best_path}")
            
        train_env.save(stats_path)
        print(f"[Agent Trainer] Model saved to {final_path}")
        print(f"[Agent Trainer] VecNormalize saved to {stats_path}")

        # Smart Model Versioning & Registry Promotion
        if use_wandb:
            try:
                import wandb
                active_run = wandb.run
                if active_run is not None:
                    print(f"[Agent Trainer] Registering model artifact for {algo_name.upper()}...")
                    artifact_name = f"model_{symbol_clean.lower()}_{algo_name.lower()}"
                    artifact = wandb.Artifact(
                        name=artifact_name, 
                        type="model", 
                        description=f"{algo_name.upper()} model weights and normalization stats for {symbol_clean}."
                    )
                    
                    # Add files
                    if os.path.exists(best_path):
                        artifact.add_file(best_path, name=os.path.basename(best_path))
                    elif os.path.exists(final_path):
                        artifact.add_file(final_path, name=os.path.basename(final_path))
                    
                    if os.path.exists(stats_path):
                        artifact.add_file(stats_path, name=os.path.basename(stats_path))
                    
                    # Define aliases
                    aliases = ["latest", "candidate"]
                    explained_var = None
                    if hasattr(model, "logger") and model.logger is not None:
                        explained_var = model.logger.name_to_value.get("train/explained_variance")
                    if explained_var is not None and explained_var >= 0.5:
                        aliases.extend(["best", "stable"])
                        
                    # Log artifact
                    active_run.log_artifact(artifact, aliases=aliases)
                    
                    # Link model to Model Registry
                    try:
                        registry_name = f"ROBOCHILD-{symbol_clean.upper()}-{algo_name.upper()}-Production"
                        active_run.link_artifact(artifact, f"ROBOCHILD/ROBOCHILD-SOL/model-registry/{registry_name}")
                        print(f"[Agent Trainer] Successfully linked model to registry: {registry_name}")
                    except Exception as reg_err:
                        print(f"[Agent Trainer] Model Registry linking warning: {reg_err}")
            except Exception as art_err:
                print(f"[Agent Trainer] Failed to register artifact: {art_err}")

        if wandb_run is not None:
            try:
                import wandb
                wandb.run.finish()
            except Exception:
                pass

        # Check gate threshold for Explained Variance if PPO just finished
        if algo_name == "ppo":
            explained_var = None
            if hasattr(model, "logger") and model.logger is not None:
                explained_var = model.logger.name_to_value.get("train/explained_variance")
            
            if explained_var is not None:
                # Save to config_path so sweep agent can log it
                try:
                    config_data["ppo_explained_variance"] = float(explained_var)
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=4)
                except Exception as e:
                    print(f"[Agent Trainer] Error saving explained variance to config: {e}")

            min_ev = config_data.get("min_explained_variance_for_sac", -2.0)
            if explained_var is not None:
                print(f"[Agent Trainer] PPO Explained Variance at end of training: {explained_var:.4f}")
                if explained_var < min_ev:
                    print(f"[Agent Trainer] PPO Explained Variance ({explained_var:.4f}) is below target threshold ({min_ev}). Skipping SAC and TD3 training phases as recommended.")
                    try:
                        with open(unified_progress_file, "w") as f:
                            json.dump({
                                "model_name": model_name,
                                "current_step": total_timesteps,
                                "total_steps": total_timesteps,
                                "percentage": 100.0,
                                "status": "completed (PPO only, SAC/TD3 skipped due to low Explained Variance)"
                            }, f)
                    except Exception:
                        pass
                    break

    # Complete the progress file at the end of all phases
    try:
        with open(unified_progress_file, "w") as f:
            json.dump({
                "model_name": model_name,
                "current_step": total_timesteps,
                "total_steps": total_timesteps,
                "percentage": 100.0,
                "status": "completed"
            }, f)
    except Exception:
        pass

    return model, train_env
