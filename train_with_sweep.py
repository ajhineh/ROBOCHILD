import os
import sys
import argparse
import numpy as np
import pandas as pd
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environmental variables manually from .env if present
def load_env():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip quotes if present
                    val_str = val.strip().strip('"').strip("'")
                    os.environ[key.strip()] = val_str

load_env()

# Import project modules
from src.env import fetch_real_binance_data
from src.agent.trainer import train_agent
from src.analysis.training_evaluator import UltraEnsembleEvaluator

def run_sweep_trial(symbol="sol", days_back=15, timesteps=30000, is_test_run=False):
    import wandb
    
    # 1. Initialize WandB run
    run = wandb.init()
    config = wandb.config
    
    print(f"\n==================================================")
    print(f"Starting Sweep Trial with parameters: {dict(config)}")
    print(f"==================================================")
    
    # 2. Extract hyperparameters from WandB config
    vf_coef = config.get("vf_coef", 0.8)
    ent_coef = config.get("ent_coef", 0.015)
    n_steps = config.get("n_steps", 2048)
    batch_size = config.get("batch_size", 256)
    gamma = config.get("gamma", 0.98)
    gae_lambda = config.get("gae_lambda", 0.95)
    clip_range = config.get("clip_range", 0.25)
    learning_rate = config.get("learning_rate", "linear_0.0003")
    
    ppo_weight = config.get("ppo_weight", 0.50)
    sac_weight = config.get("sac_weight", 0.30)
    td3_weight = config.get("td3_weight", 0.20)
    
    early_stopping_patience = config.get("early_stopping_patience", 3)
    eval_freq = config.get("eval_freq", 2000)
    
    ensemble_threshold = config.get("ensemble_threshold", 0.62)
    take_profit_bps = config.get("take_profit_bps", 25)
    stop_loss_bps = config.get("stop_loss_bps", 12)
    net_arch_size = config.get("net_arch_size", "small")
    lstm_hidden_size = config.get("lstm_hidden_size", 128)
    
    # 3. Create hyperparameter override dictionary for training
    override_hyperparams = {
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ppo_weight": ppo_weight,
        "sac_weight": sac_weight,
        "td3_weight": td3_weight,
        "early_stopping_patience": early_stopping_patience,
        "eval_freq": eval_freq,
        "ensemble_threshold": ensemble_threshold,
        "take_profit_bps": take_profit_bps,
        "stop_loss_bps": stop_loss_bps,
        "net_arch_size": net_arch_size,
        "lstm_hidden_size": lstm_hidden_size,
    }
    
    # Save the parameters temporarily to symbol config file so the evaluator can read them
    symbol_clean = symbol.lower().split('/')[0]
    config_path = f"models/config_{symbol_clean}.json"
    os.makedirs("models", exist_ok=True)
    
    existing_config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except Exception:
            pass
            
    existing_config.update(override_hyperparams)
    existing_config["frame_stack"] = 8  # Use the optimal stack size of 8
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(existing_config, f, indent=4)
        print(f"Updated configuration saved to {config_path}")
    except Exception as e:
        print(f"Failed to write config: {e}")
    
    # 4. Fetch Binance historical data
    binance_symbol = f"{symbol.upper()}/USDT:USDT" if ":" not in symbol else symbol.upper()
    is_meme = symbol_clean in ["bome", "popcat"]
    timeframe = "1m" if is_meme else "5m"
    
    print(f"Fetching {days_back} days of data for {binance_symbol} ({timeframe} timeframe)...")
    df = fetch_real_binance_data(
        symbol=binance_symbol,
        timeframe=timeframe,
        days_back=days_back
    )
    
    if df is None or len(df) < 500:
        print("Error: Insufficient training data retrieved.")
        return
        
    # Split into 80% Train, 20% Validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Dataset split: Train={len(train_df)} candles, Validation={len(val_df)} candles")
    
    # 5. Run Training
    model_name = f"ppo_volume_bars_child_{symbol_clean}"
    print(f"Starting agent training ({timesteps} total steps)...")
    
    train_agent(
        train_df=train_df,
        val_df=val_df,
        total_timesteps=timesteps,
        model_save_dir="models",
        tb_log_dir="tb_logs",
        model_name=model_name,
        resume=True,
        learning_rate_val=learning_rate,
        override_hyperparams=override_hyperparams
    )
    
    # 6. Evaluate on Validation Set using UltraEnsembleEvaluator
    print("Evaluating models on validation set...")
    try:
        evaluator = UltraEnsembleEvaluator(
            symbol=symbol_clean,
            base_path=".",
            market_type="futures",
            days_back=days_back
        )
        # Load the newly trained models
        evaluator.load_models()
        # Run backtest on the hold-out validation set
        backtest_res = evaluator.run_backtest_on_env(val_df, use_volume_bars=True)
        
        # Extract metrics
        val_sharpe = float(backtest_res.get("sharpe_ratio", 0.0))
        val_win_rate = float(backtest_res.get("win_rate", 0.0)) * 100.0
        val_return_pct = float(backtest_res.get("total_return", 0.0)) * 100.0
        val_profit_factor = float(backtest_res.get("profit_factor", 0.0))
        val_max_drawdown = float(backtest_res.get("max_drawdown", 0.0)) * 100.0
        
        # Load explained variance if computed during PPO training
        ppo_explained_variance = -2.0
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)
                    ppo_explained_variance = float(saved_cfg.get("ppo_explained_variance", -2.0))
            except Exception:
                pass

        print(f"Evaluation Results:")
        print(f"  - Sharpe Ratio: {val_sharpe:.4f}")
        print(f"  - Return: {val_return_pct:.2f}%")
        print(f"  - Win Rate: {val_win_rate:.1f}%")
        print(f"  - Profit Factor: {val_profit_factor:.2f}")
        print(f"  - Max Drawdown: {val_max_drawdown:.2f}%")
        print(f"  - PPO Explained Variance: {ppo_explained_variance:.4f}")
        
        # 7. Log metrics back to WandB Sweep Agent
        wandb.log({
            "val_sharpe": val_sharpe,
            "val_win_rate": val_win_rate,
            "val_return_pct": val_return_pct,
            "val_profit_factor": val_profit_factor,
            "val_max_drawdown": val_max_drawdown,
            "ppo_explained_variance": ppo_explained_variance,
            "val_explained_variance": ppo_explained_variance,
            "val_eval_mean_reward": val_return_pct
        })
    except Exception as eval_err:
        print(f"Error during evaluation: {eval_err}")
        # Log negative metrics on failure to prevent sweep agent choosing bad parameters
        wandb.log({
            "val_sharpe": -10.0,
            "val_win_rate": 0.0,
            "val_return_pct": -100.0,
            "val_profit_factor": 0.0,
            "ppo_explained_variance": -2.0,
            "val_explained_variance": -2.0,
            "val_eval_mean_reward": -100.0
        })

    # Finish run
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROBOCHILD WandB Sweeps Trainer")
    parser.add_argument("--symbol", type=str, default="sol", help="Symbol name (e.g. sol)")
    parser.add_argument("--days-back", type=int, default=15, help="Days of historical training data")
    parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps of training steps")
    parser.add_argument("--test-run", action="store_true", help="Run a quick local single-trial verification")
    args, unknown = parser.parse_known_args()
    
    # Check if WandB is installed
    try:
        import wandb
    except ImportError:
        print("Error: wandb is not installed. Run 'pip install wandb'")
        sys.exit(1)
        
    if args.test_run:
        print("Running a local test trial of train_with_sweep.py...")
        # Mock sweep configuration
        test_config = {
            "vf_coef": 0.8,
            "ent_coef": 0.015,
            "n_steps": 1024,
            "batch_size": 128,
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "clip_range": 0.25,
            "learning_rate": "constant_0.00015",
            "ppo_weight": 0.5,
            "sac_weight": 0.3,
            "td3_weight": 0.2
        }
        
        # Initialize a test run locally
        run = wandb.init(project="robochild-sweep-test", config=test_config, mode="offline")
        run_sweep_trial(symbol=args.symbol, days_back=args.days_back, timesteps=5000, is_test_run=True)
    else:
        # Run using standard wandb sweep agent wrapper
        # The agent dynamically injects parameters and calls the function
        run_sweep_trial(symbol=args.symbol, days_back=args.days_back, timesteps=args.timesteps)
