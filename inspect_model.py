import os
import sys
import numpy as np

# Add project path to sys.path
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

from src.core.rl_shared.model_loader import RLModelLoader

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
except ImportError:
    from stable_baselines3 import PPO
    USING_RECURRENT = False

def run_scenario(model, env, scenario_name, obs_vector):
    """Run prediction on a custom scenario and print the results"""
    obs_batched = np.expand_dims(obs_vector, axis=0)
    
    # Normalize if env statistics are available
    if env is not None:
        obs_normalized = env.normalize_obs(obs_batched)
    else:
        obs_normalized = obs_batched
        
    action_ratio = 0.0
    try:
        if USING_RECURRENT and hasattr(model, "policy") and "Lstm" in type(model.policy).__name__:
            episode_start = np.array([True])
            action, _ = model.predict(
                obs_normalized,
                state=None,
                episode_start=episode_start,
                deterministic=True
            )
            action_ratio = float(action.item())
        else:
            action, _ = model.predict(obs_normalized, deterministic=True)
            action_ratio = float(action.item())
    except Exception as e:
        print(f"Error running prediction: {e}")
        return
        
    direction = "LONG (BUY)" if action_ratio > 0 else "SHORT (SELL)"
    strength = abs(action_ratio)
    status = "APPROVED (Trade opens)" if strength >= 0.60 else "FILTERED (Noise - No trade)"
    
    print(f"  [{scenario_name}]")
    print(f"    - Output Action Ratio: {action_ratio:+.4f}")
    print(f"    - Predicted Direction: {direction}")
    print(f"    - Signal Strength: {strength:.2%}")
    print(f"    - Trigger Status: {status}")
    print("-" * 50)

def main():
    print("=" * 70)
    print("      ROBOCHILD Neural Network Model Analyzer & Simulator")
    print("=" * 70)
    
    models_dir = os.path.join(project_dir, "models")
    if not os.path.exists(models_dir):
        print(f"Error: models directory not found at {models_dir}")
        return
        
    zip_files = [f for f in os.listdir(models_dir) if f.endswith(".zip") and "progress" not in f]
    if not zip_files:
        print("No trained PPO models (.zip) found in models folder.")
        return
        
    print(f"Found {len(zip_files)} model files in models directory. Running simulations...")
    print("=" * 70)
    
    for filename in zip_files:
        print(f"\n[LOADING MODEL] {filename}")
        print("=" * 50)
        
        symbol_clean = "default"
        for part in filename.replace(".zip", "").split("_"):
            if part in ["popcat", "bome", "btc", "eth"]:
                symbol_clean = part
                break
                
        loader = RLModelLoader(models_dir=models_dir)
        model, env = loader.load_ppo_model(symbol_clean)
        
        if model is None:
            print("Failed to load model weights.")
            continue
            
        print(f" - Algorithm Type: {type(model).__name__}")
        print(f" - Policy Network: {type(model.policy).__name__}")
        print(f" - Normalization Layer (VecNormalize): {'Active' if env is not None else 'Inactive'}")
        print("=" * 50)
        
        # Scenarios (12 features vector)
        # 1. Flat market
        flat_obs = np.array([0.0, 0.5, 0.001, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.5, 0.0, 0.0, 0.02], dtype=np.float64)
        run_scenario(model, env, "Scenario 1: Calm Flat Market (No Momentum)", flat_obs)
        
        # 2. Bullish whale buyer (High positive OBI + high volatility)
        bullish_obs = np.array([0.0, 0.5, 0.001, 0.85, 0.0, 0.0, 0.0001, 0.0, 0.7, 0.6, 0.0, 0.08], dtype=np.float64)
        run_scenario(model, env, "Scenario 2: Strong Bullish Demand (Whale Buying, Positive OBI)", bullish_obs)
        
        # 3. Bearish whale seller (High negative OBI + high volatility)
        bearish_obs = np.array([0.0, 0.5, 0.001, -0.85, 0.0, 0.0, 0.0001, 0.0, 0.3, -0.6, 0.0, 0.08], dtype=np.float64)
        run_scenario(model, env, "Scenario 3: Strong Bearish Supply (Whale Selling, Negative OBI)", bearish_obs)
        
        # 4. High Volatility panic
        panic_obs = np.array([0.0, 0.5, 0.005, -0.3, 0.0, 0.0, 0.0005, 0.0, 0.4, -0.8, 0.2, 0.35], dtype=np.float64)
        run_scenario(model, env, "Scenario 4: High Volatility Panic (Negative Sentiment, High Volatility)", panic_obs)
        
        print("\n" + "#" * 60)

if __name__ == "__main__":
    main()
