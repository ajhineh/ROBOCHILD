import os
import sys
import numpy as np

# Add project path to sys.path
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.core.rl_shared.model_loader import RLModelLoader

try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
except ImportError:
    from stable_baselines3 import PPO
    USING_RECURRENT = False

def run_ensemble_scenario(models_dict, symbol, scenario_name, obs_12):
    """
    اجرای شبیه‌سازی سناریو روی مدل‌های Ensemble (PPO + SAC + TD3)
    """
    print(f"\n  [{scenario_name}]")
    
    # تبدیل بردار ۱۲ بعدی به ۱۲۰ بعدی با انباشت ۱۰ فریم
    obs_120 = np.concatenate([obs_12] * 10)
    
    ppo_model, ppo_env = models_dict.get("ppo", (None, None))
    sac_model, sac_env = models_dict.get("sac", (None, None))
    td3_model, td3_env = models_dict.get("td3", (None, None))
    
    ppo_action = 0.0
    sac_action = 0.0
    td3_action = 0.0
    
    # ۱. پیش‌بینی PPO
    if ppo_model is not None:
        try:
            obs_ppo = RLModelLoader.normalize_observation(obs_120, ppo_env)
            if USING_RECURRENT and hasattr(ppo_model, "policy") and "Lstm" in type(ppo_model.policy).__name__:
                action, _ = ppo_model.predict(np.expand_dims(obs_ppo, axis=0), state=None, episode_start=np.array([True]), deterministic=True)
                ppo_action = float(action.item())
            else:
                action, _ = ppo_model.predict(np.expand_dims(obs_ppo, axis=0), deterministic=True)
                ppo_action = float(action.item())
        except Exception as e:
            print(f"    - Error predicting PPO: {e}")
            ppo_action = 0.0
            
    # ۲. پیش‌بینی SAC
    if sac_model is not None:
        try:
            obs_sac = RLModelLoader.normalize_observation(obs_120, sac_env)
            action, _ = sac_model.predict(np.expand_dims(obs_sac, axis=0), deterministic=True)
            sac_action = float(action.item())
        except Exception as e:
            print(f"    - Error predicting SAC: {e}")
            sac_action = 0.0
            
    # ۳. پیش‌بینی TD3
    if td3_model is not None:
        try:
            obs_td3 = RLModelLoader.normalize_observation(obs_120, td3_env)
            action, _ = td3_model.predict(np.expand_dims(obs_td3, axis=0), deterministic=True)
            td3_action = float(action.item())
        except Exception as e:
            print(f"    - Error predicting TD3: {e}")
            td3_action = 0.0
            
    # نرمال‌سازی ریاضی تصمیمات
    ppo_norm = float(np.tanh(ppo_action / 0.40))
    sac_norm = float(np.tanh(sac_action / 0.30))
    td3_norm = float(np.tanh(td3_action / 0.30))
    
    # فرآیند ترکیب و وزن‌دهی
    w_ppo, w_sac, w_td3 = 0.45, 0.30, 0.25
    ensemble_action = w_ppo * ppo_norm + w_sac * sac_norm + w_td3 * td3_norm
    
    direction = "LONG (BUY)" if ensemble_action > 0 else "SHORT (SELL)"
    strength = abs(ensemble_action)
    status = "APPROVED (Trade opens)" if strength >= 0.60 else "FILTERED (Noise - No trade)"
    
    print(f"    - PPO Prediction: raw={ppo_action:+.4f}, norm={ppo_norm:+.4f} (weight={w_ppo:.2f})")
    print(f"    - SAC Prediction: raw={sac_action:+.4f}, norm={sac_norm:+.4f} (weight={w_sac:.2f})")
    print(f"    - TD3 Prediction: raw={td3_action:+.4f}, norm={td3_norm:+.4f} (weight={w_td3:.2f})")
    print(f"    - Ensemble Action: {ensemble_action:+.4f}")
    print(f"    - Predicted Direction: {direction}")
    print(f"    - Signal Strength: {strength:.2%}")
    print(f"    - Ensemble Decision: {status}")
    print("-" * 60)

def run_old_scenario(model, env, scenario_name, obs_12):
    """
    اجرای شبیه‌سازی سناریو روی مدل قدیمی تک عاملی PPO
    """
    print(f"\n  [{scenario_name}]")
    obs_batched = np.expand_dims(obs_12, axis=0)
    
    if env is not None:
        obs_normalized = env.normalize_obs(obs_batched)
    else:
        obs_normalized = obs_batched
        
    action_ratio = 0.0
    try:
        if USING_RECURRENT and hasattr(model, "policy") and "Lstm" in type(model.policy).__name__:
            action, _ = model.predict(obs_normalized, state=None, episode_start=np.array([True]), deterministic=True)
            action_ratio = float(action.item())
        else:
            action, _ = model.predict(obs_normalized, deterministic=True)
            action_ratio = float(action.item())
    except Exception as e:
        print(f"    - Error predicting Old PPO: {e}")
        return
        
    direction = "LONG (BUY)" if action_ratio > 0 else "SHORT (SELL)"
    strength = abs(action_ratio)
    status = "APPROVED (Trade opens)" if strength >= 0.25 else "FILTERED (Noise - No trade)"
    
    print(f"    - Output Action Ratio: {action_ratio:+.4f}")
    print(f"    - Predicted Direction: {direction}")
    print(f"    - Signal Strength: {strength:.2%}")
    print(f"    - Trigger Status: {status}")
    print("-" * 60)

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
        print("No trained models (.zip) found in models folder.")
        return
        
    print(f"Found {len(zip_files)} model files in models directory. Running simulations...")
    print("=" * 70)
    
    # شناسایی و گروه بندی نمادها
    symbols = set()
    for f in zip_files:
        for part in f.replace(".zip", "").split("_"):
            if part in ["popcat", "bome", "btc", "eth"]:
                symbols.add(part)
                break
                
    if not symbols:
        symbols.add("default")
        
    for symbol_clean in sorted(symbols):
        print(f"\n[ANALYZING SYMBOL] {symbol_clean.upper()}")
        print("=" * 75)
        
        loader = RLModelLoader(models_dir=models_dir)
        models_dict = loader.load_ensemble_models(symbol_clean)
        
        ppo_model, ppo_env = models_dict.get("ppo", (None, None))
        sac_model, sac_env = models_dict.get("sac", (None, None))
        td3_model, td3_env = models_dict.get("td3", (None, None))
        
        is_ensemble = ppo_model is not None and (sac_model is not None or td3_model is not None)
        
        if ppo_model is None:
            # تلاش برای بارگذاری مدل قدیمی
            ppo_model, ppo_env = loader.load_ppo_model(symbol_clean)
            is_ensemble = False
            
        if ppo_model is None:
            print(f"❌ Failed to load any model for {symbol_clean}.")
            continue
            
        if is_ensemble:
            print("🟢 Ensemble Architecture Detected!")
            print(f"  - PPO (LSTM): {'Active' if ppo_model is not None else 'Inactive'}")
            print(f"  - SAC (MLP): {'Active' if sac_model is not None else 'Inactive'}")
            print(f"  - TD3 (MLP): {'Active' if td3_model is not None else 'Inactive'}")
        else:
            print("🟡 Old Single PPO Architecture Detected!")
            print(f"  - PPO Network: {type(ppo_model.policy).__name__}")
            print(f"  - VecNormalize: {'Active' if ppo_env is not None else 'Inactive'}")
            
        print("=" * 75)
        
        # Scenarios (12 features vector)
        scenarios = {
            "Scenario 1: Calm Flat Market (No Momentum)": 
                np.array([0.0, 0.5, 0.001, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.5, 0.0, 0.0, 0.02], dtype=np.float64),
            "Scenario 2: Strong Bullish Demand (Whale Buying, Positive OBI)": 
                np.array([0.0, 0.5, 0.001, 0.85, 0.0, 0.0, 0.0001, 0.0, 0.7, 0.6, 0.0, 0.08], dtype=np.float64),
            "Scenario 3: Strong Bearish Supply (Whale Selling, Negative OBI)": 
                np.array([0.0, 0.5, 0.001, -0.85, 0.0, 0.0, 0.0001, 0.0, 0.3, -0.6, 0.0, 0.08], dtype=np.float64),
            "Scenario 4: High Volatility Panic (Negative Sentiment, High Volatility)": 
                np.array([0.0, 0.5, 0.005, -0.3, 0.0, 0.0, 0.0005, 0.0, 0.4, -0.8, 0.2, 0.35], dtype=np.float64),
        }
        
        for name, obs in scenarios.items():
            if is_ensemble:
                run_ensemble_scenario(models_dict, symbol_clean, name, obs)
            else:
                run_old_scenario(ppo_model, ppo_env, name, obs)
                
        print("\n" + "#" * 75)

if __name__ == "__main__":
    main()
