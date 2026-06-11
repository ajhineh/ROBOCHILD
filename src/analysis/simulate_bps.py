import sys
import os
import pandas as pd
import numpy as np
import json

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis.training_evaluator import UltraEnsembleEvaluator
from src.env.trading_env import FuturesTradingEnv
from src.core.rl_shared.model_loader import RLModelLoader

def run_discrete_simulation(symbol, df_data, evaluator, tp_bps, sl_bps, threshold=0.65, capital=4000.0, trade_capital_pct=25.0, leverage=15, max_drawdown_limit=None, hedge_mode=False):
    starting_price = df_data.iloc[0].get("futures_price", df_data.iloc[0].get("mid_price", 100.0))
    trade_allocation = capital * (trade_capital_pct / 100.0) 
    max_position_value = trade_allocation * leverage 
    max_inventory = max_position_value / starting_price
    
    env = FuturesTradingEnv(df_data, max_inventory=max_inventory, symbol=symbol.upper())
    obs, info = env.reset()
    
    portfolio_value = capital
    cash = capital
    
    ppo_model = evaluator.models.get("ppo")
    sac_model = evaluator.models.get("sac")
    td3_model = evaluator.models.get("td3")
    
    ppo_env = evaluator.envs.get("ppo")
    sac_env = evaluator.envs.get("sac")
    td3_env = evaluator.envs.get("td3")
    
    ppo_w = evaluator.ppo_w
    sac_w = evaluator.sac_w
    td3_w = evaluator.td3_w
    
    portfolio_history = [portfolio_value]
    active_positions = {"long": None, "short": None} 
    trades = []
    cooldown_steps = 0
    cost_ratio = 0.0012  # 12 BPS total roundtrip transaction cost (fees + slippage)
    cooldown_limit = 1
    
    # Daily drawdown variables
    daily_pnl = 0.0
    current_drawdown = 0.0
    last_date = None
    blocked_entries_count = 0
    
    for step in range(len(env.df) - 1):
        env.current_step = step
        
        row = env.df.iloc[env.current_step]
        close_price = row.get("futures_price", row.get("mid_price", 100.0))
        high_price = row.get("high", close_price)
        low_price = row.get("low", close_price)
        
        # Check daily reset
        row_time = row.name
        current_date = row_time.date()
        if last_date is None or current_date != last_date:
            daily_pnl = 0.0
            current_drawdown = 0.0
            last_date = current_date
            
        # Update net position in the environment for state observation
        net_pos = 0.0
        for s in ["long", "short"]:
            pos = active_positions[s]
            if pos is not None:
                side_multiplier = 1.0 if s == "long" else -1.0
                net_pos += side_multiplier * (pos["margin"] * leverage / close_price)
        env.position = net_pos
            
        obs = env._get_observation()
        env.obs_history.append(obs)
        stacked_obs = np.concatenate(list(env.obs_history))
        
        ppo_action = 0.0
        if ppo_model is not None:
            try:
                obs_ppo = RLModelLoader.normalize_observation(stacked_obs, ppo_env) if ppo_env else stacked_obs
                action, _ = ppo_model.predict(obs_ppo, deterministic=True)
                ppo_action = float(action[0])
            except Exception as e:
                pass
                
        sac_action = 0.0
        if sac_model is not None:
            try:
                obs_sac = RLModelLoader.normalize_observation(stacked_obs, sac_env) if sac_env else stacked_obs
                action, _ = sac_model.predict(obs_sac, deterministic=True)
                sac_action = float(action[0])
            except Exception as e:
                pass
                
        td3_action = 0.0
        if td3_model is not None:
            try:
                obs_td3 = RLModelLoader.normalize_observation(stacked_obs, td3_env) if td3_env else stacked_obs
                action, _ = td3_model.predict(obs_td3, deterministic=True)
                td3_action = float(action[0])
            except Exception as e:
                pass
                
        ppo_norm = float(np.tanh(ppo_action / 0.40))
        sac_norm = float(np.tanh(sac_action / 0.30))
        td3_norm = float(np.tanh(td3_action / 0.30))
        
        ensemble_action = ppo_w * ppo_norm + sac_w * sac_norm + td3_w * td3_norm
        
        if cooldown_steps > 0:
            cooldown_steps -= 1
            
        # 1. Monitor active positions exit
        for side in ["long", "short"]:
            pos = active_positions[side]
            if pos is not None:
                entry_p = pos["entry_price"]
                tp = pos["tp"]
                sl = pos["sl"]
                tp1 = pos["tp1"]
                tp1_hit = pos["tp1_hit"]
                margin = pos["margin"]
                
                # Check TP1 hit
                if not tp1_hit:
                    if (side == "long" and high_price >= tp1) or (side == "short" and low_price <= tp1):
                        pos["tp1_hit"] = True
                        exit_amount = margin * 0.5
                        pnl_pct = ((tp1 - entry_p) / entry_p) * leverage
                        if side == "short":
                            pnl_pct = -pnl_pct
                        pnl_usdt = exit_amount * (pnl_pct - cost_ratio)
                        portfolio_value += pnl_usdt
                        cash += exit_amount + pnl_usdt
                        
                        # Update daily PnL and drawdown
                        daily_pnl += pnl_usdt
                        if daily_pnl >= 0:
                            current_drawdown = 0.0
                        else:
                            current_drawdown = abs(daily_pnl)
                            
                        # Move SL to entry
                        pos["sl"] = entry_p
                        pos["margin"] = margin * 0.5
                        
                # Check SL hit
                sl_hit = False
                exit_price = close_price
                if side == "long" and low_price <= pos["sl"]:
                    sl_hit = True
                    exit_price = pos["sl"]
                elif side == "short" and high_price >= pos["sl"]:
                    sl_hit = True
                    exit_price = pos["sl"]
                    
                # Check TP2 hit
                tp_hit = False
                if not sl_hit:
                    if side == "long" and high_price >= tp:
                        tp_hit = True
                        exit_price = tp
                    elif side == "short" and low_price <= tp:
                        tp_hit = True
                        exit_price = tp
                        
                if sl_hit or tp_hit:
                    exit_amount = pos["margin"]
                    pnl_pct = ((exit_price - entry_p) / entry_p) * leverage
                    if side == "short":
                        pnl_pct = -pnl_pct
                    pnl_usdt = exit_amount * (pnl_pct - cost_ratio)
                    portfolio_value += pnl_usdt
                    cash += exit_amount + pnl_usdt
                    
                    # Update daily PnL and drawdown
                    daily_pnl += pnl_usdt
                    if daily_pnl >= 0:
                        current_drawdown = 0.0
                    else:
                        current_drawdown = abs(daily_pnl)
                        
                    trades.append({
                        "side": side,
                        "entry": entry_p,
                        "exit": exit_price,
                        "pnl_usdt": pnl_usdt,
                        "result": "TP" if tp_hit else "SL"
                    })
                    active_positions[side] = None
                    cooldown_steps = cooldown_limit
                    
        # 2. Trigger entry
        if cooldown_steps == 0 and abs(ensemble_action) >= threshold:
            proposed_side = "long" if ensemble_action > 0 else "short"
            
            can_enter = False
            if hedge_mode:
                # In Hedge Mode, we can open if the same side doesn't have an active trade
                can_enter = active_positions[proposed_side] is None
            else:
                # In One-way Mode, both sides must be empty (no trade at all)
                can_enter = (active_positions["long"] is None) and (active_positions["short"] is None)
                
            if can_enter:
                if max_drawdown_limit is not None and current_drawdown >= max_drawdown_limit:
                    blocked_entries_count += 1
                else:
                    margin_required = portfolio_value * (trade_capital_pct / 100.0)
                    if cash >= margin_required:
                        entry_p = close_price
                        tp_ratio = tp_bps / 10000.0
                        sl_ratio = sl_bps / 10000.0
                        tp1_ratio = (sl_bps * 1.5) / 10000.0
                        
                        if proposed_side == "long":
                            tp = entry_p * (1.0 + tp_ratio)
                            sl = entry_p * (1.0 - sl_ratio)
                            tp1 = entry_p * (1.0 + tp1_ratio)
                        else:
                            tp = entry_p * (1.0 - tp_ratio)
                            sl = entry_p * (1.0 + sl_ratio)
                            tp1 = entry_p * (1.0 - tp1_ratio)
                            
                        active_positions[proposed_side] = {
                            "side": proposed_side,
                            "entry_price": entry_p,
                            "tp": tp,
                            "sl": sl,
                            "tp1": tp1,
                            "tp1_hit": False,
                            "margin": margin_required
                        }
                        cash -= margin_required
                        
        env.current_step += 1
        portfolio_history.append(portfolio_value)
        
    total_return = (portfolio_value - capital) / capital
    win_trades = [t for t in trades if t["pnl_usdt"] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0.0
    
    return {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "portfolio_value": portfolio_value,
        "blocked_entries": blocked_entries_count
    }

if __name__ == "__main__":
    import sys
    symbol = sys.argv[1].lower() if len(sys.argv) > 1 else "bnb"
    
    for days_back in [5, 10]:
        print(f"\n==================================================")
        print(f"🚀 Running BPS Simulation for {symbol.upper()} over {days_back} days...")
        print(f"==================================================")
        
        evaluator = UltraEnsembleEvaluator(symbol=symbol, market_type="futures", days_back=days_back)
        evaluator.load_models()
        df_data = evaluator.fetch_evaluation_data()
        
        # 1. One-way mode, Original (TP=15, SL=10)
        print(f"\n--- CASE 1: One-way Mode | Original (TP=15, SL=10) ---")
        res_ow_orig = run_discrete_simulation(symbol, df_data, evaluator, tp_bps=15, sl_bps=10, hedge_mode=False)
        print(f"Total Return: {res_ow_orig['total_return']*100:+.2f}%")
        print(f"Number of Trades: {res_ow_orig['num_trades']}")
        print(f"Win Rate: {res_ow_orig['win_rate']*100:.1f}%")
        print(f"Final Balance: ${res_ow_orig['portfolio_value']:.2f}")
        
        # 2. One-way mode, Proposed (TP=40, SL=20)
        print(f"\n--- CASE 2: One-way Mode | Proposed (TP=40, SL=20) ---")
        res_ow_prop = run_discrete_simulation(symbol, df_data, evaluator, tp_bps=40, sl_bps=20, hedge_mode=False)
        print(f"Total Return: {res_ow_prop['total_return']*100:+.2f}%")
        print(f"Number of Trades: {res_ow_prop['num_trades']}")
        print(f"Win Rate: {res_ow_prop['win_rate']*100:.1f}%")
        print(f"Final Balance: ${res_ow_prop['portfolio_value']:.2f}")
        
        # 3. Hedge mode, Original (TP=15, SL=10)
        print(f"\n--- CASE 3: Hedge Mode | Original (TP=15, SL=10) ---")
        res_hd_orig = run_discrete_simulation(symbol, df_data, evaluator, tp_bps=15, sl_bps=10, hedge_mode=True)
        print(f"Total Return: {res_hd_orig['total_return']*100:+.2f}%")
        print(f"Number of Trades: {res_hd_orig['num_trades']}")
        print(f"Win Rate: {res_hd_orig['win_rate']*100:.1f}%")
        print(f"Final Balance: ${res_hd_orig['portfolio_value']:.2f}")
        
        # 4. Hedge mode, Proposed (TP=40, SL=20)
        print(f"\n--- CASE 4: Hedge Mode | Proposed (TP=40, SL=20) ---")
        res_hd_prop = run_discrete_simulation(symbol, df_data, evaluator, tp_bps=40, sl_bps=20, hedge_mode=True)
        print(f"Total Return: {res_hd_prop['total_return']*100:+.2f}%")
        print(f"Number of Trades: {res_hd_prop['num_trades']}")
        print(f"Win Rate: {res_hd_prop['win_rate']*100:.1f}%")
        print(f"Final Balance: ${res_hd_prop['portfolio_value']:.2f}")
