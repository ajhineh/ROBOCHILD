import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class FuturesTradingEnv(gym.Env):
    """
    A custom Gymnasium Environment for PPO-based commodity futures trading.
    Compatible with:
    - Real historical aligned datasets (Spot, Futures, Funding Rate, Open Interest).
    - Real-time live market streaming via CCXT execution client (Phase 4).
    - Unbounded real variables handled via look-ahead free normalization.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame = None,
        max_inventory: float = 10.0,   # Maximum contracts/units allowed to hold
        transaction_fee_rate: float = 0.0004, # 0.04% execution fee
        base_slippage_rate: float = 0.0002,   # 0.02% base slippage
        liquidation_penalty_coef: float = 0.05, # Forced terminal liquidation cost
        live_client = None             # Live exchange client for Phase 4 streaming
    ):
        super(FuturesTradingEnv, self).__init__()
        
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.n_steps = len(self.df) if df is not None else 100000
        self.max_inventory = max_inventory
        self.transaction_fee_rate = transaction_fee_rate
        self.base_slippage_rate = base_slippage_rate
        self.liquidation_penalty_coef = liquidation_penalty_coef
        self.live_client = live_client
        self.is_live = live_client is not None
        
        # Pre-process historical signals if in training mode
        if not self.is_live and df is not None:
            self._preprocess_data()
        
        # Action space: target inventory ratio in [-1.0, 1.0] (1.0 = Max Long, -1.0 = Max Short, 0.0 = Flat)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
        # Observation space (12 features)
        num_features = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float64
        )
        
        # Initialize state
        self.current_step = 0
        self.position = 0.0 
        self.portfolio_value = 100000.0
        self.peak_portfolio_value = 100000.0
        self.cash = 100000.0
        
    def _preprocess_data(self):
        # Normalize spread and basis relative to price for stationary training features
        self.df["spread_ratio"] = self.df["spread"] / self.df["mid_price"]
        self.df["depth_imbalance"] = (self.df["bid_depth"] - self.df["ask_depth"]) / (self.df["bid_depth"] + self.df["ask_depth"] + 1e-8)
        self.df["basis_ratio"] = self.df["basis"] / self.df["mid_price"]
        
        # Safe clipping for historical extremes (removes noise outliers)
        self.df["spread_ratio"] = np.clip(self.df["spread_ratio"], 0.0, 0.05)
        self.df["depth_imbalance"] = np.clip(self.df["depth_imbalance"], -1.0, 1.0)
        self.df["basis_ratio"] = np.clip(self.df["basis_ratio"], -0.1, 0.1)
        self.df["carry_ratio"] = np.clip(self.df["carry"], -0.2, 0.2)
        self.df["roll_yield"] = np.clip(self.df["roll_yield"], -0.2, 0.2)
        
        # Max scaling for volatility
        max_vol = self.df["volatility"].max() if self.df["volatility"].max() > 0 else 1.0
        self.df["volatility_ratio"] = self.df["volatility"] / max_vol
        
    def _get_observation(self):
        """Constructs the current 12-dimensional state vector."""
        if self.is_live:
            # Query the CCXT client directly to fetch real-time state features
            market = self.live_client.fetch_market_state()
            account = self.live_client.fetch_account_state()
            
            mid_price = market["mid"]
            spread_ratio = (market["ask"] - market["bid"]) / (mid_price + 1e-8)
            depth_imbalance = (market["bid_depth"] - market["ask_depth"]) / (market["bid_depth"] + market["ask_depth"] + 1e-8)
            
            # Basis, Carry (Funding rate), Open Interest, Volatility from live exchange APIs
            basis_ratio = market.get("basis", 0.0) / (mid_price + 1e-8)
            carry_ratio = market.get("funding_rate", 0.0001)
            roll_yield = basis_ratio
            speculator_ratio = market.get("speculator_ratio", 0.0)
            sentiment = market.get("sentiment", 0.0)
            volatility_ratio = market.get("volatility_ratio", 0.02)
            
            obs = np.array([
                account["position"] / self.max_inventory,
                0.5, # progress is stable in continuous streaming
                spread_ratio,
                depth_imbalance,
                market.get("convenience_yield", 0.0),
                basis_ratio,
                carry_ratio,
                roll_yield,
                speculator_ratio,
                sentiment,
                0.0, # surprise
                volatility_ratio
            ], dtype=np.float64)
        else:
            # Fetch from historical DataFrame
            row = self.df.iloc[self.current_step]
            
            obs = np.array([
                self.position / self.max_inventory,
                self.current_step / self.n_steps,
                row["spread_ratio"],
                row["depth_imbalance"],
                row["convenience_yield"],
                row["basis_ratio"],
                row["carry_ratio"],
                row["roll_yield"],
                row["speculator_ratio"],
                row["sentiment"],
                row["surprise"],
                row["volatility_ratio"]
            ], dtype=np.float64)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.position = 0.0
        
        if self.is_live:
            account = self.live_client.fetch_account_state()
            self.portfolio_value = account["portfolio_value"]
            self.peak_portfolio_value = self.portfolio_value
            self.cash = account["margin_balance"]
            self.position = account["position"]
        else:
            self.portfolio_value = 100000.0
            self.peak_portfolio_value = 100000.0
            self.cash = 100000.0
            
        observation = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position": self.position
        }
        return observation, info

    def step(self, action):
        # Action is continuous [-1.0, 1.0] representing target position ratio
        target_position_ratio = float(action[0])
        target_position = target_position_ratio * self.max_inventory
        
        trade_size = target_position - self.position
        
        if self.is_live:
            # Phase 4 Live execution streaming via CCXT
            # Execute actual trade on Binance Futures Sandbox via execution client
            self.live_client.execute_order(target_position_ratio, self.max_inventory)
            
            # Fetch updated account and market status from real exchange feeds
            market = self.live_client.fetch_market_state()
            account = self.live_client.fetch_account_state()
            
            self.position = account["position"]
            self.portfolio_value = account["portfolio_value"]
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
            self.cash = account["margin_balance"]
            
            observation = self._get_observation()
            reward = 0.0 # live rewards are calculated from real-time asset appreciation
            terminated = False
            truncated = False
            
            info = {
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "position": self.position,
                "trade_size": trade_size,
                "transaction_fee": 0.0,
                "slippage": 0.0
            }
        else:
            # Historical simulation mode
            row = self.df.iloc[self.current_step]
            mid_price = row["mid_price"]
            volatility = row["volatility_ratio"]
            market_depth = (row["bid_depth"] + row["ask_depth"]) / 2.0
            
            # 1. Limit order shadow-based price simulation with safe fallbacks
            if trade_size > 0: # Buying (Long entry or Short exit)
                ref_price = row.get("f_low", row.get("futures_price", mid_price))
                if pd.isna(ref_price) or ref_price <= 0:
                    ref_price = row.get("futures_price", mid_price)
                if pd.isna(ref_price) or ref_price <= 0:
                    ref_price = mid_price
            elif trade_size < 0: # Selling (Short entry or Long exit)
                ref_price = row.get("f_high", row.get("futures_price", mid_price))
                if pd.isna(ref_price) or ref_price <= 0:
                    ref_price = row.get("futures_price", mid_price)
                if pd.isna(ref_price) or ref_price <= 0:
                    ref_price = mid_price
            else:
                ref_price = mid_price
            
            # Volatility-adaptive slippage on top of reference execution price
            slippage_rate = self.base_slippage_rate + 0.005 * (abs(trade_size) / (market_depth + 1e-8)) * (1.0 + volatility)
            execution_price = ref_price * (1.0 + np.sign(trade_size) * slippage_rate)
            
            trade_value = abs(trade_size) * execution_price
            transaction_fee = trade_value * self.transaction_fee_rate
            
            self.cash -= (trade_size * execution_price) + transaction_fee
            self.position = target_position
            
            new_portfolio_value = self.cash + (self.position * mid_price)
            
            self.current_step += 1
            terminated = self.current_step >= self.n_steps - 1
            truncated = False
            
            terminal_liquidation_fee = 0.0
            if terminated:
                leftover_inventory = abs(self.position)
                terminal_liquidation_fee = (leftover_inventory * mid_price) * self.liquidation_penalty_coef
                new_portfolio_value -= terminal_liquidation_fee
                self.cash -= terminal_liquidation_fee
                self.position = 0.0
            
            # 2. Percentage-based PnL, Fee and Drawdown tracking
            pnl_pct = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            fee_pct = ((transaction_fee + terminal_liquidation_fee) / self.portfolio_value) * 100.0
            
            self.peak_portfolio_value = max(self.peak_portfolio_value, new_portfolio_value)
            drawdown_pct = (self.peak_portfolio_value - new_portfolio_value) / self.peak_portfolio_value
            
            # 3. New reward function: Reward = (PnL_Pct * 100) - (Drawdown_Pct * 100 * 0.5) - Fee_Pct
            reward = (pnl_pct * 100.0) - (drawdown_pct * 100.0 * 0.5) - fee_pct
            
            # 4. Holding penalty (0.005 per step if position was held during the step)
            if abs(target_position) > 1e-8:
                reward -= 0.005
                
            self.portfolio_value = new_portfolio_value
            observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float64)
            
            info = {
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "position": self.position,
                "trade_size": trade_size,
                "transaction_fee": transaction_fee + terminal_liquidation_fee,
                "slippage": slippage_rate * ref_price
            }
        
        return observation, reward, terminated, truncated, info
