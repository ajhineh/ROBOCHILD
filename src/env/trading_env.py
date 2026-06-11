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
        liquidation_penalty_coef: float = 0.0008, # Reduced to 0.08% to match normal close transaction fee + slippage proxy
        live_client = None,            # Live exchange client for Phase 4 streaming
        symbol: str = None             # Symbol name
    ):
        super(FuturesTradingEnv, self).__init__()
        
        self.symbol = symbol
        self.df = self._build_volume_bars(df.copy()) if df is not None else pd.DataFrame()
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
            
        # Frame stacking configurations
        from collections import deque
        self.n_stack = 10
        self.obs_history = deque(maxlen=self.n_stack)
        
        # Action space: target inventory ratio in [-1.0, 1.0] (1.0 = Max Long, -1.0 = Max Short, 0.0 = Flat)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
        # Observation space (12 features * 10 stacked frames = 120 features)
        num_features = 12 * self.n_stack
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
        
        # Triple-Barrier Method variables
        self.tp_bps = 25
        self.sl_bps = 12
        if self.symbol:
            config_path = f"models/config_{self.symbol.lower()}.json"
            import os, json
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        self.tp_bps = cfg.get("take_profit_bps", 25)
                        self.sl_bps = cfg.get("stop_loss_bps", 12)
                except Exception:
                    pass
        
        self.entry_price = 0.0
        self.tp_price = 0.0
        self.sl_price = 0.0
        self.trade_steps = 0
        self.max_holding_steps = 48  # 4 hours if 5-min candles
        self.live_prices = []
        
        
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

        # Apply fractional differentiation on mid_price to obtain a stationary price feature
        from src.analysis.frac_diff import find_optimal_d, frac_diff_ffd
        series = self.df["mid_price"]
        opt_d = find_optimal_d(series)
        
        # Save to config if symbol is set
        if self.symbol:
            config_path = f"models/config_{self.symbol.lower()}.json"
            import os, json
            cfg = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    pass
            cfg["frac_diff_d"] = opt_d
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=4)
            except Exception:
                pass
        
        price_frac = frac_diff_ffd(series, opt_d)
        self.df["price_frac"] = price_frac.bfill().ffill()
        
    def _get_observation(self):
        """Constructs the current 12-dimensional state vector."""
        if self.is_live:
            # Query the CCXT client directly to fetch real-time state features
            market = self.live_client.fetch_market_state()
            account = self.live_client.fetch_account_state()
            
            mid_price = market["mid"]
            self.live_prices.append(mid_price)
            if len(self.live_prices) > 1000:
                self.live_prices.pop(0)
            
            # load d from config
            frac_diff_d = 0.35
            if self.symbol:
                config_path = f"models/config_{self.symbol.lower()}.json"
                import os, json
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                            frac_diff_d = cfg.get("frac_diff_d", 0.35)
                    except Exception:
                        pass
            from src.analysis.frac_diff import get_latest_frac_diff
            price_frac = get_latest_frac_diff(self.live_prices, frac_diff_d)
            
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
                price_frac, # fractionally differentiated price feature
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
                row["price_frac"], # fractionally differentiated price feature
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
            
        # Reset observation history buffer
        obs = self._get_observation()
        self.obs_history.clear()
        for _ in range(self.n_stack):
            self.obs_history.append(obs)
            
        stacked_obs = np.concatenate(list(self.obs_history))
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position": self.position
        }
        return stacked_obs, info

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
            
            obs = self._get_observation()
            self.obs_history.append(obs)
            stacked_obs = np.concatenate(list(self.obs_history))
            
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
            
            # 1. Price execution simulation on current candle close price (Removes look-ahead bias f_low/f_high)
            ref_price = row.get("futures_price", mid_price)
            if pd.isna(ref_price) or ref_price <= 0:
                ref_price = mid_price
            
            # ردیابی و بررسی لمس مرزهای سه‌گانه دِ پرادو (Triple-Barrier Method)
            barrier_hit = None
            
            if abs(self.position) > 1e-8:
                # پوزیشن فعال است: شمارش گام‌ها و بررسی حد سود/ضرر کندل جاری
                self.trade_steps += 1
                high_p = row.get("high", mid_price)
                low_p = row.get("low", mid_price)
                
                if self.position > 0: # Long
                    if low_p <= self.sl_price:
                        barrier_hit = "SL"
                        target_position = 0.0
                        ref_price = self.sl_price
                    elif high_p >= self.tp_price:
                        barrier_hit = "TP"
                        target_position = 0.0
                        ref_price = self.tp_price
                    elif self.trade_steps >= self.max_holding_steps:
                        barrier_hit = "Time"
                        target_position = 0.0
                else: # Short
                    if high_p >= self.sl_price:
                        barrier_hit = "SL"
                        target_position = 0.0
                        ref_price = self.sl_price
                    elif low_p <= self.tp_price:
                        barrier_hit = "TP"
                        target_position = 0.0
                        ref_price = self.tp_price
                    elif self.trade_steps >= self.max_holding_steps:
                        barrier_hit = "Time"
                        target_position = 0.0
                        
                if barrier_hit is not None:
                    trade_size = target_position - self.position
            else:
                # پوزیشن صاف است: تنظیم اهداف خروج در صورت پوزیشن‌گیری جدید
                if abs(target_position) > 1e-8:
                    self.entry_price = ref_price
                    self.trade_steps = 0
                    tp_ratio = self.tp_bps / 10000.0
                    sl_ratio = self.sl_bps / 10000.0
                    if target_position > 0: # Long
                        self.tp_price = ref_price * (1.0 + tp_ratio)
                        self.sl_price = ref_price * (1.0 - sl_ratio)
                    else: # Short
                        self.tp_price = ref_price * (1.0 - tp_ratio)
                        self.sl_price = ref_price * (1.0 + sl_ratio)

            # Volatility-adaptive slippage on top of reference execution price
            slippage_rate = self.base_slippage_rate + 0.005 * (abs(trade_size) / (market_depth + 1e-8)) * (1.0 + volatility)
            execution_price = ref_price * (1.0 + np.sign(trade_size) * slippage_rate)
            
            trade_value = abs(trade_size) * execution_price
            transaction_fee = trade_value * self.transaction_fee_rate
            
            self.cash -= (trade_size * execution_price) + transaction_fee
            prev_position = self.position
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
            
            # 3. محاسبه پاداش مبتنی بر روش مرز سه‌گانه (Path-Dependent Triple-Barrier)
            if barrier_hit == "TP":
                reward = 1.5 + (pnl_pct * 100.0) - fee_pct
            elif barrier_hit == "SL":
                reward = -2.0 + (pnl_pct * 100.0) - fee_pct
            elif barrier_hit == "Time":
                reward = (pnl_pct * 100.0) - 0.2 - fee_pct
            elif abs(prev_position) > 1e-8 and abs(self.position) < 1e-8:
                # خروج دستی اختیاری توسط خود مدل عصبی
                reward = (pnl_pct * 100.0) - fee_pct
            else:
                # عدم تغییر موقعیت یا نگهداری فعال بدون برخورد به مرزها: پاداش صفر
                reward = 0.0
            
            # اعمال جریمه تغییر پوزیشن جهت جلوگیری از نوسان کاذب معاملاتی
            if abs(trade_size) > 1e-8 and barrier_hit is None:
                action_penalty_pct = (trade_value / self.portfolio_value) * 0.5
                reward -= action_penalty_pct * 100.0
                
            self.portfolio_value = new_portfolio_value
            
            obs = self._get_observation() if not terminated else np.zeros(12, dtype=np.float64)
            self.obs_history.append(obs)
            stacked_obs = np.concatenate(list(self.obs_history))
            
            info = {
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "position": self.position,
                "trade_size": trade_size,
                "transaction_fee": transaction_fee + terminal_liquidation_fee,
                "slippage": slippage_rate * ref_price
            }
        return stacked_obs, reward, terminated, truncated, info

    def _build_volume_bars(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        تبدیل کندل‌های زمانی به کندل‌های دلاری تطبیقی با نوسان (Volatility-Adjusted Dollar Bars)
        و محاسبه شاخص OBI با وزن حجمی (Volume-Weighted OBI) در طول کل زمان شکل‌گیری کندل حجمی.
        """
        import numpy as np
        import pandas as pd

        # بررسی وجود ستون‌های مورد نیاز
        if "volume" not in raw_df.columns or "mid_price" not in raw_df.columns:
            return raw_df

        # محاسبه نوسان متحرک بازده قیمت برای ترشولدهای پویا
        pct_change = raw_df["mid_price"].pct_change()
        rolling_std = pct_change.rolling(window=288, min_periods=30).std()
        
        volume_bars = []
        current_volume = 0.0
        current_ticks = []
        
        for idx in range(len(raw_df)):
            row = raw_df.iloc[idx]
            # حجم دلاری اتمیک این کندل زمانی
            vol_val = row["volume"] * row["mid_price"]
            current_ticks.append(row)
            current_volume += vol_val
            
            # محاسبه انحراف معیار جاری
            vol = rolling_std.iloc[idx]
            if pd.isna(vol) or vol <= 0:
                vol = 0.015 # فالبک ۱.۵ درصد نوسان
                
            # ترشولد دلار بار تطبیقی بر اساس نوسان (فرمول دِ پرادو)
            # نوسان بالا -> ترشولد کمتر (شکار دقیق‌تر کندل‌های انفجاری)
            # نوسان پایین -> ترشولد بیشتر (کاهش معاملاتی رِنج و بیهوده)
            base_dollar_target = 50000.0
            gamma = 100.0
            v_thresh = base_dollar_target / (1.0 + gamma * vol)
                
            if current_volume >= v_thresh:
                # بستن کندل حجمی و استخراج مقادیر جدید
                opens = current_ticks[0]["open"] if "open" in current_ticks[0] else current_ticks[0]["mid_price"]
                closes = current_ticks[-1]["close"] if "close" in current_ticks[-1] else current_ticks[-1]["mid_price"]
                highs = max([t.get("high", t["mid_price"]) for t in current_ticks])
                lows = min([t.get("low", t["mid_price"]) for t in current_ticks])
                
                # محاسبه OBI وزن‌دهی شده با حجم در کل طول زمان شکل‌گیری کندل حجمی (Comment 2)
                total_vol = sum([t["volume"] for t in current_ticks])
                if total_vol > 0:
                    # موازنه اتمیک Bid/Ask عمق دفترچه سفارش با وزن‌دهی حجم معاملات
                    vw_obi = sum([
                        t["volume"] * (
                            (t.get("bid_depth", 1.0) - t.get("ask_depth", 1.0)) / 
                            (t.get("bid_depth", 1.0) + t.get("ask_depth", 1.0) + 1e-8)
                        ) for t in current_ticks
                    ]) / total_vol
                else:
                    vw_obi = 0.0
                
                bar = {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "mid_price": closes,
                    "volume": total_vol,
                    "bid_depth": sum([t.get("bid_depth", 1.0) for t in current_ticks]),
                    "ask_depth": sum([t.get("ask_depth", 1.0) for t in current_ticks]),
                    "depth_imbalance": vw_obi, # موازنه انباشته OBI
                    "spread": np.mean([t.get("spread", 0.0) for t in current_ticks]),
                    "basis": np.mean([t.get("basis", 0.0) for t in current_ticks]),
                    "carry": np.mean([t.get("carry", 0.0001) for t in current_ticks]),
                    "volatility": np.std([t["mid_price"] for t in current_ticks]) / np.mean([t["mid_price"] for t in current_ticks]) if len(current_ticks) > 1 else 0.02,
                    "convenience_yield": np.mean([t.get("convenience_yield", 0.0) for t in current_ticks]),
                    "speculator_ratio": np.mean([t.get("speculator_ratio", 0.5) for t in current_ticks]),
                    "sentiment": np.mean([t.get("sentiment", 0.0) for t in current_ticks]),
                    "surprise": np.mean([t.get("surprise", 0.0) for t in current_ticks]),
                }
                
                # فیلدهای دلخواه برای فالبک و تطابق با استراتژی YoYo
                for fld in ["spread_ratio", "basis_ratio", "volatility_ratio", "carry_ratio", "roll_yield"]:
                    if fld in current_ticks[0]:
                        bar[fld] = np.mean([t[fld] for t in current_ticks])
                for fld in ["f_low", "f_high", "futures_price", "spread", "basis", "carry"]:
                    if fld in current_ticks[0]:
                        if fld == "f_low":
                            bar[fld] = min([t[fld] for t in current_ticks])
                        elif fld == "f_high":
                            bar[fld] = max([t[fld] for t in current_ticks])
                        else:
                            bar[fld] = current_ticks[-1][fld]
                
                volume_bars.append(bar)
                current_volume = 0.0
                current_ticks = []
                
        if len(volume_bars) == 0:
            return raw_df
            
        return pd.DataFrame(volume_bars)
