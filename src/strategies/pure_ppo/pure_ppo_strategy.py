import os
import json
import time
import logging
import asyncio
import random
from collections import deque
from typing import Dict, List, Optional, Callable, Literal, TypedDict
import numpy as np

from src.config import Config
from src.core.rl_shared.state_parser import RLStateParser
from src.core.rl_shared.model_loader import RLModelLoader

logger = logging.getLogger("ROBORDER.PurePPOStrategy")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ActivePPOTrade(TypedDict, total=False):
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    amount: float
    leverage: int
    sl: float
    tp: float
    tp1: float
    tp2: float
    tp1_hit: bool
    timestamp: int
    status: Literal["pending", "filled"]
    diagnostic_report: Optional[dict]


class PurePPOStrategy:
    """
    استراتژی معاملاتی خالص مبتنی بر یادگیری تقویت‌پذیر پایان‌به‌پایان (End-to-End PPO-LSTM).
    این کلاس با تفکیک کاملا ناهمگام بخش پردازش داده از ترد اصلی صرافی، سرعت فوق‌العاده بالا
    و تصمیم‌گیری بلادرنگ عصبی بر مبنای بردار وضعیت ۱۲ بعدی را ارائه می‌دهد.
    """
    def __init__(
        self,
        symbols: List[str],
        quote_denomination: Literal["USDT", "SOL"] = "USDT",
        history_file_path: str = "microtick_history.json"
    ):
        self.symbols = symbols
        self.quote_denomination = quote_denomination
        self.history_file_path = os.path.abspath(history_file_path)

        # صف دریافت ناهمگام تیک‌های قیمتی تفکیک شده برای هر نماد جهت پیشگیری از لغزش و تداخل نمادها
        self.tick_queues: Dict[str, asyncio.Queue] = {sym: asyncio.Queue() for sym in symbols}
        self.worker_task: Optional[asyncio.Task] = None
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        self.should_run = True

        # ردیاب زمان آخرین پیش‌بینی عصبی برای هر نماد جهت اعمال محدودیت نرخ (Rate Limiting)
        self.last_prediction_time: Dict[str, float] = {sym: 0.0 for sym in symbols}

        # بافرهای تاریخچه تیک‌های قیمتی برای هر نماد (نگهداری آخرین ۶۰ ثانیه تیک‌ها)
        self.prices: Dict[str, deque] = {sym: deque() for sym in symbols}
        
        # تاریخچه وضعیت‌ها برای Frame Stacking
        self.obs_history: Dict[str, deque] = {sym: deque(maxlen=10) for sym in symbols}
        
        # مدیریت مدل‌های یادگیری تقویت‌پذیر و آمار نرمال‌سازی
        self.loader = RLModelLoader()
        self.models: Dict[str, Any] = {}
        self.normalized_envs: Dict[str, Any] = {}
        self.lstm_states: Dict[str, Optional[tuple]] = {sym: None for sym in symbols}

        # مدیریت موقعیت‌ها و سفارشات فعال ربات
        self.active_trades: Dict[str, ActivePPOTrade] = {}
        self.last_exit_times: Dict[str, int] = {}
        self.last_order_placed_time: Dict[str, float] = {sym: 0.0 for sym in symbols}

        # متغیرهای مربوط به اتصال LOB و تراکنش‌های DEX که از هسته هیبریدی تغذیه می‌شوند
        self.latest_lob: Optional[dict] = None
        self.recent_dex_trades: List[dict] = []

        # بافرهای تجمیع حجم متحرک برای تشخیص بسته‌شدن کندل‌های حجمی
        self.volume_accumulators: Dict[str, float] = {sym: 0.0 for sym in symbols}
        self.volume_ticks_history: Dict[str, List[dict]] = {sym: [] for sym in symbols}
        self.volume_bar_completed: Dict[str, bool] = {sym: False for sym in symbols}
        self.volume_bar_price: Dict[str, float] = {sym: 0.0 for sym in symbols}
        
        # بافر قیمت‌های آخرین کندل‌های حجمی تکمیل شده جهت محاسبه نوسان پویا
        self.completed_bar_prices: Dict[str, deque] = {sym: deque(maxlen=288) for sym in symbols}

        # تاریخچه سیگنال‌ها و آمارهای مالی داشبورد ربات
        self.history = {
            "signals": [],
            "stats": {
                "totalTrades": 0,
                "wins": 0,
                "losses": 0,
                "totalPnL": 0.0
            }
        }

        # کالبک‌های خروجی جهت ارسال پیام ورود/خروج به هسته صرافی/مدیریت ریسک
        self.on_entry_callback: Optional[Callable[[dict], None]] = None
        self.on_exit_callback: Optional[Callable[[str, dict, float, float, str], None]] = None

        self.load_history()

    async def start(self) -> None:
        """راه‌اندازی ترد ناهمگام استراتژی و بارگذاری مدل‌های Ensemble"""
        self.should_run = True
        
        # بارگذاری پویای مدل‌های Ensemble برای تمام جفت‌ارزها
        for sym in self.symbols:
            models_dict = self.loader.load_ensemble_models(sym)
            ppo_model, _ = models_dict.get("ppo", (None, None))
            if ppo_model is not None:
                self.models[sym] = models_dict
                logger.info(f"🧠 Ensemble Models loaded successfully for {sym}")
            else:
                self.models[sym] = None
                logger.warning(f"⚠️ Could not load Ensemble PPO model weights for {sym}. Artificial mock actions will be used.")

        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("🚀 PurePPOStrategy background worker loop started.")

    def set_callbacks(
        self,
        on_entry: Callable[[dict], None],
        on_exit: Callable[[str, dict, float, float, str], None]
    ):
        """تنظیم کالبک‌ها جهت تعامل با هسته صرافی"""
        self.on_entry_callback = on_entry
        self.on_exit_callback = on_exit

    def load_history(self) -> None:
        try:
            if os.path.exists(self.history_file_path):
                with open(self.history_file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load PPO history: {e}")

    def save_history(self) -> None:
        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save PPO history: {e}")

    def log_signal(self, signal: dict) -> None:
        self.history["signals"].append(signal)
        self.save_history()

    async def initialize_candles(self, exchange) -> None:
        """متد تطبیق رابط کاربری جهت سازگاری کامل با هسته اصلی ربات"""
        logger.info("📡 [PurePPOStrategy] Dynamic model agent initialization completed.")
        
    def feed_tick(self, symbol: str, price: float, timestamp: int, lob_result: Optional[dict] = None, dex_trades: Optional[list] = None) -> None:
        """تغذیه ناهمگام تیک‌های قیمتی زمان‌واقعی به صف دریافت بدون بلاک کردن ترد فراخوان"""
        try:
            if symbol not in self.tick_queues:
                self.tick_queues[symbol] = asyncio.Queue()
            self.tick_queues[symbol].put_nowait({
                "symbol": symbol,
                "price": price,
                "timestamp": timestamp,
                "lob_result": lob_result,
                "dex_trades": dex_trades
            })
            
            # راه‌اندازی ترد پس‌زمینه به صورت پویا در صورت اضافه شدن نماد جدید در اجرای لایو
            if self.worker_task and not self.worker_task.done():
                if symbol not in self.worker_tasks or self.worker_tasks[symbol].done():
                    self.worker_tasks[symbol] = asyncio.create_task(self._worker_loop_for_symbol(symbol))
        except Exception as e:
            logger.error(f"Error putting tick into PPO queue for {symbol}: {e}")

    def feed_trade(self, symbol: str, price: float, amount: float, side: str, timestamp: int) -> None:
        """تغذیه معاملات بازار به تجمیع‌کننده حجم لایو جهت ایجاد کندل‌های حجمی"""
        try:
            if symbol not in self.volume_accumulators:
                self.volume_accumulators[symbol] = 0.0
                self.volume_ticks_history[symbol] = []
                self.volume_bar_completed[symbol] = False
                self.volume_bar_price[symbol] = 0.0
                
            trade_val = amount * price
            self.volume_accumulators[symbol] += trade_val
            self.volume_ticks_history[symbol].append({
                "price": price,
                "amount": amount
            })
            
            # محاسبه پویای نوسان جاری از تاریخچه کندل‌های کامل شده قبلی (دِ پرادو)
            if symbol not in self.completed_bar_prices:
                self.completed_bar_prices[symbol] = deque(maxlen=288)
                
            completed_prices = list(self.completed_bar_prices[symbol])
            if len(completed_prices) >= 10:
                returns_list = [
                    (completed_prices[i] - completed_prices[i-1]) / completed_prices[i-1]
                    for i in range(1, len(completed_prices))
                ]
                vol = float(np.std(returns_list))
            else:
                vol = 0.015 # فالبک ۱.۵ درصد نوسان
                
            # ترشولد دلار بار تطبیقی بر اساس نوسان
            base_dollar_target = 50000.0
            gamma = 100.0
            v_thresh = base_dollar_target / (1.0 + gamma * vol)
                
            # اگر حجم انباشته شده از حد آستانه عبور کند، کندل حجمی کامل می‌شود
            if self.volume_accumulators[symbol] >= v_thresh:
                total_amount = sum([t["amount"] for t in self.volume_ticks_history[symbol]])
                if total_amount > 0:
                    avg_price = sum([t["price"] * t["amount"] for t in self.volume_ticks_history[symbol]]) / total_amount
                else:
                    avg_price = price
                    
                self.volume_bar_completed[symbol] = True
                self.volume_bar_price[symbol] = avg_price
                
                # ثبت قیمت جهت محاسبات نوسان بعدی
                self.completed_bar_prices[symbol].append(avg_price)
                
                logger.info(
                    f"📦 Volatility-Adjusted Dollar Bar Completed for {symbol} | "
                    f"Vol Target: ${v_thresh:,.2f} | Volatility: {vol*100:.2f}% | "
                    f"Weighted Price: ${avg_price:.6f} | AI Evaluation Activated."
                )
                
                # بازنشانی بافر
                self.volume_accumulators[symbol] = 0.0
                self.volume_ticks_history[symbol] = []
        except Exception as e:
            logger.error(f"Error aggregating live volume bar for {symbol}: {e}")

    async def _worker_loop(self) -> None:
        """حلقه همگام‌ساز ناهمگام اصلی جهت هماهنگی حلقه‌های اختصاصی نمادها"""
        self.worker_tasks = {}
        for sym in self.symbols:
            self.worker_tasks[sym] = asyncio.create_task(self._worker_loop_for_symbol(sym))
        
        try:
            await asyncio.gather(*self.worker_tasks.values())
        except asyncio.CancelledError:
            for sym, task in self.worker_tasks.items():
                if not task.done():
                    task.cancel()
            raise

    async def _worker_loop_for_symbol(self, symbol: str) -> None:
        """حلقه اختصاصی نماد جهت پردازش داده‌ها و پیش‌بینی مدل عصبی بدون تداخل با سایر ارزها"""
        if symbol not in self.tick_queues:
            self.tick_queues[symbol] = asyncio.Queue()
        queue = self.tick_queues[symbol]
        
        while self.should_run:
            try:
                tick = await queue.get()
                await self._handle_tick_async(tick)
                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in PPO async worker for {symbol}: {e}")
                await asyncio.sleep(0.1)

    async def _handle_tick_async(self, tick: dict) -> None:
        symbol = tick["symbol"]
        price = tick["price"]
        now = tick["timestamp"]
        lob_result = tick.get("lob_result")
        dex_trades = tick.get("dex_trades", [])

        # ۱. راه‌اندازی و مقداردهی تنبل (Lazy Initialization) نمادهای پویا
        if symbol not in self.prices:
            self.prices[symbol] = deque()
        if symbol not in self.lstm_states:
            self.lstm_states[symbol] = None
        if symbol not in self.last_order_placed_time:
            self.last_order_placed_time[symbol] = 0.0
        if symbol not in self.last_prediction_time:
            self.last_prediction_time[symbol] = 0.0

        # ۲. بارگذاری تنبل مدل عصبی در صورت لزوم
        if symbol not in self.models:
            models_dict = self.loader.load_ensemble_models(symbol)
            ppo_model, _ = models_dict.get("ppo", (None, None))
            if ppo_model is not None:
                self.models[symbol] = models_dict
                logger.info(f"🧠 Ensemble Models loaded dynamically via tick for {symbol}")
            else:
                self.models[symbol] = None
                logger.warning(f"⚠️ Could not load Ensemble models for {symbol}. Artificial mock actions will be used.")

        # ۳. به‌روزرسانی بافر محلی قیمت‌ها
        history = self.prices[symbol]
        history.append({"price": price, "timestamp": now})
        
        # پاکسازی تیک‌های قیمتی قدیمی‌تر از ۶۰ ثانیه
        cutoff = now - 60000
        while history and history[0]["timestamp"] < cutoff:
            history.popleft()

        # ۲. بررسی فیلترهای فاندامنتال (اخبار کلان و فاندینگ ریت) و لغو سفارشات معلق PPO
        active_trade = self.active_trades.get(symbol)
        if hasattr(self, "engine") and self.engine:
            drift_now_ms = now + Config.CLOCK_DRIFT_MS
            macro_ok = self.engine._check_macro_news_window(drift_now_ms)
            funding_ok = self.engine._check_funding_rate_window(drift_now_ms)
            
            if not macro_ok or not funding_ok:
                if active_trade:
                    if active_trade["status"] == "pending":
                        logger.warning(
                            f"🚨 Pending limit order for {symbol} cancelled due to fundamental block! "
                            f"Macro News: {'OK' if macro_ok else 'BLOCKED'} | Funding Rate: {'OK' if funding_ok else 'BLOCKED'}"
                        )
                        self._cancel_trade(symbol, now)
                        return
                else:
                    # هیچ موقعیت بازی نداریم و به دلیل شرایط بلاک خبری/فاندینگ ریت مجاز به ثبت سفارش جدید نیستیم
                    return

        # ۳. بررسی پوزیشن باز فعال
        if active_trade:
            await self._monitor_position_async(symbol, price, now, active_trade)
            return

        # ۴. دروازه‌بانی کندل حجمی: پیش‌بینی عصبی فقط در صورت تکمیل یک کندل حجمی انجام می‌شود
        if not self.volume_bar_completed.get(symbol, False):
            return
            
        # استفاده از قیمت میانگین وزنی کندل حجمی به جای قیمت تیک لحظه‌ای
        price = self.volume_bar_price.get(symbol, price)
        self.volume_bar_completed[symbol] = False

        # ۳. بررسی دوره استراحت (Cooldown)
        last_exit = self.last_exit_times.get(symbol, 0)
        if (now - last_exit) < Config.COOLDOWN_MS:
            return

        # جلوگیری از ارسال سفارشات مکرر در فواصل خیلی کوتاه
        now_sec = time.time()
        if (now_sec - self.last_order_placed_time[symbol]) < 5.0:
            return

        # ۸. جلوگیری از اجرای مکرر پیش‌بینی مدل عصبی جهت پیشگیری از لغزش و تداخل ناشی از اورلود CPU (Rate Limiting)
        last_pred_time = self.last_prediction_time.get(symbol, 0.0)
        if (now_sec - last_pred_time) < 0.2:  # حداکثر ۵ پیش‌بینی در ثانیه برای هر ارز
            return

        # ۴. نمونه‌گیری وضعیت و استخراج ویژگی‌های ۱۲ بعدی
        # تخمین نوسانات متحرک ساده در بافر قیمت‌ها
        volatility_ratio = 0.02
        if len(history) >= 5:
            prices_list = [p["price"] for p in history]
            volatility_ratio = float(np.std(prices_list) / np.mean(prices_list))

        # استفاده از پارسر ویژگی برای ساخت بردار ورودی
        obs = RLStateParser.parse_market_state(
            symbol=symbol,
            lob_result=lob_result,
            dex_trades=dex_trades,
            volatility_ratio=volatility_ratio,
            account_position=0.0,  # پوزیشن فعلی صفر (آماده ورود)
            max_inventory=10.0,
            mid_price=price,
            funding_rate=0.0001,
            basis_ratio=0.0
        )

        # مقداردهی به بافر فریم‌ها (Frame Stacking)
        if symbol not in self.obs_history:
            self.obs_history[symbol] = deque(maxlen=10)
            
        if len(self.obs_history[symbol]) == 0:
            for _ in range(10):
                self.obs_history[symbol].append(obs)
        else:
            self.obs_history[symbol].append(obs)
            
        stacked_obs = np.concatenate(list(self.obs_history[symbol]))

        # ثبت زمان پیش‌بینی فعلی پیش از اجرای مدل
        self.last_prediction_time[symbol] = now_sec

        models_dict = self.models.get(symbol)
        
        ensemble_action = 0.0
        fallback_to_ppo = False
        ppo_action = 0.0
        
        diag_report = None

        if isinstance(models_dict, dict):
            ppo_model, ppo_env = models_dict.get("ppo", (None, None))
            sac_model, sac_env = models_dict.get("sac", (None, None))
            td3_model, td3_env = models_dict.get("td3", (None, None))
            
            # بررسی لود بودن مدل‌ها جهت تعیین فالبک
            if ppo_model is None:
                fallback_to_ppo = True
                ppo_action = 0.0
            elif sac_model is None or td3_model is None:
                logger.warning(f"⚠️ SAC or TD3 model is missing for {symbol}. Falling back to PPO-only prediction.")
                fallback_to_ppo = True
            
            # ۱. پیش‌بینی PPO
            if ppo_model is not None:
                try:
                    # بررسی ابعاد ورودی مورد انتظار مدل جهت پایداری کامل
                    model_obs_shape = ppo_model.observation_space.shape[0] if hasattr(ppo_model, "observation_space") else 120
                    if model_obs_shape == 12:
                        obs_ppo = RLModelLoader.normalize_observation(obs, ppo_env)
                    else:
                        obs_ppo = RLModelLoader.normalize_observation(stacked_obs, ppo_env)
                    
                    last_state = self.lstm_states[symbol]
                    if hasattr(ppo_model, "policy") and "Lstm" in type(ppo_model.policy).__name__:
                        episode_start = np.array([last_state is None])
                        action, next_state = ppo_model.predict(
                            obs_ppo,
                            state=last_state,
                            episode_start=episode_start,
                            deterministic=True
                        )
                        self.lstm_states[symbol] = next_state
                        ppo_action = float(action[0])
                    else:
                        action, _ = ppo_model.predict(obs_ppo, deterministic=True)
                        ppo_action = float(action[0])
                except Exception as ppo_err:
                    logger.error(f"❌ Error predicting PPO for {symbol}: {ppo_err}")
                    ppo_action = 0.0
                    fallback_to_ppo = True
            
            # ۲ و ۳. پیش‌بینی SAC و TD3 در صورت عدم فالبک
            if not fallback_to_ppo:
                try:
                    obs_sac = RLModelLoader.normalize_observation(stacked_obs, sac_env)
                    sac_act_raw, _ = sac_model.predict(obs_sac, deterministic=True)
                    sac_action = float(sac_act_raw[0])
                    
                    obs_td3 = RLModelLoader.normalize_observation(stacked_obs, td3_env)
                    td3_act_raw, _ = td3_model.predict(obs_td3, deterministic=True)
                    td3_action = float(td3_act_raw[0])
                    
                    # نرمال‌سازی با np.tanh()
                    ppo_norm = float(np.tanh(ppo_action / 0.40))
                    sac_norm = float(np.tanh(sac_action / 0.30))
                    td3_norm = float(np.tanh(td3_action / 0.30))
                    
                    # محاسبه وزن‌های انسیبل تطبیقی
                    weights = self.get_adaptive_weights(symbol, price, now_sec)
                    w_ppo = weights.get("ppo", 0.50)
                    w_sac = weights.get("sac", 0.30)
                    w_td3 = weights.get("td3", 0.20)
                    
                    ensemble_action = w_ppo * ppo_norm + w_sac * sac_norm + w_td3 * td3_norm
                    
                    # محاسبه بزرگترین سهم در تصمیم‌گیری برای تعیین مدل لیدر
                    contrib_ppo = abs(w_ppo * ppo_norm)
                    contrib_sac = abs(w_sac * sac_norm)
                    contrib_td3 = abs(w_td3 * td3_norm)
                    contribs = {"PPO": contrib_ppo, "SAC": contrib_sac, "TD3": contrib_td3}
                    deciding_model = max(contribs, key=contribs.get)
                    
                    # ایجاد گزارش عیب‌یابی برای لاگ و وزن‌دهی انطباقی بعدی
                    diag_report = {
                        "ppo_raw": ppo_action,
                        "ppo_norm": ppo_norm,
                        "sac_raw": sac_action,
                        "sac_norm": sac_norm,
                        "td3_raw": td3_action,
                        "td3_norm": td3_norm,
                        "weights": weights,
                        "ensemble_action": ensemble_action,
                        "fallback": False,
                        "deciding_model": deciding_model
                    }
                    
                    logger.info(
                        f"📊 Ensemble components for {symbol}: "
                        f"PPO: {ppo_action:+.4f} (norm: {ppo_norm:+.4f}, w: {w_ppo:.2f}) | "
                        f"SAC: {sac_action:+.4f} (norm: {sac_norm:+.4f}, w: {w_sac:.2f}) | "
                        f"TD3: {td3_action:+.4f} (norm: {td3_norm:+.4f}, w: {w_td3:.2f}) | "
                        f"Result: {ensemble_action:+.4f}"
                    )
                except Exception as ensemble_err:
                    logger.error(f"❌ Error executing Ensemble models for {symbol}: {ensemble_err}. Falling back to PPO-only.")
                    fallback_to_ppo = True
        else:
            # سازگاری عقب‌رو با مدل‌های تک PPO قدیمی
            fallback_to_ppo = True
            model = models_dict
            env = self.normalized_envs.get(symbol)
            if model is not None:
                try:
                    # برای مدل قدیمی، بردار ورودی همان obs ۱۲ بعدی است
                    obs_ppo = RLModelLoader.normalize_observation(obs, env)
                    last_state = self.lstm_states[symbol]
                    if hasattr(model, "policy") and "Lstm" in type(model.policy).__name__:
                        episode_start = np.array([last_state is None])
                        action, next_state = model.predict(
                            obs_ppo,
                            state=last_state,
                            episode_start=episode_start,
                            deterministic=True
                        )
                        self.lstm_states[symbol] = next_state
                        ppo_action = float(action[0])
                    else:
                        action, _ = model.predict(obs_ppo, deterministic=True)
                        ppo_action = float(action[0])
                except Exception as err:
                    logger.error(f"❌ Error predicting old single model: {err}")
                    ppo_action = 0.0

        # بارگذاری پویای آستانه تصمیم‌گیری (Threshold) از فایل تنظیمات در صورت وجود
        symbol_clean = symbol.split('/')[0].lower()
        config_path = os.path.join("models", f"config_{symbol_clean}.json")
        cfg_threshold = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    cfg_threshold = cfg.get("ensemble_threshold")
            except Exception as e:
                logger.error(f"⚠️ Error reading ensemble_threshold from config: {e}")

        if fallback_to_ppo:
            if isinstance(models_dict, dict):
                # فالبک برای مدل انسیبل جدید: اکشن نرمالایز شده PPO به عنوان مبنا
                ppo_norm = float(np.tanh(ppo_action / 0.40))
                ensemble_action = ppo_norm
                threshold = cfg_threshold if cfg_threshold is not None else 0.65
                diag_report = {
                    "ppo_raw": ppo_action,
                    "ppo_norm": ppo_norm,
                    "ensemble_action": ensemble_action,
                    "fallback": True,
                    "deciding_model": "PPO"
                }
                logger.info(f"🔄 Fallback triggered: Using PPO-only normalized action {ensemble_action:+.4f} with threshold {threshold}")
            else:
                # برای مدل‌های تک قدیمی
                ensemble_action = ppo_action
                threshold = cfg_threshold if cfg_threshold is not None else (0.60 if symbol in ["POPCAT/USDT:USDT", "BOME/USDT:USDT"] else 0.25)
                diag_report = {
                    "ppo_raw": ppo_action,
                    "ensemble_action": ensemble_action,
                    "old_model_mode": True,
                    "deciding_model": "PPO"
                }
                logger.info(f"🔄 Old Model Mode: Using PPO raw action {ensemble_action:+.4f} with threshold {threshold}")
        else:
            threshold = cfg_threshold if cfg_threshold is not None else 0.65

        # ۷. ارزیابی حد آستانه ورود سیگنال (Trigger Threshold)
        adjusted_action = ensemble_action
        
        if abs(adjusted_action) >= threshold:
            side: Literal["long", "short"] = "long" if adjusted_action > 0 else "short"
            await self._trigger_ppo_trade(symbol, price, side, adjusted_action, now, diag_report)

    def _detect_market_regime(self, symbol: str, current_price: float, now_sec: float) -> str:
        """
        تشخیص رژیم بازار بر اساس نوسانات استاندارد و جهت روند در یک بازه ۳۰ دقیقه‌ای.
        خروجی‌ها:
        - "stable_trend": نوسان کم، جهت روند قوی (مناسب PPO)
        - "choppy_range": نوسان بالا، بدون روند مشخص (مناسب SAC)
        - "extreme_breakout": نوسان بالا و روند قوی (مناسب TD3)
        - "default": حالت عادی
        """
        if not hasattr(self, "regime_prices"):
            self.regime_prices = {}
        if symbol not in self.regime_prices:
            self.regime_prices[symbol] = deque()
            
        history = self.regime_prices[symbol]
        history.append({"price": current_price, "timestamp": now_sec})
        
        # نگهداری ۳۰ دقیقه آخر داده‌ها برای ارزیابی رژیم بازار
        cutoff = now_sec - 1800.0
        while history and history[0]["timestamp"] < cutoff:
            history.popleft()
            
        # اطمینان از اینکه داده‌های موجود حداقل ۵ دقیقه (۳۰۰ ثانیه) از زمان را پوشش می‌دهند
        if len(history) < 2 or (history[-1]["timestamp"] - history[0]["timestamp"]) < 300.0:
            return "default"
            
        prices_list = [p["price"] for p in history]
        mean_p = np.mean(prices_list)
        std_p = np.std(prices_list) / mean_p if mean_p > 0 else 0
        
        # درصد تغییر قیمت کل بازه
        price_change = abs(prices_list[-1] - prices_list[0]) / prices_list[0] if prices_list[0] > 0 else 0
        
        # مقادیر آستانه برای نوسان و قدرت روند
        vol_threshold = 0.0015   # 0.15% انحراف معیار
        trend_threshold = 0.003  # 0.3% تغییر قیمت
        
        is_high_vol = std_p > vol_threshold
        is_strong_trend = price_change > trend_threshold
        
        if is_strong_trend and not is_high_vol:
            return "stable_trend"
        elif is_high_vol and not is_strong_trend:
            return "choppy_range"
        elif is_high_vol and is_strong_trend:
            return "extreme_breakout"
        else:
            return "default"

    def get_adaptive_weights(self, symbol: str, current_price: Optional[float] = None, now_sec: Optional[float] = None) -> Dict[str, float]:
        """
        محاسبه داینامیک وزن‌های Ensemble بر اساس عملکرد تاریخی مدل‌ها در ۱۴ تا ۳۰ روز گذشته
        و اعمال ضرایب تشویقی/تنبیهی بر اساس رژیم فعلی بازار و تخصص ذاتی هر یک از مدل‌های سه‌گانه.
        """
        # تلاش برای بارگذاری وزن‌های پیش‌فرض اولیه از فایل پیکربندی نماد
        symbol_clean = symbol.split('/')[0].lower()
        config_path = os.path.join("models", f"config_{symbol_clean}.json")
        default_weights = {"ppo": 0.50, "sac": 0.30, "td3": 0.20}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    default_weights["ppo"] = cfg.get("ppo_weight", default_weights["ppo"])
                    default_weights["sac"] = cfg.get("sac_weight", default_weights["sac"])
                    default_weights["td3"] = cfg.get("td3_weight", default_weights["td3"])
            except Exception as e:
                logger.error(f"⚠️ Error reading default weights from config: {e}")
        
        # اعمال تنظیمات رژیم بازار در صورت ارسال قیمت و زمان لایو
        if current_price is not None and now_sec is not None:
            regime = self._detect_market_regime(symbol, current_price, now_sec)
            if regime == "stable_trend":
                # تقویت مدل محافظه‌کار PPO در جهت ترندهای باثبات
                default_weights["ppo"] *= 1.4
                default_weights["sac"] *= 0.8
                default_weights["td3"] *= 0.8
            elif regime == "choppy_range":
                # تقویت مدل نوسان‌گیر SAC در بازارهای بدون ترند و نوسانی
                default_weights["ppo"] *= 0.6
                default_weights["sac"] *= 1.5
                default_weights["td3"] *= 0.8
            elif regime == "extreme_breakout":
                # تقویت مدل TD3 در جهت جهش‌ها و ریزش‌های ناگهانی بازار
                default_weights["ppo"] *= 0.7
                default_weights["sac"] *= 0.7
                default_weights["td3"] *= 1.6
                
            # نرمال‌سازی مجدد وزن‌های پایه
            sum_base = sum(default_weights.values())
            for k in default_weights:
                default_weights[k] /= sum_base

        signals = self.history.get("signals", [])
        if not signals:
            return default_weights
            
        now_time = now_sec if now_sec is not None else time.time()
        days_14_ago = now_time - (14 * 24 * 3600)
        
        recent_trades = []
        for s in signals:
            if s.get("symbol") != symbol:
                continue
            if "pnl" not in s or s.get("pnl") is None:
                continue
            trade_time = s.get("time", 0)
            if trade_time < days_14_ago:
                continue
            recent_trades.append(s)
            
        if len(recent_trades) < 5:
            return default_weights
            
        scores = {"ppo": 0.0, "sac": 0.0, "td3": 0.0}
        counts = {"ppo": 0, "sac": 0, "td3": 0}
        
        for trade in recent_trades:
            pnl = trade["pnl"]
            diag = trade.get("diagnostic_report")
            if not diag or not isinstance(diag, dict):
                continue
                
            is_win = pnl > 0
            
            trade_type = trade.get("type", "")
            if "SELL_EXIT" in trade_type:
                entry_direction = 1.0  # LONG
            elif "BUY_EXIT" in trade_type:
                entry_direction = -1.0  # SHORT
            else:
                continue
                
            for algo in ["ppo", "sac", "td3"]:
                norm_key = f"{algo}_norm"
                if norm_key in diag:
                    val = diag[norm_key]
                    is_aligned = (val * entry_direction) > 0
                    
                    if (is_win and is_aligned) or (not is_win and not is_aligned):
                        scores[algo] += 1.0
                    counts[algo] += 1
                    
        final_weights = {}
        total_score = 0.0
        
        for algo in ["ppo", "sac", "td3"]:
            if counts[algo] > 0:
                accuracy = scores[algo] / counts[algo]
                base_w = default_weights[algo]
                final_weights[algo] = base_w * (0.5 + accuracy)
            else:
                final_weights[algo] = default_weights[algo]
            total_score += final_weights[algo]
            
        if total_score > 0:
            for algo in final_weights:
                final_weights[algo] /= total_score
        else:
            final_weights = default_weights
            
        final_weights["ppo"] = max(min(final_weights["ppo"], 0.60), 0.35)
        final_weights["sac"] = max(min(final_weights["sac"], 0.45), 0.20)
        final_weights["td3"] = max(min(final_weights["td3"], 0.35), 0.15)
        
        sum_w = sum(final_weights.values())
        for algo in final_weights:
            final_weights[algo] /= sum_w
            
        return final_weights

    async def _trigger_ppo_trade(
        self,
        symbol: str,
        price: float,
        side: Literal["long", "short"],
        action_ratio: float,
        timestamp: int,
        diagnostic_report: Optional[dict] = None
    ) -> None:
        """ایجاد پوزیشن معاملاتی شبیه‌سازی شده و ارسال سیگنال پیشنهادی به هسته صرافی جهت کنترل فیلترهای ۲۹ گانه"""
        self.last_order_placed_time[symbol] = time.time()

        # ۱. محاسبه حجم داینامیک بر اساس درصد سرمایه مجاز
        max_capital = (Config.TRADE_CAPITAL_PCT / 100.0) * Config.CURRENT_BALANCE
        amount_usdt = abs(action_ratio) * max_capital
        
        # اطمینان از قرارگیری حجم در محدوده‌های ایمن
        amount_usdt = min(amount_usdt, max_capital)
        if amount_usdt <= 5.0:  # حداقل حجم معامله ۵ تتر
            return

        # ۲. بارگذاری پارامترهای BPS اختصاصی ارز در صورت وجود
        symbol_clean = symbol.split('/')[0].lower()
        config_path = os.path.join("models", f"config_{symbol_clean}.json")
        tp_bps = Config.TAKE_PROFIT_BPS
        sl_bps = Config.STOP_LOSS_BPS
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    tp_bps = cfg.get("take_profit_bps", tp_bps)
                    sl_bps = cfg.get("stop_loss_bps", sl_bps)
            except Exception as e:
                logger.error(f"⚠️ Error reading take_profit/stop_loss from config: {e}")

        try:
            sl_pct = sl_bps / 10000.0  # حد ضرر به درصد (مثلا 0.001 برای 10 BPS)
            # اهرم به گونه‌ای محاسبه می‌شود که ضربدر حد ضرر درصد ریسک مجاز را پوشش دهد
            calculated_leverage = int((Config.YOYO_RISK_PCT / 100.0) / sl_pct)
            if calculated_leverage <= 0:
                calculated_leverage = Config.DEFAULT_LEVERAGE
        except Exception:
            calculated_leverage = Config.DEFAULT_LEVERAGE
            
        leverage = min(max(calculated_leverage, 1), Config.MAX_LEVERAGE)

        # ۳. محاسبه تارگت‌ها بر مبنای BPS تنظیم شده
        tp_ratio = tp_bps / 10000.0
        sl_ratio = sl_bps / 10000.0
        tp1_ratio = (sl_bps * 1.5) / 10000.0 # TP1 is 1.5x risk

        if side == "long":
            tp1 = price * (1 + tp1_ratio)
            tp2 = price * (1 + tp_ratio)
            sl = price * (1 - sl_ratio)
        else:
            tp1 = price * (1 - tp1_ratio)
            tp2 = price * (1 - tp_ratio)
            sl = price * (1 + sl_ratio)

        # ۴. ایجاد پوزیشن در حالت pending
        new_trade: ActivePPOTrade = {
            "symbol": symbol,
            "side": side,
            "entry_price": price,
            "amount": amount_usdt,
            "leverage": leverage,
            "sl": sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "tp1_hit": False,
            "timestamp": timestamp,
            "status": "pending",
            "diagnostic_report": diagnostic_report
        }

        self.active_trades[symbol] = new_trade
        logger.info(f"🏹 Neural Network Ensemble proposed {side.upper()} order for {symbol} at ${price:.6f} | Weight: {action_ratio:+.2f}")

        # ۴. فید کردن سیگنال به موتور تصمیم‌گیر جهت اعمال فیلترهای ۲۹ گانه
        if self.on_entry_callback:
            try:
                # آماده‌سازی دیتا منطبق با ورودی handle_execute_entry در main.py
                self.on_entry_callback({
                    "symbol": symbol,
                    "side": side,
                    "entry_price_quote": price,
                    "entry_price_usdt": price,
                    "amount": amount_usdt,
                    "leverage": leverage,
                    "take_profit_quote": tp2,
                    "stop_loss_quote": sl,
                    "timestamp": timestamp,
                    "is_yoyo": True,  # جهت تطبیق کامل با معماری کالبک بدون تغییر main
                    "yoyo_data": new_trade
                })
            except Exception as e:
                logger.error(f"Error in PPO on_entry callback execution: {e}")

    async def _monitor_position_async(self, symbol: str, price: float, now: int, trade: ActivePPOTrade) -> None:
        """پایش مداوم وضعیت تارگت‌های معامله فعال"""
        if trade["status"] == "pending":
            # در بازار لایو وضعیت به filled تغییر می‌کند، در شبیه‌ساز با برخورد قیمت پر می‌شود
            if trade["side"] == "long" and price <= trade["entry_price"]:
                trade["status"] = "filled"
                trade["timestamp"] = now
                logger.info(f"💥 PPO LONG Limit Filled for {symbol} at ${trade['entry_price']:.6f}!")
            elif trade["side"] == "short" and price >= trade["entry_price"]:
                trade["status"] = "filled"
                trade["timestamp"] = now
                logger.info(f"💥 PPO SHORT Limit Filled for {symbol} at ${trade['entry_price']:.6f}!")
            return

        # ۱. بررسی لمس حد سود اول (TP1) و تبدیل به ریسک‌فری (Break-Even) تعدیل شده با کارمزد
        tp1 = trade.get("tp1", trade["tp"])
        tp2 = trade.get("tp2", trade["tp"])
        tp1_hit = trade.get("tp1_hit", False)

        if not tp1_hit:
            # بررسی لمس TP1 برای خروج حجم تنظیم‌شده
            if (trade["side"] == "long" and price >= tp1) or (trade["side"] == "short" and price <= tp1):
                tp1_exit_fraction = getattr(Config, "TP1_EXIT_PCT", 50.0) / 100.0
                exit_amount = trade["amount"] * tp1_exit_fraction
                remaining_amount = trade["amount"] * (1.0 - tp1_exit_fraction)
                
                pnl_pct = ((tp1 - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
                if trade["side"] == "short":
                    pnl_pct = -pnl_pct
                
                # محاسبه سود تتر
                pnl_usdt = exit_amount * (pnl_pct / 100.0)
                
                logger.info(f"🎯 TP1 Hit for {symbol} | Exiting {getattr(Config, 'TP1_EXIT_PCT', 50.0)}% volume | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
                
                # بروزرسانی حجم باقی‌مانده و پرچم TP1
                trade["amount"] = remaining_amount
                trade["tp1_hit"] = True
                
                # انتقال حد ضرر (SL) به نقطه ورود واقعی تعدیل شده با کارمزد رفت و برگشت (0.08% برای 2 * 0.04% صرافی)
                fee_rate = 0.0004
                round_trip_fee = 2.0 * fee_rate
                if trade["side"] == "long":
                    trade["sl"] = trade["entry_price"] * (1.0 + round_trip_fee)
                else:
                    trade["sl"] = trade["entry_price"] * (1.0 - round_trip_fee)
                
                logger.info(f"🛡️ SL moved to Fee-Adjusted Break-Even for remaining volume at ${trade['sl']:.6f}")
                
                # ثبت رویداد خروج پله‌ای در تاریخچه محلی
                self.log_signal({
                    "symbol": symbol,
                    "type": "SELL_EXIT" if trade["side"] == "long" else "BUY_EXIT",
                    "price": tp1,
                    "time": int(now / 1000),
                    "exitReason": "TP1",
                    "fullyExited": False,
                    "pnl": pnl_usdt,
                    "strategy": "PurePPOStrategy",
                    "diagnostic_report": trade.get("diagnostic_report")
                })
                
                # بروزرسانی آمارهای معاملاتی برای خروج پله‌ای اول
                stats = self.history["stats"]
                stats["totalTrades"] += 1
                if pnl_usdt > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["totalPnL"] = stats.get("totalPnL", 0.0) + pnl_usdt
                self.save_history()

                # فراخوانی کالبک خروج صرافی برای ۵۰٪ پوزیشن
                if self.on_exit_callback:
                    try:
                        self.on_exit_callback(symbol, trade, tp1, pnl_usdt, "TP1")
                    except Exception as e:
                        logger.error(f"Error in PPO TP1 callback: {e}")
                
                return

        # ۲. بررسی حد ضرر (SL یا BE)
        if (trade["side"] == "long" and price <= trade["sl"]) or (trade["side"] == "short" and price >= trade["sl"]):
            pnl_pct = ((trade["sl"] - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
            if trade["side"] == "short":
                pnl_pct = -pnl_pct
            
            pnl_usdt = trade["amount"] * (pnl_pct / 100.0)
            exit_reason = "BE" if tp1_hit else "SL"
            
            logger.info(f"🚪 EXIT ({exit_reason}) PPO {trade['side'].upper()} for {symbol} at ${price:.6f} | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
            self._close_ppo_position(symbol, trade, price, pnl_usdt, exit_reason, now)
            return

        # ۳. بررسی حد سود نهایی (TP2 یا همان TP اصلی)
        if (trade["side"] == "long" and price >= tp2) or (trade["side"] == "short" and price <= tp2):
            pnl_pct = ((tp2 - trade["entry_price"]) / trade["entry_price"]) * 100 * trade["leverage"]
            if trade["side"] == "short":
                pnl_pct = -pnl_pct
                
            pnl_usdt = trade["amount"] * (pnl_pct / 100.0)
            logger.info(f"🚪 EXIT (Take-Profit 2) PPO {trade['side'].upper()} for {symbol} at ${price:.6f} | PnL: {pnl_pct:.2f}% ({pnl_usdt:.4f} USDT)")
            self._close_ppo_position(symbol, trade, price, pnl_usdt, "TP2" if tp1_hit else "TP", now)
            return

    def _cancel_trade(self, symbol: str, now: int) -> None:
        """ابطال معامله در صورت رد شدن توسط فیلترهای کنترلی"""
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            del self.active_trades[symbol]
            self.last_exit_times[symbol] = now
            self.lstm_states[symbol] = None  # بازنشانی حافظه LSTM برای معامله بعدی
            
            if self.on_exit_callback:
                try:
                    self.on_exit_callback(symbol, trade, trade["entry_price"], 0.0, "CANCEL")
                except Exception as e:
                    logger.error(f"Error in PPO cancel callback: {e}")

    def _close_ppo_position(
        self,
        symbol: str,
        trade: ActivePPOTrade,
        exit_price: float,
        pnl_usdt: float,
        reason: str,
        now: int
    ) -> None:
        """بستن کامل پوزیشن و به‌روزرسانی داشبورد مالی"""
        del self.active_trades[symbol]
        self.last_exit_times[symbol] = now
        self.lstm_states[symbol] = None

        # ثبت خروج در تاریخچه محلی
        self.log_signal({
            "symbol": symbol,
            "type": "SELL_EXIT" if trade["side"] == "long" else "BUY_EXIT",
            "price": exit_price,
            "time": int(now / 1000),
            "exitReason": reason,
            "fullyExited": True,
            "pnl": pnl_usdt,
            "strategy": "PurePPOStrategy",
            "diagnostic_report": trade.get("diagnostic_report")
        })

        # بروزرسانی آمارهای معاملاتی
        stats = self.history["stats"]
        is_second_stage = trade.get("tp1_hit", False)
        
        if not is_second_stage:
            stats["totalTrades"] += 1
            if pnl_usdt > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
                
        stats["totalPnL"] = stats.get("totalPnL", 0.0) + pnl_usdt
        self.save_history()

        # فعال‌سازی کالبک خروج صرافی
        if self.on_exit_callback:
            try:
                self.on_exit_callback(symbol, trade, exit_price, pnl_usdt, reason)
            except Exception as e:
                logger.error(f"Error in PPO on_exit callback: {e}")

    def force_close_position(self, symbol: str, current_price: float, now: int) -> bool:
        """بستن فوری و دستی یک موقعیت معاملاتی توسط کاربر از داشبورد"""
        if symbol not in self.active_trades:
            return False
            
        trade = self.active_trades[symbol]
        entry = trade["entry_price"]
        side = trade["side"]
        leverage = trade["leverage"]
        amount = trade["amount"]
        
        if side == "long":
            pnl_pct = ((current_price - entry) / entry) * 100.0 * leverage
        else:
            pnl_pct = ((entry - current_price) / entry) * 100.0 * leverage
            
        pnl_usdt = amount * (pnl_pct / 100.0)
        logger.info(f"🚪 FORCE CLOSE PPO {side.upper()} for {symbol} at ${current_price:.6f} | PnL: {pnl_pct:+.2f}% ({pnl_usdt:+.4f} USDT)")
        
        self._close_ppo_position(
            symbol=symbol,
            trade=trade,
            exit_price=current_price,
            pnl_usdt=pnl_usdt,
            reason="FORCE_DASHBOARD_CLOSE",
            now=now
        )
        return True

    async def stop(self) -> None:
        """توقف ایمن ترد ناهمگام استراتژی"""
        self.should_run = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        # اطمینان از لغو تمام کارگران اختصاصی نمادها
        for sym, task in list(self.worker_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.worker_tasks.clear()
        logger.info("🔌 PurePPOStrategy background tasks stopped.")
