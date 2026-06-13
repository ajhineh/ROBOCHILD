import os
import logging
import numpy as np
import gymnasium as gym
from typing import Tuple, Optional, Any

logger = logging.getLogger("ROBORDER.RLShared.ModelLoader")

# تلاش برای بارگذاری RecurrentPPO
try:
    from sb3_contrib import RecurrentPPO
    USING_RECURRENT = True
    logger.info("🧠 sb3-contrib RecurrentPPO available for LSTM models.")
except ImportError:
    from stable_baselines3 import PPO
    USING_RECURRENT = False
    logger.info("🧠 sb3-contrib not found. Falling back to standard Feed-Forward PPO.")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class DummyEnvForVecNormalize(gym.Env):
    """
    محیط فرضی ساده جهت راه‌اندازی و بارگذاری آمارهای مقیاس‌گذاری VecNormalize.
    """
    def __init__(self, shape: int = 120):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )
        self.shape_val = shape

    def reset(self, seed=None, options=None):
        return np.zeros(self.shape_val, dtype=np.float64), {}

    def step(self, action):
        return np.zeros(self.shape_val, dtype=np.float64), 0.0, False, False, {}


class RLModelLoader:
    """
    کلاس مدیریت بارگذاری پویای مدل‌های هوش مصنوعی و آمار نرمال‌سازی آن‌ها.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = os.path.abspath(models_dir)

    def _sync_model_from_registry(self, symbol: str, algo: str) -> Tuple[Optional[str], Optional[str]]:
        """
        همگام‌سازی و دانلود فایل‌های مدل و VecNormalize از WandB Model Registry با مدیریت کش ۲۴ ساعته.
        """
        import time
        import shutil
        use_wandb = os.getenv("USE_WANDB", "false").lower() in ["true", "1"]
        if not use_wandb:
            return None, None
            
        symbol_clean = symbol.split('/')[0].upper()
        algo_upper = algo.upper()
        
        # مشخصات رجیستری و تگ هدف (با اولویت candidate و سپس production)
        entity = os.getenv("WANDB_ENTITY", "ROBOCHILD")
        project = os.getenv("WANDB_PROJECT", "ROBOCHILD-SOL")
        registry_name = f"ROBOCHILD-{symbol_clean}-{algo_upper}-Production"
        
        cache_dir = os.path.join(self.models_dir, "wandb_cache", algo.lower())
        os.makedirs(cache_dir, exist_ok=True)
        
        # نام فرضی فایل‌های دانلود شده
        model_filename = f"ppo_volume_bars_child_{symbol.split('/')[0].lower()}_{algo.lower()}_best.zip"
        stats_filename = f"ppo_volume_bars_child_{symbol.split('/')[0].lower()}_{algo.lower()}_vec_normalize.pkl"
        
        model_cache_path = os.path.join(cache_dir, model_filename)
        stats_cache_path = os.path.join(cache_dir, stats_filename)
        
        # بررسی انقضای کش ۲۴ ساعته (۸۶۴۰۰ ثانیه)
        cache_valid = False
        if os.path.exists(model_cache_path) and os.path.exists(stats_cache_path):
            file_age = time.time() - os.path.getmtime(model_cache_path)
            if file_age < 86400: # کمتر از ۲۴ ساعت
                cache_valid = True
                logger.info(f"⏳ Local cache for {algo_upper} model is still valid (age: {file_age/3600:.1f} hours). Using cached files.")
                return model_cache_path, stats_cache_path
        
        # در صورت انقضای کش یا عدم وجود، تلاش برای دانلود از WandB
        logger.info(f"🔄 Cache expired or missing. Attempting to download {algo_upper} model from WandB Registry...")
        try:
            import wandb
            api = wandb.Api()
            
            # تلاش برای پیدا کردن نسخه با تگ candidate و فالبک به production و سپس latest
            artifact = None
            for tag in ["candidate", "production", "latest"]:
                try:
                    artifact_path = f"{entity}/{project}/{registry_name}:{tag}"
                    logger.info(f"🔍 Checking Model Registry path: {artifact_path}")
                    artifact = api.artifact(artifact_path)
                    if artifact is not None:
                        logger.info(f"🎯 Found model version with tag '{tag}'")
                        break
                except Exception:
                    continue
            
            if artifact is None:
                logger.warning(f"⚠️ Model Registry {registry_name} not found in WandB. Falling back to local files.")
                return None, None
                
            # دانلود آرتیفکت به یک پوشه موقت و انتقال فایل‌ها به پوشه کش اصلی
            download_dir = artifact.download()
            
            # پیدا کردن فایل‌های دانلود شده در پوشه موقت و کپی به محل کش نهایی
            for f in os.listdir(download_dir):
                src_file = os.path.join(download_dir, f)
                if f.endswith(".zip"):
                    shutil.copy2(src_file, model_cache_path)
                elif f.endswith(".pkl"):
                    shutil.copy2(src_file, stats_cache_path)
            
            # بروزرسانی زمان تغییر فایل کش جهت مدیریت انقضای بعدی
            os.utime(model_cache_path, None)
            os.utime(stats_cache_path, None)
            
            logger.info(f"🟢 Successfully synced and cached {algo_upper} model from WandB Registry.")
            return model_cache_path, stats_cache_path
            
        except Exception as e:
            logger.error(f"❌ Error syncing model from WandB Registry: {e}. Falling back to standard local files.")
            return None, None

    def load_ppo_model(self, symbol: str) -> Tuple[Optional[Any], Optional[VecNormalize]]:
        """
        بارگذاری مدل هوش مصنوعی و فایل VecNormalize متناظر با توکن معاملاتی.
        """
        symbol_clean = symbol.split('/')[0].lower()
        
        # تلاش برای همگام‌سازی از رجیستری WandB
        model_path, stats_path = self._sync_model_from_registry(symbol, "ppo")
        
        if model_path is None or stats_path is None:
            # ۱. پیدا کردن نام فایل‌های مدل و آمار با اولویت بهترین مدل (Best)
            model_filename = f"ppo_volume_bars_child_{symbol_clean}_best.zip"
            stats_filename = f"ppo_volume_bars_child_{symbol_clean}_vec_normalize.pkl"
            
            model_path = os.path.join(self.models_dir, model_filename)
            stats_path = os.path.join(self.models_dir, stats_filename)
            
            # در صورت عدم وجود مدل بهترین اختصاصی، تلاش برای لود مدل نهایی اختصاصی
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Dedicated PPO best model for {symbol} not found. Trying fallback to final model...")
                model_filename = f"ppo_volume_bars_child_{symbol_clean}_final.zip"
                model_path = os.path.join(self.models_dir, model_filename)
                
            # مسیر پشتیبان کلی در صورت عدم وجود مدل اختصاصی نماد
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Dedicated PPO model for {symbol} not found at {model_path}. Loading default best model...")
                model_path = os.path.join(self.models_dir, "ppo_volume_bars_child_best.zip")
                stats_path = os.path.join(self.models_dir, "ppo_volume_bars_child_vec_normalize.pkl")
                
            # مسیر پشتیبان نهایی در صورت عدم وجود مدل پیش‌فرض بهترین
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Default best model not found. Fallback to default final model...")
                model_path = os.path.join(self.models_dir, "ppo_volume_bars_child_final.zip")
            
        if not os.path.exists(model_path):
            logger.error(f"❌ Critical: PPO model file not found at {model_path}")
            return None, None
            
        # ۲. بارگذاری مدل PPO/RecurrentPPO
        model = None
        try:
            if USING_RECURRENT:
                try:
                    # تلاش برای لود به عنوان RecurrentPPO
                    model = RecurrentPPO.load(model_path)
                    logger.info(f"🟢 Successfully loaded RecurrentPPO (PPO-LSTM) model from {os.path.basename(model_path)}")
                except Exception as recurrent_err:
                    # اگر فایل با RecurrentPPO ساخته نشده باشد، پس‌روی به PPO معمولی
                    logger.warning(f"⚠️ RecurrentPPO load failed: {recurrent_err}. Attempting fallback to standard PPO...")
                    model = PPO.load(model_path)
                    logger.info(f"🟢 Successfully loaded standard PPO model from {os.path.basename(model_path)}")
            else:
                model = PPO.load(model_path)
                logger.info(f"🟢 Successfully loaded standard PPO model from {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"❌ Failed to load PPO model weights from {model_path}: {e}")
            return None, None

        # ۳. بارگذاری فایل آمار نرمال‌سازی VecNormalize
        normalized_env = None
        if os.path.exists(stats_path):
            try:
                # ایجاد محیط فرضی جهت الصاق آمار
                dummy_venv = DummyVecEnv([lambda: DummyEnvForVecNormalize(shape=12)])
                normalized_env = VecNormalize.load(stats_path, dummy_venv)
                # اطمینان از اینکه آپدیت آمار در فاز اجرا بسته است
                normalized_env.training = False
                normalized_env.norm_reward = False
                logger.info(f"🟢 Successfully loaded VecNormalize statistics from {os.path.basename(stats_path)}")
            except Exception as e:
                logger.error(f"❌ Failed to load VecNormalize statistics from {stats_path}: {e}")
        else:
            logger.warning(f"⚠️ VecNormalize statistics file not found at {stats_path}. Observations will not be scaled.")
            
        return model, normalized_env

    def load_ensemble_models(self, symbol: str) -> dict:
        """
        بارگذاری همزمان مدل‌های PPO، SAC و TD3 به همراه آمارهای نرمال‌سازی مربوطه.
        """
        symbol_clean = symbol.split('/')[0].lower()
        models = {}
        
        for algo in ["ppo", "sac", "td3"]:
            # تلاش برای همگام‌سازی از رجیستری WandB برای هر یک از مدل‌های Ensemble
            model_path, stats_path = self._sync_model_from_registry(symbol, algo)
            
            if model_path is None or stats_path is None:
                model_filename = f"ppo_volume_bars_child_{symbol_clean}_{algo}_best.zip"
                stats_filename = f"ppo_volume_bars_child_{symbol_clean}_{algo}_vec_normalize.pkl"
                
                model_path = os.path.join(self.models_dir, model_filename)
                stats_path = os.path.join(self.models_dir, stats_filename)
                
                # بررسی فایل‌های فالبک نهایی
                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Dedicated Ensemble {algo.upper()} best model for {symbol} not found. Trying fallback to final model...")
                    model_filename = f"ppo_volume_bars_child_{symbol_clean}_{algo}_final.zip"
                    model_path = os.path.join(self.models_dir, model_filename)
                    
                # در صورتی که فایل انسیبل پیدا نشد و در فاز لود PPO بودیم، تلاش برای لود مدل تکی قدیمی
                if not os.path.exists(model_path) and algo == "ppo":
                    logger.warning(f"⚠️ Dedicated Ensemble PPO not found. Checking for old single PPO models...")
                    model_filename = f"ppo_volume_bars_child_{symbol_clean}_best.zip"
                    stats_filename = f"ppo_volume_bars_child_{symbol_clean}_vec_normalize.pkl"
                    model_path = os.path.join(self.models_dir, model_filename)
                    stats_path = os.path.join(self.models_dir, stats_filename)
                    if not os.path.exists(model_path):
                        model_filename = f"ppo_volume_bars_child_{symbol_clean}_final.zip"
                        model_path = os.path.join(self.models_dir, model_filename)
            
            if not os.path.exists(model_path):
                logger.error(f"❌ Ensemble model {algo.upper()} not found for {symbol} at {model_path}")
                models[algo] = (None, None)
                continue
                
            model = None
            try:
                if algo == "ppo":
                    if USING_RECURRENT:
                        try:
                            model = RecurrentPPO.load(model_path)
                            logger.info(f"🟢 Loaded RecurrentPPO (PPO-LSTM) for Ensemble from {os.path.basename(model_path)}")
                        except Exception:
                            model = PPO.load(model_path)
                            logger.info(f"🟢 Loaded standard PPO for Ensemble from {os.path.basename(model_path)}")
                    else:
                        model = PPO.load(model_path)
                        logger.info(f"🟢 Loaded standard PPO for Ensemble from {os.path.basename(model_path)}")
                elif algo == "sac":
                    from stable_baselines3 import SAC
                    model = SAC.load(model_path)
                    logger.info(f"🟢 Loaded SAC model for Ensemble from {os.path.basename(model_path)}")
                elif algo == "td3":
                    from stable_baselines3 import TD3
                    model = TD3.load(model_path)
                    logger.info(f"🟢 Loaded TD3 model for Ensemble from {os.path.basename(model_path)}")
            except Exception as e:
                logger.error(f"❌ Failed to load Ensemble model weights for {algo.upper()}: {e}")
                models[algo] = (None, None)
                continue
                
            # بارگذاری فایل VecNormalize متناظر
            normalized_env = None
            if os.path.exists(stats_path):
                try:
                    # بررسی کنید آیا فایل آماری قدیمی است (اندازه ۱۲) یا جدید (اندازه ۱۲۰)
                    # برای فایل‌های انسیبل اندازه ۱۲۰ و برای مدل تکی قدیمی اندازه ۱۲ است
                    import pickle
                    with open(stats_path, "rb") as f_stats:
                        stats_data = pickle.load(f_stats)
                    stats_shape = stats_data.obs_rms.mean.shape[0] if hasattr(stats_data, 'obs_rms') else 120
                    
                    dummy_venv = DummyVecEnv([lambda: DummyEnvForVecNormalize(shape=stats_shape)])
                    normalized_env = VecNormalize.load(stats_path, dummy_venv)
                    normalized_env.training = False
                    normalized_env.norm_reward = False
                    logger.info(f"🟢 Loaded VecNormalize statistics for {algo.upper()} with shape {stats_shape} from {os.path.basename(stats_path)}")
                except Exception as e:
                    logger.error(f"❌ Failed to load VecNormalize statistics for {algo.upper()}: {e}")
            else:
                logger.warning(f"⚠️ VecNormalize statistics file not found for {algo.upper()} at {stats_path}")
                
            models[algo] = (model, normalized_env)
            
        return models

    @staticmethod
    def normalize_observation(obs: np.ndarray, normalized_env: Optional[VecNormalize]) -> np.ndarray:
        """
        اعمال نرمال‌سازی و مقیاس‌دهی روی بردار وضعیت ورودی.
        """
        if normalized_env is None:
            return obs
            
        try:
            # مدل‌های Stable Baselines ورودی دو بعدی (دسته ای) می‌پذیرند
            obs_batched = np.expand_dims(obs, axis=0)
            obs_normalized = normalized_env.normalize_obs(obs_batched)
            return obs_normalized[0]
        except Exception as e:
            logger.error(f"Error normalizing observation: {e}")
            return obs
