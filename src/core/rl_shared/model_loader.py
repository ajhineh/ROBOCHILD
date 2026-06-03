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
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float64
        )

    def reset(self, seed=None, options=None):
        return np.zeros(12, dtype=np.float64), {}

    def step(self, action):
        return np.zeros(12, dtype=np.float64), 0.0, False, False, {}


class RLModelLoader:
    """
    کلاس مدیریت بارگذاری پویای مدل‌های هوش مصنوعی و آمار نرمال‌سازی آن‌ها.
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = os.path.abspath(models_dir)

    def load_ppo_model(self, symbol: str) -> Tuple[Optional[Any], Optional[VecNormalize]]:
        """
        بارگذاری مدل هوش مصنوعی و فایل VecNormalize متناظر با توکن معاملاتی.
        """
        symbol_clean = symbol.split('/')[0].lower()
        
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
                dummy_venv = DummyVecEnv([lambda: DummyEnvForVecNormalize()])
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
