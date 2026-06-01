import numpy as np
import logging

logger = logging.getLogger("ROBORDER.RLShared.StateParser")

class RLStateParser:
    """
    مبدل و پارسر وضعیت بازار برای سیستم یادگیری تقویت‌پذیر.
    این کلاس داده‌های دریافتی زمان‌واقعی اردر بوک، معاملات بلاکچین، قیمت‌ها و اکانت را
    به یک بردار وضعیت استاندارد ۱۲ بعدی مناسب برای مدل PPO-LSTM تبدیل می‌کند.
    """
    @staticmethod
    def parse_market_state(
        symbol: str,
        lob_result: dict,
        dex_trades: list,
        volatility_ratio: float,
        account_position: float,
        max_inventory: float,
        mid_price: float,
        funding_rate: float = 0.0001,
        basis_ratio: float = 0.0
    ) -> np.ndarray:
        """
        تبدیل پارامترهای بازار به یک بردار وضعیت ۱۲ بعدی ترتیبی.
        """
        try:
            # ۱. نسبت پوزیشن به سقف مجاز
            pos_ratio = account_position / max_inventory if max_inventory > 0 else 0.0
            
            # ۲. پیشرفت دوره (در حالت استریم زنده مقدار ثابت 0.5 قرار می‌گیرد)
            progress = 0.5
            
            # ۳. نسبت اسپرد به قیمت میانی
            spread_ratio = 0.0
            depth_imbalance = 0.0
            
            if lob_result:
                # محاسبه اسپرد از بهترین Bid و Ask در صورت وجود داده
                best_bid = lob_result.get("best_bid", mid_price)
                best_ask = lob_result.get("best_ask", mid_price)
                if mid_price > 0:
                    spread_ratio = abs(best_ask - best_bid) / mid_price
                
                # شاخص عدم تعادل عمق اردر بوک
                depth_imbalance = lob_result.get("depth_imbalance", 0.0)
            
            # ۴. پارامترهای فاندینگ و مبنا
            carry_ratio = funding_rate
            roll_yield = basis_ratio
            
            # ۵. احساسات بازار بر اساس تعادل تراکنش‌های بلاکچین (DEX Trades Balance)
            sentiment = 0.0
            speculator_ratio = 0.0
            if dex_trades:
                dex_buy = sum([t["amount"] for t in dex_trades if t["side"] == "buy"])
                dex_sell = sum([t["amount"] for t in dex_trades if t["side"] == "sell"])
                total_dex = dex_buy + dex_sell
                if total_dex > 0:
                    sentiment = (dex_buy - dex_sell) / total_dex
                    # حجم تراکنش‌های نسبی به عنوان نسبت معاملات سفته‌بازی
                    speculator_ratio = min(1.0, total_dex / 100000.0)
            
            # بردار ویژگی نهایی منطبق بر observation_space محیط FuturesTradingEnv
            obs = np.array([
                pos_ratio,            # 0: Position ratio
                progress,             # 1: Progress
                spread_ratio,         # 2: Spread ratio
                depth_imbalance,      # 3: Depth imbalance (OBI)
                0.0,                  # 4: Convenience yield (default 0)
                basis_ratio,          # 5: Basis ratio
                carry_ratio,          # 6: Carry ratio (Funding rate)
                roll_yield,           # 7: Roll yield
                speculator_ratio,     # 8: Speculator ratio
                sentiment,            # 9: Market sentiment
                0.0,                  # 10: Surprise (default 0)
                volatility_ratio      # 11: Volatility ratio
            ], dtype=np.float32)
            
            # محدودسازی مقادیر جهت جلوگیری از ورود اعداد نامعتبر (NaN/Inf)
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
            return obs
            
        except Exception as e:
            logger.error(f"Error parsing market state for {symbol}: {e}")
            return np.zeros(12, dtype=np.float32)
