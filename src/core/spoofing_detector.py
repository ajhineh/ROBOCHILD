import logging
from collections import deque
from typing import Dict, List, Optional, Literal, TypedDict

logger = logging.getLogger("ROBORDER.SpoofingDetector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LOBTrade(TypedDict):
    timestamp: int
    price: float
    amount: float
    side: Literal["buy", "sell"]


class LOBAnalysisResult(TypedDict):
    symbol: str
    mid_price: float
    raw_obi: float
    market_buy_vol: float
    market_sell_vol: float
    spoof_detected: bool
    spoof_type: Literal["none", "buy_spoof", "sell_spoof"]
    spoof_info: str
    confirmed_side: Literal["long", "short", "flat"]


class SpoofingDetector:
    """
    تحلیل‌گر بلادرنگ دفترچه سفارشات (LOB) و جریان معاملات (Order Flow) برای تشخیص دیوارکشی‌های فریبکارانه (Spoofing).
    این کلاس با تطبیق حجم سفارشات حذف شده از دفترچه با حجم معاملات انجام شده واقعی مارکت، فریب خریدار/فروشنده را تشخیص می‌دهد.
    """
    def __init__(
        self,
        depth_levels: int = 5,                # عمق محاسباتی دفترچه سفارشات
        trade_window_seconds: int = 10,       # پنجره زمانی جمع‌آوری حجم معاملات فعال
        spoof_threshold_pct: float = 0.15     # آستانه لغو غیرواقعی حجم دیوارهای بید/اسک (15 درصد)
    ):
        self.depth_levels = depth_levels
        self.trade_window_seconds = trade_window_seconds
        self.spoof_threshold_pct = spoof_threshold_pct

        # مخازن بافرهای مربوط به نمادهای مختلف
        self.recent_trades: Dict[str, deque] = {}
        
        # ذخیره آخرین حجم‌های دیده شده در پله‌های بالای دفترچه
        self.prev_bid_vols: Dict[str, float] = {}
        self.prev_ask_vols: Dict[str, float] = {}

    def _init_buffers(self, symbol: str) -> None:
        """آماده‌سازی بافرهای داده برای هر جفت‌ارز جدید"""
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = deque()
        if symbol not in self.prev_bid_vols:
            self.prev_bid_vols[symbol] = 0.0
        if symbol not in self.prev_ask_vols:
            self.prev_ask_vols[symbol] = 0.0

    def add_trade(self, symbol: str, price: float, amount: float, side: Literal["buy", "sell"], timestamp: int) -> None:
        """اضافه کردن یک معامله نهایی شده (Market Trade) به جریان معاملات اخیر جفت‌ارز"""
        self._init_buffers(symbol)
        
        trade: LOBTrade = {
            "timestamp": timestamp,
            "price": price,
            "amount": amount,
            "side": side
        }
        
        self.recent_trades[symbol].append(trade)
        self._trim_trades(symbol, timestamp)

    def _trim_trades(self, symbol: str, current_timestamp_ms: int) -> None:
        """پاک‌سازی معاملات قدیمی‌تر از محدوده پنجره زمانی فعال"""
        trades = self.recent_trades[symbol]
        cutoff = current_timestamp_ms - (self.trade_window_seconds * 1000)
        while trades and trades[0]["timestamp"] < cutoff:
            trades.popleft()

    def process_order_book(self, symbol: str, bids: List[List[float]], asks: List[List[float]], current_timestamp_ms: int) -> Optional[LOBAnalysisResult]:
        """
        پردازش عمیق تغییرات دفترچه سفارشات LOB و استخراج OBI و دیوارهای کاذب (Spoofing).
        
        فرمت ورودی bids و asks:
        bids = [[price_1, volume_1], [price_2, volume_2], ...]
        """
        self._init_buffers(symbol)
        self._trim_trades(symbol, current_timestamp_ms)

        if len(bids) < self.depth_levels or len(asks) < self.depth_levels:
            logger.warning(f"LOB depth for {symbol} is less than required depth level of {self.depth_levels}")
            return None

        # ۱. محاسبه حجم انباشته پله‌های بالای دفترچه (Passive Liquidity)
        current_bid_vol = sum([bid[1] for bid in bids[:self.depth_levels]])
        current_ask_vol = sum([ask[1] for ask in asks[:self.depth_levels]])
        mid_price = (bids[0][0] + asks[0][0]) / 2.0

        # ۲. محاسبه مجموع کل خریدها و فروش‌های مارکتی انجام شده اخیر (Active Executed Volume)
        trades = self.recent_trades[symbol]
        market_buy_vol = sum([t["amount"] for t in trades if t["side"] == "buy"])
        market_sell_vol = sum([t["amount"] for t in trades if t["side"] == "sell"])

        # ۳. محاسبه شاخص عدم تعادل دفترچه سفارشات (OBI - Order Book Imbalance)
        # خروجی OBI بین بازه -1.0 تا +1.0 است
        total_vol = current_bid_vol + current_ask_vol
        raw_obi = (current_bid_vol - current_ask_vol) / total_vol if total_vol > 0 else 0.0

        # ۴. تحلیل لغو سفارشات و کشف Spoofing
        # اگر حجم زیادی از دیوارهای سفارشات ناپدید شود بدون اینکه معامله مارکتی با آن حجم ثبت شده باشد، لغو سفارش (Spoofing) تشخیص داده می‌شود.
        spoof_detected = False
        spoof_type: Literal["none", "buy_spoof", "sell_spoof"] = "none"
        spoof_info = "پایدار (No Spoofing)"

        prev_bid_vol = self.prev_bid_vols[symbol]
        prev_ask_vol = self.prev_ask_vols[symbol]

        if prev_bid_vol > 0.0 and prev_ask_vol > 0.0:
            delta_bid_vol = prev_bid_vol - current_bid_vol
            delta_ask_vol = prev_ask_vol - current_ask_vol

            # بررسی لغو سفارشات خرید سنگین (Buy Spoofing)
            # کاهش در بید منهای معاملات فروش مارکت، بزرگتر از آستانه مجاز (15٪ از حجم قبلی)
            if delta_bid_vol > 0 and (delta_bid_vol - market_sell_vol) > (prev_bid_vol * self.spoof_threshold_pct):
                spoof_detected = True
                spoof_type = "buy_spoof"
                spoof_info = "⚠️ لغو سفارش خرید سنگین (Buy Spoofing Detected)"

            # بررسی لغو سفارشات فروش سنگین (Sell Spoofing)
            # کاهش در اسک منهای معاملات خرید مارکت، بزرگتر از آستانه مجاز (15٪ از حجم قبلی)
            elif delta_ask_vol > 0 and (delta_ask_vol - market_buy_vol) > (prev_ask_vol * self.spoof_threshold_pct):
                spoof_detected = True
                spoof_type = "sell_spoof"
                spoof_info = "⚠️ لغو سفارش فروش سنگین (Sell Spoofing Detected)"

        # ثبت حجم‌های فعلی به عنوان مقادیر قبلی برای تیک‌های بعدی
        self.prev_bid_vols[symbol] = current_bid_vol
        self.prev_ask_vols[symbol] = current_ask_vol

        # ۵. تایید جهت حرکت واقعی قیمت بر اساس ترکیب OBI فیلتر شده و جریان سفارشات
        confirmed_side: Literal["long", "short", "flat"] = "flat"

        if raw_obi > 0.3:
            if spoof_type == "buy_spoof":
                # دیوار خرید فریبکارانه بوده؛ سیگنال خرید فیلتر و مسدود می‌شود
                confirmed_side = "flat"
            else:
                # تایید لانگ به شرط وجود خریدار مارکت واقعی پرقدرت در بازار
                if market_buy_vol > market_sell_vol:
                    confirmed_side = "long"
                else:
                    confirmed_side = "flat"
        
        elif raw_obi < -0.3:
            if spoof_type == "sell_spoof":
                # دیوار فروش فریبکارانه بوده؛ سیگنال فروش فیلتر و مسدود می‌شود
                confirmed_side = "flat"
            else:
                # تایید شورت به شرط وجود فروشنده مارکت واقعی پرقدرت در بازار
                if market_sell_vol > market_buy_vol:
                    confirmed_side = "short"
                else:
                    confirmed_side = "flat"

        result: LOBAnalysisResult = {
            "symbol": symbol,
            "mid_price": mid_price,
            "raw_obi": raw_obi,
            "market_buy_vol": market_buy_vol,
            "market_sell_vol": market_sell_vol,
            "spoof_detected": spoof_detected,
            "spoof_type": spoof_type,
            "spoof_info": spoof_info,
            "confirmed_side": confirmed_side
        }

        return result
