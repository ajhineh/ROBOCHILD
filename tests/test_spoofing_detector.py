import sys
import os
import unittest
import time

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.spoofing_detector import SpoofingDetector


class TestSpoofingDetector(unittest.TestCase):
    def setUp(self):
        # ساخت یک دتکتور آزمایشی با مشخصات سریع
        self.detector = SpoofingDetector(
            depth_levels=3,
            trade_window_seconds=2,       # پنجره زمانی ۲ ثانیه‌ای برای راحتی تست
            spoof_threshold_pct=0.15       # آستانه ۱۵ درصد لغو فریبکارانه
        )
        self.symbol = "POPCAT/USDT:USDT"

    def test_obi_calculation(self):
        """بررسی محاسبات صحیح و خالص شاخص عدم تعادل دفترچه سفارشات (OBI)"""
        # دفترچه با عدم تعادل شدید خرید (Bid-Heavy)
        # قیمت‌ها و حجم‌ها
        bids = [[1.00, 100.0], [0.99, 100.0], [0.98, 100.0]]  # مجموع حجم خرید: ۳۰۰
        asks = [[1.01, 50.0], [1.02, 50.0], [1.03, 50.0]]     # مجموع حجم فروش: ۱۵۰
        
        now = int(time.time() * 1000)
        result = self.detector.process_order_book(self.symbol, bids, asks, now)

        self.assertIsNotNone(result)
        # OBI = (300 - 150) / (300 + 150) = 150 / 450 = 0.3333...
        self.assertAlmostEqual(result["raw_obi"], 0.3333, places=3)
        self.assertEqual(result["mid_price"], 1.005)

    def test_trades_tracking_and_trimming(self):
        """بررسی درستی پایش معاملات بلادرنگ و فیلتر کردن معاملات منقضی شده خارج از پنجره زمانی"""
        now = int(time.time() * 1000)

        # اضافه کردن چند معامله به ترتیب زمانی (قدیمی‌ترین به جدیدترین)
        self.detector.add_trade(self.symbol, 1.0, 20.0, "sell", now - 2500) # ۲.۵ ثانیه پیش (خارج از پنجره ۲ ثانیه‌ای)
        self.detector.add_trade(self.symbol, 1.0, 10.0, "buy", now - 1500)  # ۱.۵ ثانیه پیش (داخل پنجره ۲ ثانیه‌ای)
        self.detector.add_trade(self.symbol, 1.0, 15.0, "buy", now)        # لحظه فعلی (داخل پنجره)

        # ارسال دفترچه سفارشات برای تحریک فرآیند فیلترینگ و محاسبه
        bids = [[1.00, 10.0], [0.99, 10.0], [0.98, 10.0]]
        asks = [[1.01, 10.0], [1.02, 10.0], [1.03, 10.0]]
        
        result = self.detector.process_order_book(self.symbol, bids, asks, now)

        # بررسی اینکه معامله ۲.۵ ثانیه پیش پاک شده است و فقط مجموع خریدهای ۱۰ و ۱۵ دلاری مانده‌اند
        self.assertEqual(result["market_buy_vol"], 25.0)
        self.assertEqual(result["market_sell_vol"], 0.0)

    def test_buy_spoofing_detection(self):
        """بررسی کشف دیوارهای خرید فریبکارانه (Buy Spoofing) در صورت ریزش شدید حجم بید بدون معامله معادل"""
        now = int(time.time() * 1000)

        # فریم اول: حجم بالای بیدها (مجموع ۳۰۰)
        bids_1 = [[1.00, 100.0], [0.99, 100.0], [0.98, 100.0]]
        asks_1 = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]]
        self.detector.process_order_book(self.symbol, bids_1, asks_1, now)

        # فریم دوم: ۵۰ درصد از حجم بید ناگهانی لغو می‌شود (بید جدید مجموعاً ۱۵۰ واحد)
        # در این میان هیچ معامله فروش مارکتی انجام نشده است
        bids_2 = [[1.00, 50.0], [0.99, 50.0], [0.98, 50.0]]
        asks_2 = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]]
        
        result = self.detector.process_order_book(self.symbol, bids_2, asks_2, now + 100)

        # باید اسپوفینگ بید کشف شود (کاهش ۱۵۰ واحد بدون معامله معادل، که از آستانه ۱۵٪ یعنی ۴۵ واحد بیشتر است)
        self.assertTrue(result["spoof_detected"])
        self.assertEqual(result["spoof_type"], "buy_spoof")
        self.assertEqual(result["confirmed_side"], "flat")  # اگرچه OBI خام منفی است اما باید خنثی بماند

    def test_sell_spoofing_detection(self):
        """بررسی کشف دیوارهای فروش فریبکارانه (Sell Spoofing) در صورت ریزش شدید حجم اسک بدون معامله معادل"""
        now = int(time.time() * 1000)

        # فریم اول: حجم بالای اسک‌ها (مجموع ۳۰۰)
        bids_1 = [[1.00, 100.0], [0.99, 100.0], [0.98, 100.0]]
        asks_1 = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]]
        self.detector.process_order_book(self.symbol, bids_1, asks_1, now)

        # فریم دوم: ۵۰ درصد از حجم اسک ناگهانی لغو می‌شود (اسک جدید مجموعاً ۱۵۰ واحد)
        # هیچ معامله خرید مارکتی انجام نشده است
        bids_2 = [[1.00, 100.0], [0.99, 100.0], [0.98, 100.0]]
        asks_2 = [[1.01, 50.0], [1.02, 50.0], [1.03, 50.0]]
        
        result = self.detector.process_order_book(self.symbol, bids_2, asks_2, now + 100)

        self.assertTrue(result["spoof_detected"])
        self.assertEqual(result["spoof_type"], "sell_spoof")
        self.assertEqual(result["confirmed_side"], "flat")

    def test_genuine_volume_drops_do_not_trigger_spoofing(self):
        """بررسی اینکه لغو دیواری که ناشی از پر شدن توسط معاملات مارکت واقعی است، اسپوفینگ در نظر گرفته نشود."""
        now = int(time.time() * 1000)

        # فریم اول: مجموع بید ۳۰۰
        bids_1 = [[1.00, 100.0], [0.99, 100.0], [0.98, 100.0]]
        asks_1 = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]]
        self.detector.process_order_book(self.symbol, bids_1, asks_1, now)

        # اضافه کردن یک معامله فروش مارکت واقعی سنگین به اندازه ۱۴۰ واحد
        self.detector.add_trade(self.symbol, 1.00, 140.0, "sell", now + 50)

        # فریم دوم: حجم بیدها کاهش یافته و ۱۵۰ شده است (کاهش ۱۵۰ واحدی بید)
        # چون ۱۴۰ واحد آن ناشی از معامله مارکت بوده، حجم لغو خالص فقط ۱۰ واحد است (کمتر از آستانه ۱۵٪ یعنی ۴۵ واحد)
        bids_2 = [[1.00, 50.0], [0.99, 50.0], [0.98, 50.0]]
        asks_2 = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]]
        
        result = self.detector.process_order_book(self.symbol, bids_2, asks_2, now + 100)

        # نباید خطای فریب و اسپوف صادر شود چون خریدار واقعی پوزیشن خود را پر کرده است
        self.assertFalse(result["spoof_detected"])
        self.assertEqual(result["spoof_type"], "none")

    def test_confirmed_direction_signals(self):
        """بررسی درستی تایید نهایی سیگنال‌های جهت معاملات بر پایه OBI فیلترشده و جریان سفارشات"""
        now = int(time.time() * 1000)

        # سناریو ۱: عدم تعادل صعودی شدید بدون اسپوفینگ اما خریدار مارکتی نداریم -> FLAT
        bids = [[1.00, 300.0], [0.99, 300.0], [0.98, 300.0]] # بید انباشته ۹۰۰
        asks = [[1.01, 100.0], [1.02, 100.0], [1.03, 100.0]] # اسک انباشته ۳۰۰ -> OBI = 600/1200 = 0.50
        
        result = self.detector.process_order_book(self.symbol, bids, asks, now)
        self.assertEqual(result["confirmed_side"], "flat")  # خریدار مارکت نیست پس خنثی است

        # سناریو ۲: عدم تعادل صعودی شدید + وجود خریدار مارکت پرقدرت در ۱۰ ثانیه اخیر -> LONG تایید شده
        self.detector.add_trade(self.symbol, 1.00, 50.0, "buy", now)
        
        result = self.detector.process_order_book(self.symbol, bids, asks, now + 50)
        self.assertEqual(result["confirmed_side"], "long")


if __name__ == "__main__":
    unittest.main()
