import sys
import os
import unittest
import asyncio

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.order_executor import OrderExecutor


class TestOrderExecutor(unittest.TestCase):
    def setUp(self):
        # راه‌اندازی اگسکیوتر شبیه‌ساز با کنترل‌های سخت‌گیرانه ریسک
        self.executor = OrderExecutor(
            exchange_id="binance",
            live_trading=False,
            max_concurrent_positions=2,        # حداکثر ۲ پوزیشن همزمان باز
            max_drawdown_limit_usdt=50.0       # سقف حد ضرر روزانه ۵۰ دلار
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # ذخیره موقت تنظیمات فیلترهای بایپس شده جهت عدم تاثیرگذاری فایل .env توسعه‌دهنده در تست‌ها
        from src.config import Config
        self.old_bypassed = Config.BYPASSED_FILTERS
        self.old_bypassed_set = Config.BYPASSED_FILTERS_SET.copy()
        Config.BYPASSED_FILTERS = ""
        Config.BYPASSED_FILTERS_SET = set()

    def tearDown(self):
        self.loop.close()
        from src.config import Config
        Config.BYPASSED_FILTERS = self.old_bypassed
        Config.BYPASSED_FILTERS_SET = self.old_bypassed_set

    def test_simulation_mode_activation(self):
        """بررسی فعال‌سازی درست حالت شبیه‌ساز و عدم تلاش برای اتصال به حساب واقعی بدون کلید صرافی"""
        self.assertFalse(self.executor.live_trading)
        self.assertIsNone(self.executor.exchange)
        self.assertEqual(len(self.executor.open_positions), 0)

    def test_successful_simulated_entry(self):
        """بررسی صحت فرآیند باز کردن پوزیشن شبیه‌ساز لانگ و ذخیره درست پارامترهای ترید"""
        symbol = "POPCAT/USDT:USDT"
        
        success = self.loop.run_until_complete(
            self.executor.execute_entry(
                symbol=symbol,
                side="long",
                amount_usdt=20.0,
                leverage=10,
                take_profit_quote=1.002,
                stop_loss_quote=0.999
            )
        )

        self.assertTrue(success)
        self.assertIn(symbol, self.executor.open_positions)
        pos = self.executor.open_positions[symbol]
        self.assertEqual(pos["side"], "long")
        self.assertEqual(pos["leverage"], 10)
        self.assertEqual(pos["tp"], 1.002)
        self.assertEqual(pos["sl"], 0.999)

    def test_ptrc_max_concurrent_positions(self):
        """بررسی فیلتر ریسک PTRC: جلوگیری خودکار از معامله جدید در صورت پر بودن سقف پوزیشن‌های همزمان"""
        # باز کردن پوزیشن اول
        self.loop.run_until_complete(
            self.executor.execute_entry("POPCAT/USDT:USDT", "long", 10.0, 10, 1.02, 0.99)
        )
        # باز کردن پوزیشن دوم
        self.loop.run_until_complete(
            self.executor.execute_entry("WIF/USDT:USDT", "short", 10.0, 10, 0.98, 1.01)
        )

        # تلاش برای باز کردن پوزیشن سوم (که از سقف حد مجاز ۲ بیشتر است و باید ریجکت شود)
        success = self.loop.run_until_complete(
            self.executor.execute_entry("BOME/USDT:USDT", "long", 10.0, 10, 1.02, 0.99)
        )

        self.assertFalse(success)
        self.assertEqual(len(self.executor.open_positions), 2)
        self.assertNotIn("BOME/USDT:USDT", self.executor.open_positions)

    def test_ptrc_daily_drawdown_limit(self):
        """بررسی فیلتر ریسک PTRC: مسدود کردن کامل معاملات جدید در صورت رسیدن ضرر انباشته روزانه حساب به سقف مجاز"""
        # تنظیم دستی دروداون روزانه به ۶۰ دلار (که بزرگتر از سقف مجاز ۵۰ دلار است)
        self.executor.current_drawdown = 60.0

        success = self.loop.run_until_complete(
            self.executor.execute_entry("POPCAT/USDT:USDT", "long", 10.0, 10, 1.02, 0.99)
        )

        # به دلیل نقض دروداون، معامله نباید اجازه ساخت داشته باشد
        self.assertFalse(success)
        self.assertEqual(len(self.executor.open_positions), 0)

    def test_exit_and_drawdown_updates(self):
        """بررسی بسته شدن صحیح پوزیشن و محاسبه و اثرگذاری سود و زیان معامله در دروداون روزانه حساب"""
        symbol = "POPCAT/USDT:USDT"

        # باز کردن موقعیت تستی
        self.loop.run_until_complete(
            self.executor.execute_entry(symbol, "long", 10.0, 10, 1.02, 0.99)
        )

        self.assertEqual(self.executor.current_drawdown, 0.0)

        # بستن موقعیت با ضرر مالی ۱۰ دلار (حد ضرر فعال شده است)
        success = self.loop.run_until_complete(
            self.executor.execute_exit(symbol, pnl_usdt=-10.0, reason="SL")
        )

        self.assertTrue(success)
        self.assertNotIn(symbol, self.executor.open_positions)
        # دروداون روزانه حساب باید به میزان ضرر (۱۰ دلار) افزایش یافته باشد
        self.assertEqual(self.executor.current_drawdown, 10.0)

        # باز کردن موقعیت دوم
        self.loop.run_until_complete(
            self.executor.execute_entry(symbol, "long", 10.0, 10, 1.02, 0.99)
        )

        # بستن موقعیت دوم با سود ۱۵ دلار (کم شدن دروداون تا کف صفر دلار)
        self.loop.run_until_complete(
            self.executor.execute_exit(symbol, pnl_usdt=15.0, reason="TP")
        )
        self.assertEqual(self.executor.current_drawdown, 0.0)

    def test_partial_exit_dynamic_pct_and_stats(self):
        """بررسی صحت اجرای خروج پله‌ای با درصد پویا و عدم حذف پوزیشن تا خروج نهایی"""
        from src.config import Config
        old_tp1_exit_pct = Config.TP1_EXIT_PCT
        Config.TP1_EXIT_PCT = 40.0 # ۴۰ درصد خروج در TP1

        try:
            symbol = "POPCAT/USDT:USDT"
            initial_balance = Config.CURRENT_BALANCE

            # باز کردن موقعیت تستی با ارزش ۱۰۰ تتر
            self.loop.run_until_complete(
                self.executor.execute_entry(symbol, "long", 100.0, 10, 1.02, 0.99)
            )

            # خروج پله‌ای اول با دلیل TP1 و سود ۵ دلار
            success = self.loop.run_until_complete(
                self.executor.execute_exit(symbol, pnl_usdt=5.0, reason="TP1")
            )

            self.assertTrue(success)
            # پوزیشن نباید از لیست پوزیشن‌های فعال حذف شده باشد
            self.assertIn(symbol, self.executor.open_positions)
            
            pos = self.executor.open_positions[symbol]
            # حجم و مارجین باید به میزان ۶۰ درصد باقیمانده باشد (۱۰۰ * ۰.۶ = ۶۰ دلار)
            self.assertAlmostEqual(pos["amount"], 60.0)
            # بالانس باید ۵ دلار سود را منظور کرده باشد
            self.assertAlmostEqual(Config.CURRENT_BALANCE, initial_balance + 5.0)

            # خروج نهایی با دلیل TP2 و سود ۸ دلار
            success = self.loop.run_until_complete(
                self.executor.execute_exit(symbol, pnl_usdt=8.0, reason="TP2")
            )

            self.assertTrue(success)
            # پوزیشن اکنون باید کاملاً بسته شده باشد
            self.assertNotIn(symbol, self.executor.open_positions)
            # بالانس کل باید ۱۳ دلار سود را در کل منظور کرده باشد
            self.assertAlmostEqual(Config.CURRENT_BALANCE, initial_balance + 13.0)
        finally:
            Config.TP1_EXIT_PCT = old_tp1_exit_pct


if __name__ == "__main__":
    unittest.main()
