import unittest
from unittest.mock import MagicMock, patch
import os
import json
import ccxt

from src.core.screener import fetch_top_altcoins_sync, generate_screener_report

class TestAltcoinScreener(unittest.TestCase):
    def setUp(self):
        # ساخت تیکرهای ماک برای تست فیلترها و مرتب‌سازی
        self.mock_tickers = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "quoteVolume": 10000000.0,
                "percentage": 2.5,
                "close": 68000.0,
                "info": {"contractType": "PerpetualOption"}
            },
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "quoteVolume": 5000000.0,
                "percentage": 1.2,
                "close": 3800.0,
                "info": {"contractType": "perpetual"}
            },
            "USDT/USDT": {
                "symbol": "USDT/USDT",
                "quoteVolume": 20000000.0,
                "percentage": 0.0,
                "close": 1.0,
                "info": {"contractType": "perpetual"}
            },
            "SOL/USDT:USDT": {
                "symbol": "SOL/USDT:USDT",
                "quoteVolume": 8000000.0,
                "percentage": -3.5,
                "close": 150.0,
                "info": {"contractType": "perpetual"}
            },
            "XRP/USDT": {
                "symbol": "XRP/USDT",
                "quoteVolume": 100000.0,
                "percentage": 0.5,
                "close": 0.5,
                "info": {"contractType": "perpetual"}
            },
            "ADA/BTC": {
                "symbol": "ADA/BTC",
                "quoteVolume": 50000.0,
                "percentage": 0.1,
                "close": 0.000008,
                "info": {"contractType": "perpetual"}
            },
            "DOT/USDT": {
                "symbol": "DOT/USDT",
                "quoteVolume": 150000.0,
                "percentage": -1.0,
                "close": 6.0,
                "info": {"contractType": "Delivery"}
            }
        }

    @patch('ccxt.binance')
    def test_fetch_top_altcoins_filtering_and_sorting(self, mock_binance_class):
        # تنظیم نمونه ماک شده صرافی بایننس
        mock_exchange = MagicMock()
        mock_exchange.fetch_tickers.return_value = self.mock_tickers
        mock_binance_class.return_value = mock_exchange

        # اجرای اسکنر روی صرافی ماک شده
        top_coins = fetch_top_altcoins_sync("binance", limit=3)

        # بررسی نتایج:
        # 1. تعداد نمادهای بازگشتی نباید از حد مجاز (limit = 3) بیشتر باشد.
        self.assertEqual(len(top_coins), 3)

        # 2. فیلتر استیبل‌کوین: USDT/USDT نباید در نتایج باشد.
        bases = [c["base"] for c in top_coins]
        self.assertNotIn("USDT", bases)

        # 3. فیلتر کوت تتر: ADA/BTC نباید در نتایج باشد.
        symbols = [c["symbol"] for c in top_coins]
        self.assertNotIn("ADA/BTC", symbols)

        # 4. فیلتر قرارداد دائمی: DOT/USDT نباید در نتایج باشد (چون contractType برابر Delivery است).
        self.assertNotIn("DOT/USDT", symbols)

        # 5. مرتب‌سازی حجم نزولی:
        # مقادیر مورد انتظار فیلتر شده و مرتب شده:
        # ۱. BTC/USDT (حجم ۱۰ میلیون)
        # ۲. SOL/USDT:USDT (حجم ۸ میلیون)
        # ۳. ETH/USDT (حجم ۵ میلیون)
        self.assertEqual(top_coins[0]["symbol"], "BTC/USDT")
        self.assertEqual(top_coins[1]["symbol"], "SOL/USDT:USDT")
        self.assertEqual(top_coins[2]["symbol"], "ETH/USDT")

    @patch('ccxt.binance')
    def test_generate_screener_report(self, mock_binance_class):
        mock_exchange = MagicMock()
        mock_exchange.fetch_tickers.return_value = self.mock_tickers
        mock_binance_class.return_value = mock_exchange

        # حذف گزارش قبلی در صورت وجود
        report_file = os.path.join("analysis", "screener_report.json")
        if os.path.exists(report_file):
            os.remove(report_file)

        # ایجاد گزارش جدید
        generate_screener_report("binance", limit=5)

        # بررسی ذخیره موفق گزارش روی دیسک
        self.assertTrue(os.path.exists(report_file))

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["exchange_id"], "binance")
        self.assertTrue(len(data["altcoins"]) > 0)
        self.assertEqual(data["altcoins"][0]["symbol"], "BTC/USDT")

if __name__ == '__main__':
    unittest.main()
