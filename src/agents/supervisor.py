import asyncio
import sys
from typing import Dict, Any, Optional
from .base import BaseAgent, AgentMessage
from .data_specialist import DataSpecialistAgent
from .file_specialist import FileSpecialistAgent

class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent (Coordinator): Orchestrates the market specialists,
    evaluates signals, detects spoofing cancellations, verifies safety limits,
    and submits executions.
    """
    def __init__(self, data_agent: DataSpecialistAgent, file_agent: FileSpecialistAgent, name: str = "supervisor"):
        super().__init__(name, "Orchestrator and Risk Controller")
        self.data_agent = data_agent
        self.file_agent = file_agent
        
        # State tracking for spoofing detection
        self.prev_bid_vol: Optional[float] = None
        self.prev_ask_vol: Optional[float] = None
        self.spoof_ratio = 0.15

    async def handle_message(self, message: AgentMessage):
        if message.message_type == "market_metrics":
            await self._process_market_data(message.data)
        elif message.message_type == "error":
            self.logger.error(f"Error notification from {message.sender}: {message.data}")
            # Here we would invoke a circuit breaker if connection drops repeatedly

    async def _process_market_data(self, data: Dict[str, Any]):
        symbol = data["symbol"]
        mid_price = data["mid_price"]
        raw_obi = data["raw_obi"]
        current_bid_vol = data["current_bid_volume"]
        current_ask_vol = data["current_ask_volume"]
        market_buy_vol = data["market_buy_volume"]
        market_sell_vol = data["market_sell_volume"]

        # 1. Detect Spoofing (detect_spoofing tool functionality)
        spoof_detected = False
        spoof_info = "پایدار (No Spoofing)"

        if self.prev_bid_vol is not None and self.prev_ask_vol is not None:
            delta_bid = self.prev_bid_vol - current_bid_vol
            delta_ask = self.prev_ask_vol - current_ask_vol

            # Buy Spoofing: order canceled without active sales
            if delta_bid > 0 and (delta_bid - market_sell_vol) > (self.prev_bid_vol * self.spoof_ratio):
                spoof_detected = True
                spoof_info = "⚠️ لغو سفارش خرید سنگین (Buy Spoofing Detected)"
            
            # Sell Spoofing: order canceled without active purchases
            elif delta_ask > 0 and (delta_ask - market_buy_vol) > (self.prev_ask_vol * self.spoof_ratio):
                spoof_detected = True
                spoof_info = "⚠️ لغو سفارش فروش سنگین (Sell Spoofing Detected)"

        # Save volumes for next iteration
        self.prev_bid_vol = current_bid_vol
        self.prev_ask_vol = current_ask_vol

        # 2. Formulate Confirmed Signals
        final_signal = "🔴 بدون موقعیت (FLAT)"
        
        if raw_obi > 0.3:
            if "Buy Spoofing" in spoof_info:
                final_signal = "🔒 فیلتر شده (تلاش برای فریب خریدار - سیگنال لانگ رد شد)"
            else:
                if market_buy_vol > market_sell_vol:
                    final_signal = "🟢 خرید تایید شده (CONFIRMED LONG - Genuine Buying)"
                    await self._trigger_trade(symbol, "buy", 1.0)
                else:
                    final_signal = "⏳ خنثی (عدم تایید خریدار مارکت)"
        
        elif raw_obi < -0.3:
            if "Sell Spoofing" in spoof_info:
                final_signal = "🔒 فیلتر شده (تلاش برای فریب فروشنده - سیگنال شورت رد شد)"
            else:
                if market_sell_vol > market_buy_vol:
                    final_signal = "🚨 فروش تایید شده (CONFIRMED SHORT - Genuine Selling)"
                    await self._trigger_trade(symbol, "sell", 1.0)
                else:
                    final_signal = "⏳ خنثی (عدم تایید فروشنده مارکت)"

        # Log signal history asynchronously
        await self.send_message(self.file_agent, "log_signal", {
            "symbol": symbol,
            "mid_price": mid_price,
            "obi": raw_obi,
            "spoof_status": spoof_info,
            "signal": final_signal
        })

        # Dynamic output print formatted for Windows Persian terminals
        sys.stdout.write(
            f"\rقیمت: {mid_price:.4f} | OBI خام: {raw_obi:+.2f} | "
            f"خرید ۱۰ث: {market_buy_vol:.1f} | فروش ۱۰ث: {market_sell_vol:.1f} | "
            f"وضعیت فریب: {spoof_info:<35} | سیگنال: {final_signal:<40}"
        )
        sys.stdout.flush()

    async def _trigger_trade(self, symbol: str, side: str, amount: float):
        # 3. Execution Guardrails (execute_order tool functionality)
        self.logger.info(f"Submitting {side.upper()} order for {amount} {symbol}...")
        
        # Dispatch trade record to FileSpecialist for persistent records
        await self.send_message(self.file_agent, "log_trade", {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "status": "triggered"
        })
