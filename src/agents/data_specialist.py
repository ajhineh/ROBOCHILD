import asyncio
from typing import Dict, Any, List
from collections import deque
from .base import BaseAgent, AgentMessage

class DataSpecialistAgent(BaseAgent):
    """
    Data Specialist Agent: Ingests WebSocket streams, aggregates trading volumes, 
    and calculates Order Book Imbalances (OBI).
    """
    def __init__(self, name: str = "data_specialist"):
        super().__init__(name, "Data Ingestion & OBI Calculator Specialist")
        self.recent_trades = deque()
        self.trade_window_ms = 10000  # 10 seconds sliding window
        self.depth_levels = 5
        self.supervisor_ref = None

    def set_supervisor(self, supervisor):
        self.supervisor_ref = supervisor

    async def handle_message(self, message: AgentMessage):
        # Handle configuration or runtime adjustments from Supervisor
        if message.message_type == "configure":
            self.depth_levels = message.data.get("depth_levels", self.depth_levels)
            self.trade_window_ms = message.data.get("trade_window_seconds", 10) * 1000
            self.logger.info(f"Configured: depth={self.depth_levels}, window={self.trade_window_ms}ms")

    async def track_trades(self, exchange, symbol: str):
        """Task: Stream executed trades via CCXT Pro and keep a local history window"""
        self.logger.info(f"Subscribed to trades stream for {symbol}")
        try:
            while self.is_running:
                trades = await exchange.watch_trades(symbol)
                now_ms = exchange.milliseconds()
                for trade in trades:
                    self.recent_trades.append({
                        'timestamp': trade['timestamp'],
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'side': trade['side']  # 'buy' or 'sell'
                    })
                
                # Prune old trades
                while self.recent_trades and (now_ms - self.recent_trades[0]['timestamp'] > self.trade_window_ms):
                    self.recent_trades.popleft()
                
                await asyncio.sleep(0.01)
        except Exception as e:
            self.logger.error(f"Error streaming trades: {e}")
            if self.supervisor_ref:
                await self.send_message(self.supervisor_ref, "error", {"source": "trades_stream", "error": str(e)})

    async def watch_order_book(self, exchange, symbol: str):
        """Task: Stream L2 order books, compute dynamic OBI metrics, and send results to Supervisor"""
        self.logger.info(f"Subscribed to order book stream for {symbol}")
        try:
            while self.is_running:
                orderbook = await exchange.watch_order_book(symbol)
                bids = orderbook['bids']
                asks = orderbook['asks']

                if len(bids) < self.depth_levels or len(asks) < self.depth_levels:
                    await asyncio.sleep(0.1)
                    continue

                # Calculate metrics (calculate_obi tool functionality)
                current_bid_vol = sum([bid[1] for bid in bids[:self.depth_levels]])
                current_ask_vol = sum([ask[1] for ask in asks[:self.depth_levels]])
                mid_price = (bids[0][0] + asks[0][0]) / 2

                raw_obi = (current_bid_vol - current_ask_vol) / (current_bid_vol + current_ask_vol)

                # Get sliding window market execution volume
                market_buy_vol = sum([t['amount'] for t in self.recent_trades if t['side'] == 'buy'])
                market_sell_vol = sum([t['amount'] for t in self.recent_trades if t['side'] == 'sell'])

                # Dispatch stats package to Supervisor Agent
                if self.supervisor_ref:
                    await self.send_message(self.supervisor_ref, "market_metrics", {
                        "symbol": symbol,
                        "mid_price": mid_price,
                        "raw_obi": raw_obi,
                        "current_bid_volume": current_bid_vol,
                        "current_ask_volume": current_ask_vol,
                        "market_buy_volume": market_buy_vol,
                        "market_sell_volume": market_sell_vol
                    })

                await asyncio.sleep(0.01)
        except Exception as e:
            self.logger.error(f"Error streaming order book: {e}")
            if self.supervisor_ref:
                await self.send_message(self.supervisor_ref, "error", {"source": "orderbook_stream", "error": str(e)})
