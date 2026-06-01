import urllib.request
import json
import sys
from typing import List, Dict

# Map common symbols to CoinGecko IDs
SYMBOL_TO_CG_ID = {
    "POPCAT": "popcat",
    "BOME": "book-of-meme",
    "WIF": "dogwifhat",
    "SOL": "solana"
}

def fetch_top_5_markets(symbol: str) -> List[Dict]:
    """
    Fetches the top 5 markets (exchanges and pairs) for a given symbol sorted by 24h volume.
    """
    base = symbol.split('/')[0].upper()
    coin_id = SYMBOL_TO_CG_ID.get(base)
    if not coin_id:
        # Fallback to search if not in hardcoded map
        search_url = f"https://api.coingecko.com/api/v3/search?query={base}"
        try:
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                search_data = json.loads(resp.read().decode())
                coins = search_data.get("coins", [])
                if coins:
                    coin_id = coins[0]["id"]
                else:
                    return []
        except Exception as e:
            print(f"Error searching coin: {e}", file=sys.stderr)
            return []

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/tickers"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            tickers = data.get("tickers", [])
            
            markets = []
            seen = set()
            for t in tickers:
                market_name = t.get("market", {}).get("name", "")
                target = t.get("target", "")
                base_token = t.get("base", "")
                pair = f"{base_token}/{target}"
                volume_usd = float(t.get("converted_volume", {}).get("usd", 0.0))
                is_anomaly = t.get("is_anomaly", False)
                is_stale = t.get("is_stale", False)
                
                # Filter duplicates and bad tickers
                key = (market_name.lower(), pair.lower())
                if market_name and volume_usd > 0 and not is_anomaly and not is_stale and key not in seen:
                    seen.add(key)
                    markets.append({
                        "exchange": market_name,
                        "pair": pair,
                        "volume_24h_usd": volume_usd,
                        "trust_score": str(t.get("trust_score", "green") or "green")
                    })
            
            # Sort by volume descending
            markets.sort(key=lambda m: m["volume_24h_usd"], reverse=True)
            return markets[:5]
    except Exception as e:
        print(f"Error fetching tickers: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BOME"
    print(f"Fetching top 5 markets for {sym.upper()}...\n")
    top_5 = fetch_top_5_markets(sym)
    if not top_5:
        print("No markets found or API request failed.")
    else:
        print(f"{'Exchange':<25} | {'Pair':<15} | {'24h Volume (USD)':<22} | {'Trust Score':<12}")
        print("-" * 80)
        for m in top_5:
            vol_str = f"${m['volume_24h_usd']:,.2f}"
            print(f"{m['exchange']:<25} | {m['pair']:<15} | {vol_str:<22} | {m['trust_score']:<12}")
