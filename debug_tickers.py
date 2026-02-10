import ccxt

def debug_tickers():
    try:
        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        tickers = exchange.fetch_tickers()
        print(f"Total tickers: {len(tickers)}")
        print("First 10 keys:")
        for k in list(tickers.keys())[:10]:
            print(f"Key: {k}, Symbol: {tickers[k]['symbol']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_tickers()
