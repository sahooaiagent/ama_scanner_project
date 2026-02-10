# AMA Pro Crypto Market Scanner User Guide

This guide explains how to use the Python-based AMA Pro Market Scanner.

## Overview
The scanner connects to Binance Futures, fetches the top 500 USDT-margined perpetual pairs by volume, and applies the "AMA Pro" Adaptive Trading System logic to identify **LONG** or **SHORT** signals across multiple timeframes.

## Features
- **Exchange**: Binance Futures (USDT-M)
- **Timeframes**: 1h, 2h, 3h, 4h, 6h, 8h, 12h, 1d
- **Logic**: 
    - **Adaptive EMAs** (Fast/Slow) based on Market Regime (Volatility, Trend, ADX).
    - **Regime Filters**: Filters out signals that contradict the broader market regime.
    - **Calculations**: Fully internal implementation using `pandas` and `numpy` (no external `pandas-ta` dependency required).
- **Output**: 
    - Real-time console output of found signals.
    - Saves all results to a CSV file (e.g., `ama_pro_scan_results_YYYYMMDD_HHMMSS.csv`).

## Prerequisites
- Python 3.7+
- Internet connection (to reach Binance API)

## Installation
1. Ensure the required libraries are installed:
   ```bash
   pip install ccxt pandas numpy
   ```
   *Note: If you encounter SSL issues on macOS, you may need `pip install "urllib3<2"`.*

## Usage
Run the script from your terminal:
```bash
python3 ama_pro_scanner.py
```

## interpreting Results
The scanner will output signals in the format:
```
!!! SIGNAL FOUND: ETH/USDT:USDT [3h] -> SHORT
```
- **Crypto Name**: The symbol (e.g., ETH/USDT:USDT)
- **Timeperiod**: The timeframe where the signal was detected (e.g., 3h)
- **Signal**: LONG or SHORT

## Customization
You can modify the `ama_pro_scanner.py` file to adjust:
- `SYMBOL_LIMIT`: Number of top pairs to scan (default: 500).
- `TIMEFRAMES`: List of timeframes to scan.
- `i_...` variables in `apply_ama_pro_logic` to tune the strategy parameters.
