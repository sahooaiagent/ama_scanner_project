## How to Run the AMA Pro Crypto Scanner Locally (MEXC Edition)

Follow these steps to run the scanner on your machine.

## Prerequisites

1.  **Terminal**: You need to use the "Terminal" application on your Mac.
2.  **Python 3**: Ensure Python 3 is installed. You can check by running `python3 --version`.
3.  **MEXC Account**: The scanner now uses MEXC for data.

## Step-by-Step Instructions

### 1. Open Terminal
Press `Command + Space`, type "Terminal", and hit Enter.

### 2. Navigate to the Project Folder
Copy and paste the following command into your terminal and press Enter:
```bash
cd /Users/sushantakumarsahoo/Downloads/sahooaiagent
```

### 3. Install Dependencies
Copy and paste this command to install the required libraries:
```bash
python3 -m pip install -r requirements.txt
```
*Note: If you see SSL errors, run this command: `python3 -m pip install "urllib3<2"`*

### 4. Run the Scanner
Now, execute the scanner script:
```bash
python3 ama_pro_scanner.py
```
**New Feature**: You will be prompted to enter how many symbols you want to scan (e.g., 100 or 500). Just type the number and press Enter.

The scanner will start fetching data from **MEXC** and printing signals to the screen.

### 5. Check Results
- **Real-time signals** will appear in the terminal (e.g., `!!! SIGNAL FOUND...`).
- **Full report**: Once the scan finishes (or if you stop it with `Ctrl+C`), a CSV file will be saved in the same folder with a name like `ama_pro_scan_results_YYYYMMDD_...csv`.

## Troubleshooting
- **Permission Denied**: If you get a permission error, try adding `sudo` before the command (e.g., `sudo python3 ...`) and enter your password.
- **Module Not Found**: Re-run the installation step (Step 3).
