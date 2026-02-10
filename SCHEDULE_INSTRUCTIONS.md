# How to Schedule the Scanner to Run Daily (macOS)

You can use the built-in Mac utility called `cron` to run your scanner automatically every day.

## Step 1: Create a Runner Script
To ensure things run smoothly, we will create a small "launcher" script that sets the correct folder.

1.  Open Terminal.
2.  Run this command to create the launcher:
    ```bash
    echo '#!/bin/bash
    cd /Users/sushantakumarsahoo/Downloads/sahooaiagent
    /usr/bin/python3 ama_pro_scanner.py >> scan_log.txt 2>&1' > /Users/sushantakumarsahoo/Downloads/sahooaiagent/run_daily.sh
    ```
3.  Make it executable:
    ```bash
    chmod +x /Users/sushantakumarsahoo/Downloads/sahooaiagent/run_daily.sh
    ```

## Step 2: Set the Schedule
We will now tell your Mac to run this script every day at a specific time (e.g., 9:00 AM).

1.  In Terminal, type:
    ```bash
    crontab -e
    ```
2.  If it asks you to choose an editor, press `1` (for nano) or use the default.
3.  Scroll to the bottom and paste this line:
    ```bash
    00 09 * * * /Users/sushantakumarsahoo/Downloads/sahooaiagent/run_daily.sh
    ```
    *(Note: `00 09` means 9:00 AM. You can change it to `30 14` for 2:30 PM, etc.)*
4.  **To Save**:
    - If using **nano**: Press `Control + O`, then `Enter`, then `Control + X`.
    - If using **vim**: Type `:wq` and press `Enter`.

## Step 3: Verify it is Scheduled
Run this command to see your active schedules:
```bash
crontab -l
```

## Important Notes:
- **Sleep Mode**: Your Mac must be awake for this to run. If it's asleep, it will skip the task. You can prevent sleep in **System Settings > Energy Saver**.
- **Logs**: The scanner will save its progress to `scan_log.txt` in the project folder. You can check it anytime to see if it ran successfully.
- **Results**: The CSV files will still be created in the `Downloads/sahooaiagent` folder as usual.
