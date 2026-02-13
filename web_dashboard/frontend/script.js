const API_URL = 'http://localhost:8000';

const timeframeSelection = document.getElementById('timeframe-selection');
const symbolLimit = document.getElementById('symbol-limit');
const scheduleTime = document.getElementById('schedule-time');
const runBtn = document.getElementById('run-btn');
const scheduleBtn = document.getElementById('schedule-btn');
const logOutput = document.getElementById('log-output');
const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
const jobsList = document.getElementById('jobs-list');
const refreshResults = document.getElementById('refresh-results');

let logInterval = null;

async function init() {
    await fetchConfig();
    await fetchResults();
    await fetchJobs();
    startLogPolling();
}

async function fetchConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        const data = await response.json();

        timeframeSelection.innerHTML = data.available_timeframes.map(tf => `
            <div>
                <input type="checkbox" id="tf-${tf}" name="timeframe" value="${tf}" class="tf-checkbox" ${['1h', '2h', '4h'].includes(tf) ? 'checked' : ''}>
                <label for="tf-${tf}" class="tf-label">${tf}</label>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

async function fetchResults() {
    try {
        const response = await fetch(`${API_URL}/results`);
        const data = await response.json();

        if (data.results) {
            resultsTable.innerHTML = data.results.map(res => `
                <tr>
                    <td>${res['Crypto Name']}</td>
                    <td>${res['Timeperiod']}</td>
                    <td class="${res['Signal'] === 'LONG' ? 'signal-long' : 'signal-short'}">${res['Signal']}</td>
                    <td>${res['Angle'] || 'N/A'}</td>
                    <td>${res['Timestamp']}</td>
                </tr>
            `).join('');
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

async function fetchLogs() {
    try {
        const response = await fetch(`${API_URL}/logs`);
        const data = await response.json();
        if (data.logs) {
            logOutput.innerText = data.logs;
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

async function fetchJobs() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();

        jobsList.innerHTML = data.jobs.map(job => `
            <div class="job-item">
                <div class="job-info">
                    <h4>Scanner Run</h4>
                    <p>Next: ${job.next_run_time}</p>
                </div>
                <button class="cancel-btn">Cancel</button>
            </div>
        `).join('') || '<p style="color: grey; font-size: 0.9rem;">No active schedules</p>';
    } catch (error) {
        console.error('Error fetching jobs:', error);
    }
}

function startLogPolling() {
    if (logInterval) clearInterval(logInterval);
    logInterval = setInterval(fetchLogs, 2000);
}

function getSelectedTimeframes() {
    return Array.from(document.querySelectorAll('input[name="timeframe"]:checked')).map(cb => cb.value);
}

async function triggerAction(isSchedule = false) {
    const timeframes = getSelectedTimeframes();
    if (timeframes.length === 0) {
        alert('Please select at least one timeframe.');
        return;
    }

    const payload = {
        timeframes,
        symbol_limit: parseInt(symbolLimit.value),
        schedule_time: isSchedule ? scheduleTime.value : null
    };

    try {
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();

        if (isSchedule) {
            alert(`Scan scheduled for ${data.run_at}`);
            fetchJobs();
        } else {
            console.log('Immediate scan started');
        }
    } catch (error) {
        console.error('Error triggering scan:', error);
        alert('Failed to trigger scan. Check backend status.');
    }
}

runBtn.addEventListener('click', () => triggerAction(false));
scheduleBtn.addEventListener('click', () => {
    if (!scheduleTime.value) {
        alert('Please select a date and time for scheduling.');
        return;
    }
    triggerAction(true);
});

refreshResults.addEventListener('click', fetchResults);

// Auto-refresh results and jobs occasionally
setInterval(fetchResults, 10000);
setInterval(fetchJobs, 10000);

init();
