const API_URL = 'http://localhost:8000';

const timeframeDropdown = document.getElementById('timeframe-dropdown');
const selectedTags = document.getElementById('selected-tags');
const timeframeOptions = document.getElementById('timeframe-options');
const symbolLimit = document.getElementById('symbol-limit');
const scheduleTime = document.getElementById('schedule-time');
const runBtn = document.getElementById('run-btn');
const scheduleBtn = document.getElementById('schedule-btn');
const logOutput = document.getElementById('log-output');
const resultsTable = document.getElementById('results-table').getElementsByTagName('tbody')[0];
const jobsList = document.getElementById('jobs-list');
const refreshResults = document.getElementById('refresh-results');

let logInterval = null;
const tfMap = {
    "15min": "15m", "30min": "30m", "45min": "45m",
    "1hour": "1h", "2hour": "2h", "4hour": "4h",
    "12hour": "12h", "1day": "1d", "2day": "2d",
    "1week": "1w", "1Month": "1M"
};
let allTimeframes = Object.keys(tfMap);
let selectedTFs = new Set(['1hour', '4hour']);

async function init() {
    renderOptions(); // Initial render with defaults
    renderTags();

    await fetchConfig();
    await fetchResults();
    await fetchJobs();
    startLogPolling();

    // Dropdown toggling
    document.addEventListener('click', (e) => {
        if (!timeframeDropdown.contains(e.target)) {
            timeframeOptions.classList.remove('active');
        }
    });

    timeframeDropdown.addEventListener('click', (e) => {
        // Only toggle if we didn't click an option or a tag-remove icon
        if (!e.target.classList.contains('option') && !e.target.classList.contains('fa-times')) {
            timeframeOptions.classList.toggle('active');
        }
    });

    // Event delegation for options
    timeframeOptions.addEventListener('click', (e) => {
        if (e.target.classList.contains('option')) {
            const val = e.target.getAttribute('data-value');
            if (selectedTFs.has(val)) {
                selectedTFs.delete(val);
            } else {
                selectedTFs.add(val);
            }
            renderOptions();
            renderTags();
        }
    });
}

const reverseTfMap = Object.fromEntries(Object.entries(tfMap).map(([k, v]) => [v, k]));

async function fetchConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        const data = await response.json();
        if (data.available_timeframes) {
            allTimeframes = data.available_timeframes.map(tf => reverseTfMap[tf] || tf);
            renderOptions();
            renderTags();
        }
    } catch (error) {
        console.error('Error fetching config:', error);
        // We still have fallback values, so it's fine
    }
}

function renderOptions() {
    timeframeOptions.innerHTML = allTimeframes.map(tf => `
        <div class="option ${selectedTFs.has(tf) ? 'selected' : ''}" data-value="${tf}">
            ${tf}
        </div>
    `).join('');
}

function renderTags() {
    if (selectedTFs.size === 0) {
        selectedTags.innerHTML = '<span class="placeholder">Select timeframes...</span>';
    } else {
        selectedTags.innerHTML = Array.from(selectedTFs).map(tf => `
            <div class="tag">
                ${tf} <i class="fas fa-times" onclick="removeTag(event, '${tf}')"></i>
            </div>
        `).join('');
    }
}

window.removeTag = (e, tf) => {
    e.stopPropagation();
    selectedTFs.delete(tf);
    renderOptions();
    renderTags();
};

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

async function triggerAction(isSchedule = false) {
    const timeframes = Array.from(selectedTFs).map(label => tfMap[label] || label);
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
