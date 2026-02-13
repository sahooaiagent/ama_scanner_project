const API_URL = 'http://localhost:8000';

const timeframeDropdown = document.getElementById('timeframe-dropdown');
const selectedTags = document.getElementById('selected-tags');
const timeframeOptions = document.getElementById('timeframe-options');
const symbolLimit = document.getElementById('symbol-limit');
const scheduleTime = document.getElementById('schedule-time');
const runBtn = document.getElementById('run-btn');
const scheduleBtn = document.getElementById('schedule-btn');
const logOutput = document.getElementById('log-output');
const refreshResultsBtn = document.getElementById('refresh-results');
const clearLogsBtn = document.getElementById('clear-logs-btn');
const jobsList = document.getElementById('jobs-list');

let logInterval = null;
const tfMap = {
    "15min": "15m", "30min": "30m", "45min": "45m",
    "1hour": "1h", "2hour": "2h", "4hour": "4h",
    "12hour": "12h", "1day": "1d", "2day": "2d",
    "1week": "1w", "1Month": "1M"
};
const reverseTfMap = Object.fromEntries(Object.entries(tfMap).map(([k, v]) => [v, k]));
let allTimeframes = Object.keys(tfMap);
let selectedTFs = new Set(['1hour', '4hour']);

async function init() {
    renderOptions();
    renderTags();

    await fetchConfig();
    await fetchResults();
    await fetchJobs();
    startLogPolling();

    // Toggling dropdown
    document.addEventListener('click', (e) => {
        if (!timeframeDropdown.contains(e.target)) {
            timeframeOptions.classList.remove('active');
        }
    });

    selectedTags.addEventListener('click', () => {
        timeframeOptions.classList.toggle('active');
    });

    // Event delegation for options
    timeframeOptions.addEventListener('click', (e) => {
        const option = e.target.closest('.option');
        if (!option) return;

        const val = option.dataset.value;
        if (selectedTFs.has(val)) {
            selectedTFs.delete(val);
        } else {
            selectedTFs.add(val);
        }
        renderOptions();
        renderTags();
    });

    // Buttons
    runBtn.addEventListener('click', runScan);
    scheduleBtn.addEventListener('click', scheduleScan);
    refreshResultsBtn.addEventListener('click', async () => {
        const icon = refreshResultsBtn.querySelector('i') || refreshResultsBtn;
        icon.classList.add('fa-spin');
        await fetchResults();
        setTimeout(() => icon.classList.remove('fa-spin'), 600);
    });

    clearLogsBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to clear all logs?')) {
            try {
                const response = await fetch(`${API_URL}/clear-logs`, { method: 'POST' });
                if (response.ok) {
                    logOutput.innerHTML = '<div class="log-entry" style="color: var(--accent-blue)">Logs cleared.</div>';
                }
            } catch (error) {
                console.error('Error clearing logs:', error);
            }
        }
    });

    // Auto-refresh
    setInterval(fetchResults, 15000);
    setInterval(fetchJobs, 20000);
}

async function fetchConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        const data = await response.json();
        if (data.available_timeframes) {
            allTimeframes = data.available_timeframes.map(tf => reverseTfMap[tf] || tf);
            renderOptions();
        }
    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

async function fetchResults() {
    try {
        const response = await fetch(`${API_URL}/results?t=${Date.now()}`);
        const data = await response.json();
        renderResults(data.results || []);
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

function renderResults(results) {
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';

    if (!results || results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 40px; color: var(--text-secondary);">No signals detected yet.</td></tr>';
        return;
    }

    results.forEach(res => {
        const row = document.createElement('tr');
        const signal = (res.Signal || "N/A").toString();
        const signalClass = signal.toLowerCase().includes('long') ? 'signal-long' : 'signal-short';

        row.innerHTML = `
            <td><strong>${res['Crypto Name'] || "Unknown"}</strong></td>
            <td><span class="badge-tf">${res.Timeperiod || "N/A"}</span></td>
            <td class="${signalClass}">${signal}</td>
            <td>${res.Angle || 0}°</td>
            <td style="font-size: 0.8rem; color: var(--text-secondary)">${res.Timestamp || "N/A"}</td>
        `;
        tbody.appendChild(row);
    });
}

function renderOptions() {
    timeframeOptions.innerHTML = allTimeframes.map(tf => `
        <div class="option ${selectedTFs.has(tf) ? 'selected' : ''}" data-value="${tf}">
            <span>${tf}</span>
            ${selectedTFs.has(tf) ? '<i class="fas fa-check"></i>' : ''}
        </div>
    `).join('');
}

function renderTags() {
    if (selectedTFs.size === 0) {
        selectedTags.innerHTML = '<span class="placeholder">Select timeframes...</span>';
        return;
    }
    selectedTags.innerHTML = Array.from(selectedTFs).map(tf => `
        <span class="tag">
            ${tf} <i class="fas fa-times" onclick="removeTf(event, '${tf}')"></i>
        </span>
    `).join('');
}

window.removeTf = (e, tf) => {
    e.stopPropagation();
    selectedTFs.delete(tf);
    renderOptions();
    renderTags();
};

async function runScan() {
    const limit = symbolLimit.value;
    const timeframes = Array.from(selectedTFs).map(tf => tfMap[tf] || tf);

    if (timeframes.length === 0) {
        alert('Please select at least one timeframe');
        return;
    }

    runBtn.disabled = true;
    const originalHtml = runBtn.innerHTML;
    runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';

    try {
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                timeframes,
                symbol_limit: parseInt(limit)
            })
        });
        if (response.ok) {
            logOutput.innerHTML += '<div class="log-entry" style="color: var(--success)">[SYSTEM] Scan started successfully.</div>';
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    } catch (error) {
        console.error('Error starting scan:', error);
    } finally {
        setTimeout(() => {
            runBtn.disabled = false;
            runBtn.innerHTML = originalHtml;
        }, 3000);
    }
}

async function scheduleScan() {
    const time = scheduleTime.value;
    const limit = symbolLimit.value;
    const timeframes = Array.from(selectedTFs).map(tf => tfMap[tf] || tf);

    if (!time) {
        alert('Please select a date and time');
        return;
    }
    if (timeframes.length === 0) {
        alert('Please select at least one timeframe');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                timeframes,
                symbol_limit: parseInt(limit),
                schedule_time: new Date(time).toISOString()
            })
        });
        if (response.ok) {
            alert('Scan scheduled successfully!');
            fetchJobs();
        }
    } catch (error) {
        console.error('Error scheduling scan:', error);
    }
}

async function fetchJobs() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        renderJobs(data.jobs || []);
    } catch (error) {
        console.error('Error fetching jobs:', error);
    }
}

function renderJobs(jobs) {
    jobsList.innerHTML = jobs.length === 0
        ? '<div class="no-jobs">No upcoming scheduled tasks.</div>'
        : jobs.map(job => `
            <div class="job-item glass">
                <div class="job-info">
                    <h4><i class="fas fa-clock"></i> Next Scan</h4>
                    <p>${new Date(job.next_run_time).toLocaleString()}</p>
                </div>
                <button class="cancel-btn">Cancel</button>
            </div>
        `).join('');
}

function startLogPolling() {
    if (logInterval) clearInterval(logInterval);
    logInterval = setInterval(fetchLogs, 2000);
}

async function fetchLogs() {
    try {
        const response = await fetch(`${API_URL}/logs?t=${Date.now()}`);
        const data = await response.json();
        if (data.logs) {
            const wasScrolledToBottom = logOutput.scrollHeight - logOutput.clientHeight <= logOutput.scrollTop + 1;

            const processedLogs = data.logs.split('\n').map(line => {
                if (!line.trim()) return '';
                let style = '';
                if (line.includes('SIGNAL FOUND')) style = 'color: var(--success); font-weight: bold;';
                if (line.includes('ERROR') || line.includes('SyntaxError') || line.includes('failed')) style = 'color: var(--danger);';
                if (line.includes('Starting Scan')) style = 'color: var(--accent-blue); border-top: 1px dotted var(--glass-border); padding-top: 5px; margin-top: 5px;';
                return `<div class="log-entry" style="${style}">${line}</div>`;
            }).join('');

            logOutput.innerHTML = processedLogs;

            if (wasScrolledToBottom) {
                logOutput.scrollTop = logOutput.scrollHeight;
            }
        }
    } catch (error) {
        console.error('Error fetching logs:', error);
    }
}

init();
