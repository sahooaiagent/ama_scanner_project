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
let allTimeframes = [];
let selectedTFs = new Set(['1h', '4h']);

async function init() {
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

    selectedTags.addEventListener('click', () => {
        timeframeOptions.classList.toggle('active');
    });
}

async function fetchConfig() {
    try {
        const response = await fetch(`${API_URL}/config`);
        const data = await response.json();
        allTimeframes = data.available_timeframes;
        renderOptions();
        renderTags();
    } catch (error) {
        console.error('Error fetching config:', error);
    }
}

function renderOptions() {
    timeframeOptions.innerHTML = allTimeframes.map(tf => `
        <div class="option ${selectedTFs.has(tf) ? 'selected' : ''}" data-value="${tf}">
            ${tf}
        </div>
    `).join('');

    document.querySelectorAll('.option').forEach(opt => {
        opt.addEventListener('click', () => {
            const val = opt.getAttribute('data-value');
            if (selectedTFs.has(val)) {
                selectedTFs.delete(val);
            } else {
                selectedTFs.add(val);
            }
            renderOptions();
            renderTags();
        });
    });
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
    const timeframes = Array.from(selectedTFs);
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
