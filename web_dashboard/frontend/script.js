const API_URL = 'http://localhost:8000';

const timeframeDropdown = document.getElementById('timeframe-dropdown');
const selectedTags = document.getElementById('selected-tags');
const timeframeOptions = document.getElementById('timeframe-options');
const exchangeSelect = document.getElementById('exchange-select');
const symbolLimit = document.getElementById('symbol-limit');
const scheduleTime = document.getElementById('schedule-time');
const runBtn = document.getElementById('run-btn');
const scheduleBtn = document.getElementById('schedule-btn');
const logOutput = document.getElementById('log-output');
const refreshResultsBtn = document.getElementById('refresh-results');
const clearLogsBtn = document.getElementById('clear-logs-btn');
const jobsList = document.getElementById('jobs-list');
const searchInput = document.getElementById('search-input');
const signalFilter = document.getElementById('signal-filter');
const timeframeFilter = document.getElementById('timeframe-filter');
const exportCsvBtn = document.getElementById('export-csv');
const resultsCount = document.getElementById('results-count');

let logInterval = null;
let allResults = [];
let currentSort = { column: null, direction: 'asc' };
const tfMap = {
    "15min": "15m", "30min": "30m", "45min": "45m",
    "1hour": "1h", "2hour": "2h", "4hour": "4h",
    "12hour": "12h", "1day": "1d", "2day": "2d",
    "1week": "1w", "1Month": "1M"
};
const reverseTfMap = Object.fromEntries(Object.entries(tfMap).map(([k, v]) => [v, k]));
let allTimeframes = Object.keys(tfMap);
let selectedTFs = new Set(['1hour', '4hour']);

// Toast Notification Function
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'info': 'fa-info-circle',
        'warning': 'fa-exclamation-triangle'
    };

    toast.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 10);

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

async function init() {
    renderOptions();
    renderTags();

    await fetchConfig();
    await fetchResults();
    await fetchJobs();
    startLogPolling();

    // Sidebar navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.dataset.section;
            document.getElementById(section)?.scrollIntoView({ behavior: 'smooth' });

            // Update active state
            document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });

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
        // Clear display immediately
        logOutput.innerHTML = '<div class="log-entry" style="color: var(--accent-blue)">Logs cleared.</div>';

        // Also clear backend logs
        try {
            await fetch(`${API_URL}/clear-logs`, { method: 'POST' });
            showToast('Logs cleared successfully', 'success');
        } catch (error) {
            console.error('Error clearing logs:', error);
            showToast('Failed to clear logs', 'error');
        }
    });

    // Search and Filter Event Listeners
    searchInput.addEventListener('input', filterAndDisplayResults);
    signalFilter.addEventListener('change', filterAndDisplayResults);
    timeframeFilter.addEventListener('change', filterAndDisplayResults);

    // Export CSV
    exportCsvBtn.addEventListener('click', exportToCSV);

    // Table Sorting
    document.querySelectorAll('.sortable').forEach(header => {
        header.addEventListener('click', () => {
            const column = header.dataset.sort;
            sortResults(column);
        });
    });

    // Auto-refresh
    setInterval(fetchResults, 15000);
    setInterval(fetchJobs, 20000);
    setInterval(checkScanStatus, 5000);
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
        allResults = data.results || [];

        // Reapply current sort if one is active
        if (currentSort.column) {
            applySortToResults();
            updateSortIcons();
        }

        filterAndDisplayResults();
    } catch (error) {
        console.error('Error fetching results:', error);
        showToast('Failed to fetch results', 'error');
    }
}

function filterAndDisplayResults() {
    let filtered = allResults;

    // Apply search filter
    const searchTerm = searchInput.value.toLowerCase();
    if (searchTerm) {
        filtered = filtered.filter(r =>
            (r['Crypto Name'] || '').toLowerCase().includes(searchTerm)
        );
    }

    // Apply signal filter
    const signal = signalFilter.value;
    if (signal !== 'all') {
        filtered = filtered.filter(r => r.Signal === signal);
    }

    // Apply timeframe filter
    const tf = timeframeFilter.value;
    if (tf !== 'all') {
        filtered = filtered.filter(r => r.Timeperiod === tf);
    }

    renderResults(filtered);
    updateResultsCount(filtered.length, allResults.length);
    updateStats();
}

function updateStats() {
    const totalSignals = allResults.length;
    const longSignals = allResults.filter(r => r.Signal === 'LONG').length;
    const shortSignals = allResults.filter(r => r.Signal === 'SHORT').length;

    // Animated counter
    animateValue('total-signals', 0, totalSignals, 1000);
    animateValue('long-signals', 0, longSignals, 1000);
    animateValue('short-signals', 0, shortSignals, 1000);

    // Update last scan time
    if (allResults.length > 0) {
        const lastScan = allResults[allResults.length - 1].Timestamp;
        if (lastScan) {
            const scanDate = new Date(lastScan);
            const now = new Date();
            const diffMs = now - scanDate;
            const diffMins = Math.floor(diffMs / 60000);

            let timeAgo;
            if (diffMins < 1) timeAgo = 'Just now';
            else if (diffMins < 60) timeAgo = `${diffMins}m ago`;
            else if (diffMins < 1440) timeAgo = `${Math.floor(diffMins / 60)}h ago`;
            else timeAgo = `${Math.floor(diffMins / 1440)}d ago`;

            document.getElementById('last-scan').textContent = timeAgo;
        }
    }
}

function animateValue(id, start, end, duration) {
    const element = document.getElementById(id);
    if (!element) return;

    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

function openTradingView(symbol, timeframe) {
    // Remove /USDT or :USDT suffix and clean symbol
    const cleanSymbol = symbol.replace(/\/USDT.*|:USDT.*/g, '');

    // Map timeframes to TradingView format
    const tvTimeframes = {
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '12h': '720',
        '1d': 'D',
        '2d': '2D',
        '1w': 'W',
        '1M': 'M'
    };

    const tvTF = tvTimeframes[timeframe] || '60';
    const url = `https://www.tradingview.com/chart/?symbol=BINANCE:${cleanSymbol}USDT&interval=${tvTF}`;

    window.open(url, '_blank');
    showToast(`Opening ${cleanSymbol} on TradingView`, 'info');
}

function updateResultsCount(filtered, total) {
    if (filtered === total) {
        resultsCount.textContent = `Showing ${total} result${total !== 1 ? 's' : ''}`;
    } else {
        resultsCount.textContent = `Showing ${filtered} of ${total} result${total !== 1 ? 's' : ''}`;
    }
}

function renderResults(results) {
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';

    if (!results || results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 40px; color: var(--text-secondary);">No signals detected yet.</td></tr>';
        return;
    }

    results.forEach(res => {
        const row = document.createElement('tr');
        const signal = (res.Signal || "N/A").toString();
        const signalClass = signal.toLowerCase().includes('long') ? 'signal-long' : 'signal-short';

        // Parse daily change
        const dailyChange = res['Daily Change'] || 'N/A';
        let dailyChangeClass = '';
        if (dailyChange !== 'N/A') {
            const changeValue = parseFloat(dailyChange);
            dailyChangeClass = changeValue >= 0 ? 'daily-change-positive' : 'daily-change-negative';
        }

        row.innerHTML = `
            <td><strong>${res['Crypto Name'] || "Unknown"}</strong></td>
            <td><span class="badge-tf">${res.Timeperiod || "N/A"}</span></td>
            <td class="${signalClass}">${signal}</td>
            <td>${res.Angle || 0}</td>
            <td class="${dailyChangeClass}">${dailyChange}</td>
            <td style="font-size: 0.8rem; color: var(--text-secondary)">${res.Timestamp || "N/A"}</td>
            <td>
                <button class="tradingview-btn" onclick="openTradingView('${res['Crypto Name']}', '${res.Timeperiod}')">
                    <i class="fas fa-chart-bar"></i> Chart
                </button>
            </td>
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
    const exchange = exchangeSelect.value;
    const timeframes = Array.from(selectedTFs).map(tf => tfMap[tf] || tf);

    if (timeframes.length === 0) {
        showToast('Please select at least one timeframe', 'warning');
        return;
    }

    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';

    // Add SCAN IN PROGRESS message
    const exchangeName = exchange.toUpperCase();
    logOutput.innerHTML = `<div class="log-entry" style="color: var(--accent-blue); font-weight: bold; border-top: 2px solid var(--accent-blue); padding-top: 10px; margin-top: 10px;">🔄 SCAN IN PROGRESS on ${exchangeName}...</div>`;
    logOutput.scrollTop = logOutput.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exchange,
                timeframes,
                symbol_limit: parseInt(limit)
            })
        });
        if (response.ok) {
            logOutput.innerHTML += '<div class="log-entry" style="color: var(--success)">[SYSTEM] Scan started successfully.</div>';
            logOutput.scrollTop = logOutput.scrollHeight;
            showToast('Scan started successfully', 'success');
        } else {
            showToast('Failed to start scan', 'error');
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run Scanner Now';
        }
    } catch (error) {
        console.error('Error starting scan:', error);
        showToast('Error starting scan', 'error');
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fas fa-play"></i> Run Scanner Now';
    }
}

async function scheduleScan() {
    const time = scheduleTime.value;
    const limit = symbolLimit.value;
    const exchange = exchangeSelect.value;
    const timeframes = Array.from(selectedTFs).map(tf => tfMap[tf] || tf);

    if (!time) {
        showToast('Please select a date and time', 'warning');
        return;
    }
    if (timeframes.length === 0) {
        showToast('Please select at least one timeframe', 'warning');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exchange,
                timeframes,
                symbol_limit: parseInt(limit),
                schedule_time: new Date(time).toISOString()
            })
        });
        if (response.ok) {
            showToast('Scan scheduled successfully!', 'success');
            scheduleTime.value = '';
            fetchJobs();
        } else {
            showToast('Failed to schedule scan', 'error');
        }
    } catch (error) {
        console.error('Error scheduling scan:', error);
        showToast('Error scheduling scan', 'error');
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
    if (jobs.length === 0) {
        jobsList.innerHTML = '<div class="no-jobs">No upcoming scheduled tasks.</div>';
        return;
    }

    jobsList.innerHTML = jobs.map(job => {
        const scheduledDate = new Date(job.next_run_time);
        const exchange = (job.exchange || 'binance').toUpperCase();
        const tfList = job.timeframes ? job.timeframes.join(', ') : 'N/A';

        return `
            <div class="job-item glass">
                <div class="job-info">
                    <h4><i class="fas fa-clock"></i> Scheduled Scan</h4>
                    <p><strong>Time:</strong> ${scheduledDate.toLocaleString()}</p>
                    <p><strong>Exchange:</strong> ${exchange}</p>
                    <p><strong>Symbols:</strong> ${job.symbol_limit || 'N/A'}</p>
                    <p><strong>Timeframes:</strong> ${tfList}</p>
                </div>
                <button class="cancel-btn" onclick="cancelJob('${job.id}')">
                    <i class="fas fa-times"></i> Cancel
                </button>
            </div>
        `;
    }).join('');
}

async function cancelJob(jobId) {
    try {
        const response = await fetch(`${API_URL}/cancel-job/${jobId}`, { method: 'POST' });
        if (response.ok) {
            showToast('Scheduled scan cancelled', 'success');
            fetchJobs();
        } else {
            showToast('Failed to cancel scan', 'error');
        }
    } catch (error) {
        console.error('Error cancelling job:', error);
        showToast('Error cancelling scan', 'error');
    }
}

function sortResults(column) {
    if (currentSort.column === column) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = column;
        currentSort.direction = 'asc';
    }

    // Update sort icons
    updateSortIcons();

    // Apply sort and refresh display
    applySortToResults();
    filterAndDisplayResults();
}

function updateSortIcons() {
    document.querySelectorAll('.sort-icon').forEach(icon => {
        icon.className = 'fas fa-sort sort-icon';
    });

    if (currentSort.column) {
        const header = document.querySelector(`[data-sort="${currentSort.column}"]`);
        if (header) {
            const icon = header.querySelector('.sort-icon');
            icon.className = `fas fa-sort-${currentSort.direction === 'asc' ? 'up' : 'down'} sort-icon active`;
        }
    }
}

function applySortToResults() {
    if (!currentSort.column) return;

    allResults.sort((a, b) => {
        let aVal, bVal;

        switch (currentSort.column) {
            case 'symbol':
                aVal = a['Crypto Name'] || '';
                bVal = b['Crypto Name'] || '';
                break;
            case 'timeframe':
                aVal = a.Timeperiod || '';
                bVal = b.Timeperiod || '';
                break;
            case 'signal':
                aVal = a.Signal || '';
                bVal = b.Signal || '';
                break;
            case 'angle':
                aVal = parseFloat(a.Angle) || 0;
                bVal = parseFloat(b.Angle) || 0;
                break;
            case 'change':
                aVal = parseFloat(a['Daily Change']) || 0;
                bVal = parseFloat(b['Daily Change']) || 0;
                break;
            case 'timestamp':
                aVal = new Date(a.Timestamp || 0);
                bVal = new Date(b.Timestamp || 0);
                break;
            default:
                return 0;
        }

        if (aVal < bVal) return currentSort.direction === 'asc' ? -1 : 1;
        if (aVal > bVal) return currentSort.direction === 'asc' ? 1 : -1;
        return 0;
    });
}

function exportToCSV() {
    if (allResults.length === 0) {
        showToast('No results to export', 'warning');
        return;
    }

    const headers = ['Crypto Name', 'Timeperiod', 'Signal', 'Angle', 'Daily Change', 'Timestamp'];
    const csv = [
        headers.join(','),
        ...allResults.map(r => [
            r['Crypto Name'] || '',
            r.Timeperiod || '',
            r.Signal || '',
            r.Angle || '',
            r['Daily Change'] || '',
            r.Timestamp || ''
        ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ama_pro_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);

    showToast('Results exported successfully', 'success');
}

async function checkScanStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();

        if (data.scan_running) {
            runBtn.disabled = true;
            runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
        } else if (runBtn.disabled && !runBtn.innerHTML.includes('Scanning')) {
            // Don't re-enable if it's disabled for another reason
        } else if (!runBtn.innerHTML.includes('Scanning')) {
            runBtn.disabled = false;
            runBtn.innerHTML = '<i class="fas fa-play"></i> Run Scanner Now';
        }
    } catch (error) {
        console.error('Error checking scan status:', error);
    }
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

                // Highlight scan status messages
                if (line.includes('SCAN COMPLETED')) {
                    style = 'color: var(--success); font-weight: bold; border-bottom: 2px solid var(--success); padding-bottom: 10px; margin-bottom: 10px; font-size: 1.1rem;';
                }
                else if (line.includes('Starting') && (line.includes('Scan') || line.includes('scan'))) {
                    style = 'color: var(--accent-blue); font-weight: bold; border-top: 2px solid var(--accent-blue); padding-top: 10px; margin-top: 10px;';
                }
                else if (line.includes('SIGNAL FOUND')) {
                    style = 'color: var(--success); font-weight: bold;';
                }
                else if (line.includes('ERROR') || line.includes('SyntaxError') || line.includes('failed')) {
                    style = 'color: var(--danger);';
                }

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
