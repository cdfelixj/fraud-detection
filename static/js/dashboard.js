// Dashboard JavaScript functionality

// Global variables
let charts = {};
let refreshInterval;
let simulationInterval;
let isSimulating = false;
let realtimeChart;

// Utility function to safely destroy charts
function destroyChart(chartRef) {
    if (chartRef && typeof chartRef.destroy === 'function') {
        try {
            chartRef.destroy();
        } catch (e) {
            console.warn('Error destroying chart:', e);
        }
    }
    return null;
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startAutoRefresh();
});

// Initialize dashboard components
function initializeDashboard() {
    console.log('Initializing fraud detection dashboard...');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Load initial chart data
    loadChartData();
    
    // Setup real-time updates
    setupRealTimeUpdates();
}

// Setup event listeners
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            refreshDashboard();
        });
    }
    
    // Auto-refresh toggle
    const autoRefreshToggle = document.getElementById('autoRefreshToggle');
    if (autoRefreshToggle) {
        autoRefreshToggle.addEventListener('change', function() {
            if (this.checked) {
                startAutoRefresh();
            } else {
                stopAutoRefresh();
            }
        });
    }
    
    // Alert acknowledgment buttons
    document.querySelectorAll('.acknowledge-alert').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            acknowledgeAlert(this.dataset.alertId);
        });
    });
    
    // Model training button
    const trainBtn = document.getElementById('trainModelsBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', function() {
            trainModels();
        });
    }
    
    // Prediction button
    const predictBtn = document.getElementById('runPredictionsBtn');
    if (predictBtn) {
        predictBtn.addEventListener('click', function() {
            runPredictions();
        });
    }
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Load and display chart data
function loadChartData() {
    fetch('/api/chart-data')
        .then(response => response.json())
        .then(data => {
            updateCharts(data);
        })
        .catch(error => {
            console.error('Error loading chart data:', error);
            // Don't show notification for chart data errors to avoid spam
        });
}

// Update all charts with new data
function updateCharts(data) {
    // Update amount distribution chart
    updateAmountDistributionChart(data);
    
    // Update class distribution chart
    updateClassDistributionChart(data);
    
    // Update performance chart if available
    updatePerformanceChart(data);
    
    // Update time series chart
    updateTimeSeriesChart(data);
}

// Update amount distribution chart
function updateAmountDistributionChart(data) {
    const ctx = document.getElementById('amountDistributionChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    charts.amountDistribution = destroyChart(charts.amountDistribution);
    
    // Create amount bins
    const amounts = data.amounts || [];
    const bins = createAmountBins(amounts);
    
    charts.amountDistribution = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins.labels,
            datasets: [{
                label: 'Transaction Count',
                data: bins.counts,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Transactions'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Transaction Amount ($)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Transaction Amount Distribution'
                }
            }
        }
    });
}

// Update class distribution chart
function updateClassDistributionChart(data) {
    const ctx = document.getElementById('classDistributionChart');
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    charts.classDistribution = destroyChart(charts.classDistribution);
    
    const normalCount = data.normal_amounts ? data.normal_amounts.length : 0;
    const fraudCount = data.fraud_amounts ? data.fraud_amounts.length : 0;
    
    charts.classDistribution = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Fraud'],
            datasets: [{
                data: [normalCount, fraudCount],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Class Distribution'
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Update performance metrics chart
function updatePerformanceChart(data) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx || !data.performance_history) return;
    
    // Destroy existing chart if it exists
    charts.performance = destroyChart(charts.performance);
    
    const performanceData = data.performance_history;
    
    charts.performance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: performanceData.map(p => p.date),
            datasets: [
                {
                    label: 'Precision',
                    data: performanceData.map(p => p.precision),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Recall',
                    data: performanceData.map(p => p.recall),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'F1-Score',
                    data: performanceData.map(p => p.f1),
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Score'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Over Time'
                }
            }
        }
    });
}

// Update time series chart
function updateTimeSeriesChart(data) {
    const ctx = document.getElementById('timeSeriesChart');
    if (!ctx || !data.time_series) return;
    
    // Destroy existing chart if it exists
    charts.timeSeries = destroyChart(charts.timeSeries);
    
    const timeSeriesData = data.time_series;
    const normalData = timeSeriesData.filter(d => d.class === 0);
    const fraudData = timeSeriesData.filter(d => d.class === 1);
    
    charts.timeSeries = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Normal Transactions',
                    data: normalData.map(d => ({ x: d.x, y: d.y })),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    pointRadius: 3
                },
                {
                    label: 'Fraud Transactions',
                    data: fraudData.map(d => ({ x: d.x, y: d.y })),
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    pointRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Transaction Order'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Amount ($)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Transaction Pattern Analysis'
                }
            }
        }
    });
}

// Create amount bins for histogram
function createAmountBins(amounts) {
    if (!amounts || amounts.length === 0) {
        return { labels: [], counts: [] };
    }
    
    // Sort amounts to get better distribution
    const sortedAmounts = [...amounts].sort((a, b) => a - b);
    const min = sortedAmounts[0];
    const max = sortedAmounts[sortedAmounts.length - 1];
    
    // Create logarithmic bins for better distribution of transaction amounts
    // Most transactions are small, so we need more granular bins for smaller amounts
    const binRanges = [
        [0, 50],
        [50, 100], 
        [100, 250],
        [250, 500],
        [500, 1000],
        [1000, 2500],
        [2500, 5000],
        [5000, 10000],
        [10000, 25000],
        [25000, Infinity]
    ];
    
    const bins = [];
    const labels = [];
    
    for (let i = 0; i < binRanges.length; i++) {
        const [binStart, binEnd] = binRanges[i];
        
        const count = amounts.filter(amount => 
            amount >= binStart && (binEnd === Infinity ? true : amount < binEnd)
        ).length;
        
        bins.push(count);
        
        if (binEnd === Infinity) {
            labels.push(`$${binStart.toLocaleString()}+`);
        } else {
            labels.push(`$${binStart.toLocaleString()}-$${binEnd.toLocaleString()}`);
        }
    }
    
    return { labels, counts: bins };
}

// Setup real-time updates
function setupRealTimeUpdates() {
    // This could be expanded to use WebSockets for real-time updates
    console.log('Real-time updates initialized');
}

// Start auto-refresh
function startAutoRefresh() {
    stopAutoRefresh(); // Clear any existing interval
    refreshInterval = setInterval(() => {
        refreshDashboard();
    }, 30000); // Refresh every 30 seconds
}

// Stop auto-refresh
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

// Refresh dashboard data
function refreshDashboard() {
    console.log('Refreshing dashboard...');
    loadChartData();
    updateStatistics();
    showNotification('Dashboard refreshed', 'success');
}

// Update statistics
function updateStatistics() {
    // This would fetch updated statistics from the server
    // For now, we'll just update the chart data
    loadChartData();
}

// Train models
function trainModels() {
    const button = document.getElementById('trainModelsBtn');
    if (button) {
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Training...';
        button.disabled = true;
    }
    
    fetch('/train', { method: 'POST' })
        .then(response => {
            if (response.ok) {
                showNotification('Model training started', 'success');
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                throw new Error('Training failed');
            }
        })
        .catch(error => {
            console.error('Error training models:', error);
            showNotification('Error training models', 'danger');
        })
        .finally(() => {
            if (button) {
                button.innerHTML = '<i class="fas fa-brain me-2"></i>Train Models';
                button.disabled = false;
            }
        });
}

// Run predictions
function runPredictions() {
    const button = document.getElementById('runPredictionsBtn');
    if (button) {
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Predicting...';
        button.disabled = true;
    }
    
    fetch('/predict')
        .then(response => {
            if (response.ok) {
                showNotification('Predictions completed', 'success');
                setTimeout(() => {
                    window.location.reload();
                }, 1000);
            } else {
                throw new Error('Prediction failed');
            }
        })
        .catch(error => {
            console.error('Error running predictions:', error);
            showNotification('Error running predictions', 'danger');
        })
        .finally(() => {
            if (button) {
                button.innerHTML = '<i class="fas fa-search me-2"></i>Run Predictions';
                button.disabled = false;
            }
        });
}

// Acknowledge alert
function acknowledgeAlert(alertId) {
    fetch(`/acknowledge-alert/${alertId}`)
        .then(response => {
            if (response.ok) {
                showNotification('Alert acknowledged', 'success');
                // Remove the alert from the UI
                const alertElement = document.querySelector(`[data-alert-id="${alertId}"]`);
                if (alertElement) {
                    alertElement.style.opacity = '0.5';
                    setTimeout(() => {
                        alertElement.remove();
                    }, 500);
                }
            } else {
                throw new Error('Failed to acknowledge alert');
            }
        })
        .catch(error => {
            console.error('Error acknowledging alert:', error);
            showNotification('Error acknowledging alert', 'danger');
        });
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatNumber(number) {
    return new Intl.NumberFormat('en-US').format(number);
}

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
        refreshDashboard();
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
    
    // Destroy all charts
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
});


// Real-time simulation functions
function toggleSimulation() {
    const btn = document.getElementById("simulateBtn");
    const section = document.getElementById("simulationSection");
    
    if (!isSimulating) {
        startSimulation();
        btn.innerHTML = "<i class=\"fas fa-stop me-2\"></i>Stop Simulation";
        btn.className = "btn btn-danger w-100";
        section.style.display = "block";
        isSimulating = true;
    } else {
        stopSimulation();
        btn.innerHTML = "<i class=\"fas fa-play me-2\"></i>Start Simulation";
        btn.className = "btn btn-primary w-100";
        section.style.display = "none";
        isSimulating = false;
    }
}

function startSimulation() {
    initRealtimeChart();
    
    // Fetch new data every 2 seconds
    simulationInterval = setInterval(() => {
        fetch("/api/simulation-data")
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    showSimulationError(data.error);
                    stopSimulation();
                } else {
                    updateRealtimeChart(data);
                }
            })
            .catch(error => {
                console.error("Error fetching simulation data:", error);
                showSimulationError("Failed to fetch transaction data. Please check if data is loaded in the database.");
                stopSimulation();
            });
    }, 1000);
}

function stopSimulation() {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
    
    if (realtimeChart) {
        realtimeChart.destroy();
        realtimeChart = null;
    }
}

function initRealtimeChart() {
    const ctx = document.getElementById("realtimeChart");
    if (!ctx) return;
    
    // Destroy existing chart if it exists
    if (realtimeChart) {
        realtimeChart.destroy();
    }
    
    realtimeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Transaction Amount",
                data: [],
                borderColor: "rgba(54, 162, 235, 1)",
                backgroundColor: "rgba(54, 162, 235, 0.1)",
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: "Fraud Score",
                data: [],
                borderColor: "rgba(255, 99, 132, 1)",
                backgroundColor: "rgba(255, 99, 132, 0.1)",
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                yAxisID: "y1"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            interaction: {
                intersect: false,
            },
            layout: {
                padding: 10
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: "Time"
                    }
                },
                y: {
                    type: "linear",
                    display: true,
                    position: "left",
                    title: {
                        display: true,
                        text: "Amount ($)"
                    },
                    beginAtZero: true,
                    suggestedMax: 5000,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                },
                y1: {
                    type: "linear",
                    display: true,
                    position: "right",
                    title: {
                        display: true,
                        text: "Fraud Score"
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    max: 1,
                    min: 0,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

function updateRealtimeChart(data) {
    if (!realtimeChart || !data) return;
    
    const maxDataPoints = 15;
    
    // Add new data point
    realtimeChart.data.labels.push(data.timestamp);
    realtimeChart.data.datasets[0].data.push(data.amount);
    realtimeChart.data.datasets[1].data.push(data.fraud_score);
    
    // Remove old data points if we have too many
    if (realtimeChart.data.labels.length > maxDataPoints) {
        realtimeChart.data.labels.shift();
        realtimeChart.data.datasets[0].data.shift();
        realtimeChart.data.datasets[1].data.shift();
    }
    
    // Update the simulation status badge
    const statusBadge = document.getElementById("simulationStatus");
    if (statusBadge) {
        statusBadge.textContent = 'Live (Database)';
        statusBadge.className = 'badge bg-success ms-2';
    }
    
    // Update transaction info
    updateTransactionInfo(data);
    
    realtimeChart.update("none");
}

function showSimulationError(errorMessage) {
    // Update status badge to show error
    const statusBadge = document.getElementById("simulationStatus");
    if (statusBadge) {
        statusBadge.textContent = 'Error';
        statusBadge.className = 'badge bg-danger ms-2';
    }
    
    // Show error in transaction info area
    let infoDiv = document.getElementById("transactionInfo");
    if (!infoDiv) {
        const chartContainer = document.querySelector("#simulationSection .card-body");
        if (chartContainer) {
            infoDiv = document.createElement("div");
            infoDiv.id = "transactionInfo";
            infoDiv.className = "mt-3 p-3 border rounded";
            chartContainer.appendChild(infoDiv);
        }
    }
    
    if (infoDiv) {
        infoDiv.innerHTML = `
            <div class="alert alert-warning mb-0">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Simulation Stopped:</strong> ${errorMessage}
            </div>
        `;
    }
}

function updateTransactionInfo(data) {
    // Create or update transaction info display
    let infoDiv = document.getElementById("transactionInfo");
    if (!infoDiv) {
        const chartContainer = document.querySelector("#simulationSection .card-body");
        if (chartContainer) {
            infoDiv = document.createElement("div");
            infoDiv.id = "transactionInfo";
            infoDiv.className = "mt-3 p-3 border rounded bg-dark text-white";
            chartContainer.appendChild(infoDiv);
        }
    }
    
    if (infoDiv) {
        // Update the class in case it was created with old styling
        infoDiv.className = "mt-3 p-3 border rounded bg-dark text-white";
        
        let infoHtml = `
            <div class="row">
                <div class="col-md-6">
                    <strong>Latest Transaction:</strong><br>
                    Amount: $${data.amount}<br>
                    Time: ${data.timestamp}<br>
                    Fraud Score: ${data.fraud_score}
                </div>
                <div class="col-md-6">
                    <strong>Data Source:</strong> Database<br>
        `;
        
        if (data.transaction_id) {
            infoHtml += `Transaction ID: ${data.transaction_id}<br>`;
        }
        
        if (data.actual_class !== undefined) {
            const actualLabel = data.actual_class === 1 ? 'Fraud' : 'Normal';
            const labelClass = data.actual_class === 1 ? 'text-danger' : 'text-success';
            infoHtml += `Actual Class: <span class="${labelClass}"><strong>${actualLabel}</strong></span><br>`;
        }
        
        infoHtml += `
                </div>
            </div>
        `;
        
        infoDiv.innerHTML = infoHtml;
    }
}
