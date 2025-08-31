// Dashboard JavaScript functionality

// Global variables
let charts = {};
let refreshInterval;

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
    if (charts.amountDistribution && typeof charts.amountDistribution.destroy === 'function') {
        charts.amountDistribution.destroy();
        charts.amountDistribution = null;
    }
    
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
    if (charts.classDistribution && typeof charts.classDistribution.destroy === 'function') {
        charts.classDistribution.destroy();
        charts.classDistribution = null;
    }
    
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
    if (charts.performance && typeof charts.performance.destroy === 'function') {
        charts.performance.destroy();
        charts.performance = null;
    }
    
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
    if (charts.timeSeries && typeof charts.timeSeries.destroy === 'function') {
        charts.timeSeries.destroy();
        charts.timeSeries = null;
    }
    
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
    
    const min = Math.min(...amounts);
    const max = Math.max(...amounts);
    const binCount = 10;
    const binSize = (max - min) / binCount;
    
    const bins = [];
    const labels = [];
    
    for (let i = 0; i < binCount; i++) {
        const binStart = min + (i * binSize);
        const binEnd = min + ((i + 1) * binSize);
        
        const count = amounts.filter(amount => 
            amount >= binStart && (i === binCount - 1 ? amount <= binEnd : amount < binEnd)
        ).length;
        
        bins.push(count);
        labels.push(`$${binStart.toFixed(0)}-$${binEnd.toFixed(0)}`);
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
