import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';
import moment from 'moment';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const Dashboard = () => {
    const [stats, setStats] = useState({
        totalTransactions: 0,
        fraudTransactions: 0,
        avgFraudProbability: 0,
        highRiskTransactions: 0
    });
    const [recentTransactions, setRecentTransactions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [systemHealth, setSystemHealth] = useState({
        api: false,
        database: false,
        cache: false
    });

    useEffect(() => {
        const fetchDashboardData = async () => {
            try {
                setLoading(true);
                await Promise.all([
                    fetchStats(),
                    fetchRecentTransactions(),
                    checkSystemHealth()
                ]);
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
                setError('Failed to load dashboard data');
            } finally {
                setLoading(false);
            }
        };

        const fetchStats = async () => {
            try {
                // Mock data since we don't have real data yet
                const mockStats = {
                    totalTransactions: 1247,
                    fraudTransactions: 23,
                    avgFraudProbability: 0.12,
                    highRiskTransactions: 45
                };
                setStats(mockStats);
            } catch (error) {
                console.error('Error fetching stats:', error);
                // Set mock data as fallback
                setStats({
                    totalTransactions: 0,
                    fraudTransactions: 0,
                    avgFraudProbability: 0,
                    highRiskTransactions: 0
                });
            }
        };

        const fetchRecentTransactions = async () => {
            try {
                // Mock recent transactions
                const mockTransactions = [
                    { id: 1, amount: 150.00, timestamp: moment().subtract(5, 'minutes').toISOString(), fraudProbability: 0.85, isHighRisk: true },
                    { id: 2, amount: 45.20, timestamp: moment().subtract(12, 'minutes').toISOString(), fraudProbability: 0.15, isHighRisk: false },
                    { id: 3, amount: 2300.00, timestamp: moment().subtract(18, 'minutes').toISOString(), fraudProbability: 0.72, isHighRisk: true },
                    { id: 4, amount: 89.99, timestamp: moment().subtract(25, 'minutes').toISOString(), fraudProbability: 0.23, isHighRisk: false },
                    { id: 5, amount: 1200.00, timestamp: moment().subtract(31, 'minutes').toISOString(), fraudProbability: 0.68, isHighRisk: true }
                ];
                setRecentTransactions(mockTransactions);
            } catch (error) {
                console.error('Error fetching recent transactions:', error);
                setRecentTransactions([]);
            }
        };

        const checkSystemHealth = async () => {
            try {
                const response = await axios.get('/api/health');
                setSystemHealth({
                    api: true,
                    database: true,
                    cache: true
                });
            } catch (error) {
                console.error('System health check failed:', error);
                setSystemHealth({
                    api: false,
                    database: false,
                    cache: false
                });
            }
        };

        fetchDashboardData();

        // Set up real-time updates
        const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
        return () => clearInterval(interval);
    }, []);

    const chartData = {
        labels: recentTransactions.map(t => moment(t.timestamp).format('HH:mm')),
        datasets: [
            {
                label: 'Fraud Probability',
                data: recentTransactions.map(t => t.fraudProbability),
                borderColor: 'rgb(244, 67, 54)',
                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                tension: 0.1
            }
        ]
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top'
            },
            title: {
                display: true,
                text: 'Fraud Probability Over Time'
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 1
            }
        }
    };

    if (loading) {
        return <div className="loading">Loading dashboard...</div>;
    }

    if (error) {
        return <div className="error">{error}</div>;
    }

    return (
        <div>
            <h2>Real-Time Fraud Detection Dashboard</h2>
            
            {/* System Health Status */}
            <div className="card">
                <h3>System Status</h3>
                <div style={{ display: 'flex', gap: '1rem' }}>
                    <span style={{ color: systemHealth.api ? '#4caf50' : '#f44336' }}>
                        API: {systemHealth.api ? '✓ Online' : '✗ Offline'}
                    </span>
                    <span style={{ color: systemHealth.database ? '#4caf50' : '#f44336' }}>
                        Database: {systemHealth.database ? '✓ Connected' : '✗ Disconnected'}
                    </span>
                    <span style={{ color: systemHealth.cache ? '#4caf50' : '#f44336' }}>
                        Cache: {systemHealth.cache ? '✓ Active' : '✗ Inactive'}
                    </span>
                </div>
            </div>

            {/* Statistics Cards */}
            <div className="dashboard-container">
                <div className="card stat-card">
                    <h3>Total Transactions</h3>
                    <div className="stat-value">{stats.totalTransactions.toLocaleString()}</div>
                    <div className="stat-label">Last 24 Hours</div>
                </div>

                <div className="card stat-card">
                    <h3>Fraud Detected</h3>
                    <div className="stat-value fraud">{stats.fraudTransactions}</div>
                    <div className="stat-label">Confirmed Fraud</div>
                </div>

                <div className="card stat-card">
                    <h3>Average Risk</h3>
                    <div className="stat-value warning">{(stats.avgFraudProbability * 100).toFixed(1)}%</div>
                    <div className="stat-label">Fraud Probability</div>
                </div>

                <div className="card stat-card">
                    <h3>High Risk</h3>
                    <div className="stat-value warning">{stats.highRiskTransactions}</div>
                    <div className="stat-label">Flagged Transactions</div>
                </div>
            </div>

            {/* Fraud Probability Chart */}
            <div className="card">
                <h3>Recent Fraud Probability Trends</h3>
                <Line data={chartData} options={chartOptions} />
            </div>

            {/* Recent High-Risk Transactions */}
            <div className="card">
                <h3>Recent High-Risk Transactions</h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '2px solid #ddd' }}>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Transaction ID</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Amount</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Time</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Risk Level</th>
                                <th style={{ padding: '1rem', textAlign: 'left' }}>Probability</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentTransactions.filter(t => t.isHighRisk).map(transaction => (
                                <tr key={transaction.id} style={{ borderBottom: '1px solid #eee' }}>
                                    <td style={{ padding: '1rem' }}>#{transaction.id}</td>
                                    <td style={{ padding: '1rem' }}>${transaction.amount.toFixed(2)}</td>
                                    <td style={{ padding: '1rem' }}>{moment(transaction.timestamp).fromNow()}</td>
                                    <td style={{ padding: '1rem' }}>
                                        <span style={{ 
                                            color: transaction.fraudProbability > 0.7 ? '#f44336' : '#ff9800',
                                            fontWeight: 'bold'
                                        }}>
                                            {transaction.fraudProbability > 0.7 ? 'HIGH' : 'MEDIUM'}
                                        </span>
                                    </td>
                                    <td style={{ padding: '1rem' }}>
                                        {(transaction.fraudProbability * 100).toFixed(1)}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;