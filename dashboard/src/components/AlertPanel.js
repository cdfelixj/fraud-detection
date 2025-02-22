import React, { useEffect, useState } from 'react';
import axios from 'axios';
import moment from 'moment';

const AlertPanel = ({ alerts: propAlerts }) => {
    const [alerts, setAlerts] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Use prop alerts if provided, otherwise fetch alerts
        if (propAlerts) {
            setAlerts(propAlerts);
        } else {
            fetchAlerts();
        }

        // Set up real-time updates
        const interval = setInterval(fetchAlerts, 10000); // Update every 10 seconds
        return () => clearInterval(interval);
    }, [propAlerts]);

    const fetchAlerts = async () => {
        try {
            setLoading(true);
            setError(null);

            // Mock alerts for demonstration
            const mockAlerts = [
                {
                    id: 1,
                    type: 'high',
                    title: 'High-Risk Transaction Detected',
                    message: 'Transaction #12849 flagged with 89% fraud probability',
                    timestamp: moment().subtract(2, 'minutes').toISOString(),
                    details: {
                        transactionId: '12849',
                        amount: '$2,350.00',
                        location: 'Unknown',
                        probability: 0.89
                    }
                },
                {
                    id: 2,
                    type: 'medium',
                    title: 'Unusual Pattern Detected',
                    message: 'Multiple transactions from same IP address',
                    timestamp: moment().subtract(15, 'minutes').toISOString(),
                    details: {
                        ipAddress: '192.168.1.100',
                        transactionCount: 5,
                        timeSpan: '10 minutes'
                    }
                },
                {
                    id: 3,
                    type: 'high',
                    title: 'Card Not Present Transaction',
                    message: 'High-value CNP transaction flagged',
                    timestamp: moment().subtract(23, 'minutes').toISOString(),
                    details: {
                        transactionId: '12847',
                        amount: '$4,200.00',
                        merchant: 'Online Electronics',
                        probability: 0.76
                    }
                },
                {
                    id: 4,
                    type: 'low',
                    title: 'Velocity Check Alert',
                    message: 'Customer exceeded daily transaction limit',
                    timestamp: moment().subtract(45, 'minutes').toISOString(),
                    details: {
                        customerId: 'CUST_5547',
                        dailyLimit: '$1,000',
                        currentTotal: '$1,150'
                    }
                }
            ];

            setAlerts(mockAlerts);
        } catch (error) {
            console.error('Error fetching alerts:', error);
            setError('Failed to load alerts');
        } finally {
            setLoading(false);
        }
    };

    const getPriorityIcon = (type) => {
        switch (type) {
            case 'high':
                return '🔴';
            case 'medium':
                return '🟡';
            case 'low':
                return '🔵';
            default:
                return '⚪';
        }
    };

    const getPriorityText = (type) => {
        switch (type) {
            case 'high':
                return 'HIGH PRIORITY';
            case 'medium':
                return 'MEDIUM PRIORITY';
            case 'low':
                return 'LOW PRIORITY';
            default:
                return 'UNKNOWN';
        }
    };

    if (loading && alerts.length === 0) {
        return <div className="loading">Loading alerts...</div>;
    }

    if (error && alerts.length === 0) {
        return <div className="error">Error loading alerts: {error}</div>;
    }

    return (
        <div className="alert-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <h2>Fraud Alerts</h2>
                <div style={{ fontSize: '0.9rem', color: '#666' }}>
                    Last updated: {moment().format('HH:mm:ss')}
                </div>
            </div>

            {alerts.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', padding: '2rem', color: '#666' }}>
                    <p>✅ No active fraud alerts</p>
                    <p style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                        All transactions are being monitored continuously
                    </p>
                </div>
            ) : (
                <div>
                    {alerts.map((alert) => (
                        <div key={alert.id} className={`alert-item ${alert.type}`}>
                            <div className="alert-header">
                                <div className="alert-title">
                                    {getPriorityIcon(alert.type)} {alert.title}
                                </div>
                                <div className="alert-time">
                                    {moment(alert.timestamp).fromNow()}
                                </div>
                            </div>
                            
                            <div className="alert-details">
                                <p>{alert.message}</p>
                                
                                {alert.details && (
                                    <div style={{ 
                                        marginTop: '0.5rem', 
                                        fontSize: '0.8rem', 
                                        background: 'rgba(255,255,255,0.3)',
                                        padding: '0.5rem',
                                        borderRadius: '4px'
                                    }}>
                                        {Object.entries(alert.details).map(([key, value]) => (
                                            <div key={key} style={{ margin: '0.2rem 0' }}>
                                                <strong>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:</strong> {value}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div style={{ 
                                marginTop: '1rem', 
                                display: 'flex', 
                                justifyContent: 'space-between', 
                                alignItems: 'center' 
                            }}>
                                <span style={{ 
                                    fontSize: '0.7rem', 
                                    fontWeight: 'bold',
                                    color: alert.type === 'high' ? '#d32f2f' : alert.type === 'medium' ? '#f57c00' : '#1976d2'
                                }}>
                                    {getPriorityText(alert.type)}
                                </span>
                                
                                <div style={{ display: 'flex', gap: '0.5rem' }}>
                                    <button 
                                        style={{ 
                                            padding: '0.3rem 0.8rem', 
                                            fontSize: '0.8rem',
                                            border: 'none',
                                            borderRadius: '4px',
                                            background: '#4caf50',
                                            color: 'white',
                                            cursor: 'pointer'
                                        }}
                                        onClick={() => {
                                            // Handle acknowledge action
                                            console.log(`Acknowledged alert ${alert.id}`);
                                        }}
                                    >
                                        Acknowledge
                                    </button>
                                    <button 
                                        style={{ 
                                            padding: '0.3rem 0.8rem', 
                                            fontSize: '0.8rem',
                                            border: '1px solid #ccc',
                                            borderRadius: '4px',
                                            background: 'white',
                                            color: '#333',
                                            cursor: 'pointer'
                                        }}
                                        onClick={() => {
                                            // Handle investigate action
                                            console.log(`Investigating alert ${alert.id}`);
                                        }}
                                    >
                                        Investigate
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default AlertPanel;