import { useEffect, useState, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  Security as SecurityIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import axios from 'axios';

import LoadingSpinner from './LoadingSpinner.jsx';

// Configure dayjs
dayjs.extend(relativeTime);

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Custom hook for API calls
const useApiData = (endpoint, defaultValue, interval = 30000) => {
  const [data, setData] = useState(defaultValue);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const response = await axios.get(endpoint);
      setData(response.data);
    } catch (err) {
      console.error(`Error fetching ${endpoint}:`, err);
      setError(err.message);
      // Set mock data as fallback for demo
      if (endpoint === '/api/stats') {
        setData({
          totalTransactions: 1247,
          fraudTransactions: 23,
          avgFraudProbability: 0.12,
          highRiskTransactions: 45,
        });
      }
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
};

// System Health Component
const SystemHealth = () => {
  const { data: health, loading } = useApiData('/api/health', {
    api: false,
    database: false,
    cache: false,
  });

  if (loading) {
    return <LoadingSpinner message="Checking system health..." size={24} />;
  }

  const services = [
    { name: 'API', status: health.api, icon: '🌐' },
    { name: 'Database', status: health.database, icon: '💾' },
    { name: 'Cache', status: health.cache, icon: '⚡' },
  ];

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          System Status
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {services.map(({ name, status, icon }) => (
            <Chip
              key={name}
              icon={<span>{icon}</span>}
              label={`${name}: ${status ? 'Online' : 'Offline'}`}
              color={status ? 'success' : 'error'}
              variant={status ? 'filled' : 'outlined'}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

// Statistics Card Component
const StatCard = ({ title, value, subtitle, icon, color = 'primary' }) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {icon}
        <Typography variant="h6" sx={{ ml: 1 }}>
          {title}
        </Typography>
      </Box>
      <Typography variant="h3" color={color} gutterBottom>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        {subtitle}
      </Typography>
    </CardContent>
  </Card>
);

StatCard.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  subtitle: PropTypes.string.isRequired,
  icon: PropTypes.node.isRequired,
  color: PropTypes.string,
};

// Chart Component
const FraudChart = ({ transactions }) => {
  const chartData = useMemo(() => ({
    labels: transactions.map(t => dayjs(t.timestamp).format('HH:mm')),
    datasets: [
      {
        label: 'Fraud Probability',
        data: transactions.map(t => t.fraudProbability),
        borderColor: 'rgb(244, 67, 54)',
        backgroundColor: 'rgba(244, 67, 54, 0.1)',
        tension: 0.1,
        fill: true,
      },
    ],
  }), [transactions]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Fraud Probability Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          callback(value) {
            return (value * 100) + '%';
          },
        },
      },
    },
  }), []);

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recent Fraud Probability Trends
        </Typography>
        <Box sx={{ height: 400 }}>
          <Line data={chartData} options={chartOptions} />
        </Box>
      </CardContent>
    </Card>
  );
};

FraudChart.propTypes = {
  transactions: PropTypes.arrayOf(
    PropTypes.shape({
      timestamp: PropTypes.string.isRequired,
      fraudProbability: PropTypes.number.isRequired,
    })
  ).isRequired,
};

// Transactions Table Component
const TransactionsTable = ({ transactions }) => {
  const highRiskTransactions = useMemo(
    () => transactions.filter(t => t.isHighRisk),
    [transactions]
  );

  const getRiskLevel = (probability) => {
    if (probability > 0.7) return { level: 'HIGH', color: 'error' };
    if (probability > 0.4) return { level: 'MEDIUM', color: 'warning' };
    return { level: 'LOW', color: 'success' };
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Recent High-Risk Transactions
        </Typography>
        {highRiskTransactions.length === 0 ? (
          <Alert severity="info">No high-risk transactions found.</Alert>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Transaction ID</TableCell>
                  <TableCell align="right">Amount</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Risk Level</TableCell>
                  <TableCell align="right">Probability</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {highRiskTransactions.map((transaction) => {
                  const { level, color } = getRiskLevel(transaction.fraudProbability);
                  return (
                    <TableRow
                      key={transaction.id}
                      sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                      <TableCell component="th" scope="row">
                        #{transaction.id}
                      </TableCell>
                      <TableCell align="right">
                        ${transaction.amount.toFixed(2)}
                      </TableCell>
                      <TableCell>
                        {dayjs(transaction.timestamp).fromNow()}
                      </TableCell>
                      <TableCell>
                        <Chip label={level} color={color} size="small" />
                      </TableCell>
                      <TableCell align="right">
                        {(transaction.fraudProbability * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
};

TransactionsTable.propTypes = {
  transactions: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.number.isRequired,
      amount: PropTypes.number.isRequired,
      timestamp: PropTypes.string.isRequired,
      fraudProbability: PropTypes.number.isRequired,
      isHighRisk: PropTypes.bool.isRequired,
    })
  ).isRequired,
};

const Dashboard = () => {
  // Mock data for demo purposes
  const [recentTransactions] = useState([
    {
      id: 1,
      amount: 150.00,
      timestamp: dayjs().subtract(5, 'minutes').toISOString(),
      fraudProbability: 0.85,
      isHighRisk: true,
    },
    {
      id: 2,
      amount: 45.20,
      timestamp: dayjs().subtract(12, 'minutes').toISOString(),
      fraudProbability: 0.15,
      isHighRisk: false,
    },
    {
      id: 3,
      amount: 2300.00,
      timestamp: dayjs().subtract(18, 'minutes').toISOString(),
      fraudProbability: 0.72,
      isHighRisk: true,
    },
    {
      id: 4,
      amount: 89.99,
      timestamp: dayjs().subtract(25, 'minutes').toISOString(),
      fraudProbability: 0.23,
      isHighRisk: false,
    },
    {
      id: 5,
      amount: 1200.00,
      timestamp: dayjs().subtract(31, 'minutes').toISOString(),
      fraudProbability: 0.68,
      isHighRisk: true,
    },
  ]);

  const { data: stats, loading, error } = useApiData('/api/stats', {
    totalTransactions: 0,
    fraudTransactions: 0,
    avgFraudProbability: 0,
    highRiskTransactions: 0,
  });

  if (loading) {
    return <LoadingSpinner message="Loading dashboard..." />;
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Error loading dashboard: {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Real-Time Fraud Detection Dashboard
      </Typography>

      <SystemHealth />

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Transactions"
            value={stats.totalTransactions}
            subtitle="Last 24 Hours"
            icon={<AssessmentIcon color="primary" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Fraud Detected"
            value={stats.fraudTransactions}
            subtitle="Confirmed Fraud"
            icon={<SecurityIcon color="error" />}
            color="error"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Average Risk"
            value={`${(stats.avgFraudProbability * 100).toFixed(1)}%`}
            subtitle="Fraud Probability"
            icon={<WarningIcon color="warning" />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="High Risk"
            value={stats.highRiskTransactions}
            subtitle="Flagged Transactions"
            icon={<TrendingUpIcon color="warning" />}
            color="warning"
          />
        </Grid>
      </Grid>

      <FraudChart transactions={recentTransactions} />

      <TransactionsTable transactions={recentTransactions} />
    </Box>
  );
};

export default Dashboard;
