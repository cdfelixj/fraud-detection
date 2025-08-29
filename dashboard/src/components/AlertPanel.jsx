import { useState, useEffect, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Alert,
  Chip,
  Grid,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Badge,
  Fab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Close as CloseIcon,
  Refresh as RefreshIcon,
  NotificationImportant as NotificationIcon,
  Security as SecurityIcon,
  LocationOn as LocationIcon,
  AccountBalance as AccountIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';

import LoadingSpinner from './LoadingSpinner.jsx';

// Configure dayjs
dayjs.extend(relativeTime);

// Alert severity configurations
const SEVERITY_CONFIG = {
  high: {
    icon: <ErrorIcon />,
    color: 'error',
    label: 'High Priority',
  },
  medium: {
    icon: <WarningIcon />,
    color: 'warning',
    label: 'Medium Priority',
  },
  low: {
    icon: <InfoIcon />,
    color: 'info',
    label: 'Low Priority',
  },
};

// Mock alert data
const mockAlerts = [
  {
    id: 1,
    type: 'high',
    title: 'Suspicious Large Transaction',
    message: 'Transaction of $15,000 from unusual location detected.',
    timestamp: dayjs().subtract(2, 'minutes').toISOString(),
    details: {
      transactionId: 'TXN_001234',
      amount: 15000,
      location: 'Lagos, Nigeria',
      cardType: 'Visa Premium',
      merchantCategory: 'Online Shopping',
    },
    acknowledged: false,
  },
  {
    id: 2,
    type: 'medium',
    title: 'Multiple Failed Login Attempts',
    message: 'Account shows 5 failed login attempts in 10 minutes.',
    timestamp: dayjs().subtract(8, 'minutes').toISOString(),
    details: {
      accountId: 'ACC_567890',
      attempts: 5,
      ipAddress: '192.168.1.100',
      location: 'Unknown Location',
    },
    acknowledged: false,
  },
  {
    id: 3,
    type: 'high',
    title: 'Velocity Check Failed',
    message: 'Card used for 3 transactions in different countries within 1 hour.',
    timestamp: dayjs().subtract(15, 'minutes').toISOString(),
    details: {
      cardNumber: '**** **** **** 1234',
      countries: ['USA', 'UK', 'Germany'],
      totalAmount: 8500,
    },
    acknowledged: true,
  },
  {
    id: 4,
    type: 'low',
    title: 'Unusual Merchant Category',
    message: 'Customer made first-time purchase in gambling category.',
    timestamp: dayjs().subtract(25, 'minutes').toISOString(),
    details: {
      customerId: 'CUST_789012',
      merchantCategory: 'Gambling',
      amount: 250,
    },
    acknowledged: false,
  },
];

// Alert Item Component
const AlertItem = ({ alert, onAcknowledge, onViewDetails }) => {
  const config = SEVERITY_CONFIG[alert.type];

  return (
    <Card 
      sx={{ 
        mb: 2, 
        opacity: alert.acknowledged ? 0.6 : 1,
        border: alert.acknowledged ? '1px solid #e0e0e0' : `2px solid`,
        borderColor: alert.acknowledged ? '#e0e0e0' : `${config.color}.main`,
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <IconButton color={config.color} sx={{ mr: 1 }}>
              {config.icon}
            </IconButton>
            <Box>
              <Typography variant="h6" component="h3">
                {alert.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {dayjs(alert.timestamp).fromNow()}
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip 
              label={config.label}
              color={config.color}
              size="small"
              variant={alert.acknowledged ? 'outlined' : 'filled'}
            />
            {alert.acknowledged && (
              <Chip label="Acknowledged" color="success" size="small" variant="outlined" />
            )}
          </Box>
        </Box>

        <Typography variant="body1" sx={{ mb: 2 }}>
          {alert.message}
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            size="small"
            onClick={() => onViewDetails(alert)}
          >
            View Details
          </Button>
          {!alert.acknowledged && (
            <Button
              variant="contained"
              size="small"
              color={config.color}
              onClick={() => onAcknowledge(alert.id)}
            >
              Acknowledge
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

AlertItem.propTypes = {
  alert: PropTypes.shape({
    id: PropTypes.number.isRequired,
    type: PropTypes.oneOf(['high', 'medium', 'low']).isRequired,
    title: PropTypes.string.isRequired,
    message: PropTypes.string.isRequired,
    timestamp: PropTypes.string.isRequired,
    acknowledged: PropTypes.bool.isRequired,
    details: PropTypes.object.isRequired,
  }).isRequired,
  onAcknowledge: PropTypes.func.isRequired,
  onViewDetails: PropTypes.func.isRequired,
};

// Alert Details Dialog Component
const AlertDetailsDialog = ({ alert, open, onClose }) => {
  if (!alert) return null;

  const config = SEVERITY_CONFIG[alert.type];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
        <IconButton color={config.color} sx={{ mr: 1 }}>
          {config.icon}
        </IconButton>
        {alert.title}
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{ position: 'absolute', right: 8, top: 8 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers>
        <Typography variant="body1" paragraph>
          {alert.message}
        </Typography>
        
        <Typography variant="h6" gutterBottom>
          Alert Details
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <SecurityIcon />
            </ListItemIcon>
            <ListItemText 
              primary="Alert ID" 
              secondary={alert.id} 
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <InfoIcon />
            </ListItemIcon>
            <ListItemText 
              primary="Severity" 
              secondary={config.label} 
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <NotificationIcon />
            </ListItemIcon>
            <ListItemText 
              primary="Time" 
              secondary={dayjs(alert.timestamp).format('YYYY-MM-DD HH:mm:ss')} 
            />
          </ListItem>
          
          <Divider sx={{ my: 2 }} />
          
          {Object.entries(alert.details).map(([key, value]) => (
            <ListItem key={key}>
              <ListItemIcon>
                {key.includes('location') ? <LocationIcon /> : 
                 key.includes('account') || key.includes('card') ? <AccountIcon /> : 
                 <InfoIcon />}
              </ListItemIcon>
              <ListItemText 
                primary={key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                secondary={Array.isArray(value) ? value.join(', ') : value}
              />
            </ListItem>
          ))}
        </List>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

AlertDetailsDialog.propTypes = {
  alert: PropTypes.object,
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired,
};

// Main AlertPanel Component
const AlertPanel = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [showAcknowledged, setShowAcknowledged] = useState(true);

  // Load alerts (mock data for demo)
  useEffect(() => {
    const loadAlerts = async () => {
      try {
        setLoading(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        setAlerts(mockAlerts);
      } catch (error) {
        console.error('Error loading alerts:', error);
      } finally {
        setLoading(false);
      }
    };

    loadAlerts();
  }, []);

  // Filter alerts
  const filteredAlerts = useMemo(() => {
    return alerts.filter(alert => {
      const severityMatch = filterSeverity === 'all' || alert.type === filterSeverity;
      const acknowledgedMatch = showAcknowledged || !alert.acknowledged;
      return severityMatch && acknowledgedMatch;
    });
  }, [alerts, filterSeverity, showAcknowledged]);

  // Get alert counts
  const alertCounts = useMemo(() => {
    return alerts.reduce((acc, alert) => {
      if (!alert.acknowledged) {
        acc[alert.type] = (acc[alert.type] || 0) + 1;
        acc.total += 1;
      }
      return acc;
    }, { high: 0, medium: 0, low: 0, total: 0 });
  }, [alerts]);

  const handleAcknowledge = useCallback((alertId) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, acknowledged: true }
          : alert
      )
    );
  }, []);

  const handleViewDetails = useCallback((alert) => {
    setSelectedAlert(alert);
    setDialogOpen(true);
  }, []);

  const handleRefresh = useCallback(() => {
    setLoading(true);
    // Simulate refresh
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading alerts..." />;
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Security Alerts
        </Typography>
        
        <Fab 
          color="primary" 
          aria-label="refresh" 
          onClick={handleRefresh}
          size="medium"
        >
          <Badge badgeContent={alertCounts.total} color="error">
            <RefreshIcon />
          </Badge>
        </Fab>
      </Box>

      {/* Alert Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={3}>
          <Card sx={{ textAlign: 'center', bgcolor: 'error.light', color: 'white' }}>
            <CardContent>
              <Typography variant="h3">{alertCounts.high}</Typography>
              <Typography variant="body1">High Priority</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card sx={{ textAlign: 'center', bgcolor: 'warning.light', color: 'white' }}>
            <CardContent>
              <Typography variant="h3">{alertCounts.medium}</Typography>
              <Typography variant="body1">Medium Priority</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card sx={{ textAlign: 'center', bgcolor: 'info.light', color: 'white' }}>
            <CardContent>
              <Typography variant="h3">{alertCounts.low}</Typography>
              <Typography variant="body1">Low Priority</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={3}>
          <Card sx={{ textAlign: 'center', bgcolor: 'success.light', color: 'white' }}>
            <CardContent>
              <Typography variant="h3">{alertCounts.total}</Typography>
              <Typography variant="body1">Total Unresolved</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <FilterIcon />
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Severity</InputLabel>
              <Select
                value={filterSeverity}
                label="Severity"
                onChange={(e) => setFilterSeverity(e.target.value)}
              >
                <MenuItem value="all">All Severities</MenuItem>
                <MenuItem value="high">High Priority</MenuItem>
                <MenuItem value="medium">Medium Priority</MenuItem>
                <MenuItem value="low">Low Priority</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={showAcknowledged ? 'all' : 'unacknowledged'}
                label="Status"
                onChange={(e) => setShowAcknowledged(e.target.value === 'all')}
              >
                <MenuItem value="all">All Alerts</MenuItem>
                <MenuItem value="unacknowledged">Unacknowledged Only</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {/* Alerts List */}
      <Box>
        {filteredAlerts.length === 0 ? (
          <Alert severity="info" sx={{ textAlign: 'center' }}>
            No alerts match your current filters.
          </Alert>
        ) : (
          filteredAlerts.map(alert => (
            <AlertItem
              key={alert.id}
              alert={alert}
              onAcknowledge={handleAcknowledge}
              onViewDetails={handleViewDetails}
            />
          ))
        )}
      </Box>

      {/* Alert Details Dialog */}
      <AlertDetailsDialog
        alert={selectedAlert}
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
      />
    </Box>
  );
};

export default AlertPanel;
