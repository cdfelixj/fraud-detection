import { Component } from 'react';
import PropTypes from 'prop-types';
import { Box, Typography, Button, Alert } from '@mui/material';
import { ErrorOutline as ErrorIcon } from '@mui/icons-material';

class ErrorBoundary extends Component {
  static getDerivedStateFromError() {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    this.setState({
      error,
      errorInfo,
    });

    // You can also log the error to an error reporting service here
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '100vh',
            padding: 3,
            textAlign: 'center',
          }}
        >
          <ErrorIcon color="error" sx={{ fontSize: 64, mb: 2 }} />
          
          <Typography variant="h4" component="h1" gutterBottom color="error">
            Something went wrong
          </Typography>
          
          <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
            We&apos;re sorry, but something unexpected happened. Please try reloading the page.
          </Typography>

          <Button
            variant="contained"
            color="primary"
            onClick={this.handleReload}
            sx={{ mb: 3 }}
          >
            Reload Page
          </Button>

          {process.env.NODE_ENV === 'development' && (
            <Alert severity="error" sx={{ mt: 2, textAlign: 'left', maxWidth: 800 }}>
              <Typography variant="h6" component="div" gutterBottom>
                Error Details (Development Only):
              </Typography>
              <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                {this.state.error && this.state.error.toString()}
                {this.state.errorInfo.componentStack}
              </Typography>
            </Alert>
          )}
        </Box>
      );
    }

    return this.props.children;
  }
}

ErrorBoundary.propTypes = {
  children: PropTypes.node.isRequired,
};

export default ErrorBoundary;
