import PropTypes from 'prop-types';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { Security as SecurityIcon } from '@mui/icons-material';

const Layout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/alerts', label: 'Alerts' },
  ];

  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <SecurityIcon sx={{ mr: 2 }} />
          <Typography
            variant="h6"
            component="h1"
            sx={{ flexGrow: 1, fontWeight: 600 }}
          >
            Fraud Detection System
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            {navigationItems.map(({ path, label }) => (
              <Button
                key={path}
                color="inherit"
                onClick={() => handleNavigation(path)}
                sx={{
                  backgroundColor: location.pathname === path 
                    ? 'rgba(255, 255, 255, 0.2)' 
                    : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  },
                }}
              >
                {label}
              </Button>
            ))}
          </Box>
        </Toolbar>
      </AppBar>
      
      <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
        {children}
      </Container>
    </Box>
  );
};

Layout.propTypes = {
  children: PropTypes.node.isRequired,
};

export default Layout;
