import PropTypes from 'prop-types';
import { Box, CircularProgress, Typography } from '@mui/material';

const LoadingSpinner = ({ message = 'Loading...', size = 40 }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '200px',
        padding: 3,
      }}
    >
      <CircularProgress size={size} sx={{ mb: 2 }} />
      <Typography variant="body1" color="textSecondary">
        {message}
      </Typography>
    </Box>
  );
};

LoadingSpinner.propTypes = {
  message: PropTypes.string,
  size: PropTypes.number,
};

export default LoadingSpinner;
