/**
 * Dashboard Component
 * Main dashboard displaying overview of the Hybrid AI Content Popularity Prediction System
 * Shows system status, analytics, and navigation to other components
 */

import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
  Button,
  Alert,
  Fab,
  Tooltip,
  IconButton,
  Badge
} from '@mui/material';
import {
  TrendingUp,
  SentimentSatisfied,
  Security,
  Assessment,
  CloudUpload,
  Refresh,
  CheckCircle,
  Error,
  Warning,
  Info,
  BarChart,
  PieChart,
  Timeline,
  FilterList,
  Settings,
  Notifications
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement,
} from 'Chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const Dashboard = () => {
  const navigate = useNavigate();
  const [systemStatus, setSystemStatus] = useState({
    sentimentModel: 'Not Trained',
    engagementModel: 'Not Trained',
    popularityModel: 'Not Trained',
    apiStatus: 'Connected',
    lastUpdated: null
  });
  const [analytics, setAnalytics] = useState({
    totalAnalyses: 0,
    avgAccuracy: 0,
    fakeEngagementDetected: 0,
    popularityPredictions: 0
  });
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(false);
  const [notifications, setNotifications] = useState([]);

  // Sample data for charts (replace with actual API calls)
  const sentimentData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        label: 'Sentiment Distribution',
        data: [65, 20, 15],
        backgroundColor: [
          'rgba(75, 192, 192, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(255, 99, 132, 0.8)',
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(255, 99, 132, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  const engagementData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Views',
        data: [1200, 1900, 1500, 2100, 2400, 1800, 2200],
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
      },
      {
        label: 'Likes',
        data: [120, 190, 150, 210, 240, 180, 220],
        backgroundColor: 'rgba(255, 99, 132, 0.8)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
      },
      {
        label: 'Comments',
        data: [30, 45, 35, 50, 60, 40, 55],
        backgroundColor: 'rgba(75, 192, 192, 0.8)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
      },
    ],
  };

  const popularityData = {
    labels: ['Content 1', 'Content 2', 'Content 3', 'Content 4', 'Content 5'],
    datasets: [
      {
        label: 'Current Views',
        data: [1200, 1900, 1500, 2100, 2400],
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
      },
      {
        label: 'Predicted Future Views',
        data: [1800, 2800, 2200, 3200, 3600],
        backgroundColor: 'rgba(255, 206, 86, 0.8)',
        borderColor: 'rgba(255, 206, 86, 1)',
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Engagement Analytics',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  useEffect(() => {
    fetchSystemStatus();
    fetchAnalytics();
    fetchRecentActivity();
    
    // Set up real-time updates
    const interval = setInterval(() => {
      fetchSystemStatus();
      fetchAnalytics();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/models/status');
      const status = await response.json();
      
      setSystemStatus(prev => ({
        ...prev,
        sentimentModel: status.model_status.sentiment_model.trained ? 'Trained' : 'Not Trained',
        engagementModel: status.model_status.engagement_detector.trained ? 'Trained' : 'Not Trained',
        popularityModel: status.model_status.popularity_predictor.trained ? 'Trained' : 'Not Trained',
        apiStatus: 'Connected',
        lastUpdated: new Date().toLocaleTimeString()
      }));
    } catch (error) {
      setSystemStatus(prev => ({ ...prev, apiStatus: 'Disconnected' }));
    }
  };

  const fetchAnalytics = async () => {
    // Mock analytics data - replace with actual API calls
    setAnalytics({
      totalAnalyses: 1250,
      avgAccuracy: 87.5,
      fakeEngagementDetected: 45,
      popularityPredictions: 320
    });
  };

  const fetchRecentActivity = async () => {
    // Mock recent activity - replace with actual API calls
    setRecentActivity([
      { 
        id: 1, 
        type: 'sentiment', 
        action: 'Analyzed sentiment for 50 comments', 
        timestamp: '2 minutes ago',
        status: 'success',
        accuracy: '92%'
      },
      { 
        id: 2, 
        type: 'engagement', 
        action: 'Detected fake engagement in 3 posts', 
        timestamp: '15 minutes ago',
        status: 'warning',
        authenticity: 'Low'
      },
      { 
        id: 3, 
        type: 'popularity', 
        action: 'Predicted popularity for 25 content items', 
        timestamp: '1 hour ago',
        status: 'success',
        avgPrediction: '2,450 views'
      },
      { 
        id: 4, 
        type: 'upload', 
        action: 'Uploaded engagement dataset (1,000 rows)', 
        timestamp: '2 hours ago',
        status: 'success'
      },
      { 
        id: 5, 
        type: 'training', 
        action: 'Trained sentiment model with new data', 
        timestamp: '3 hours ago',
        status: 'success',
        accuracy: '94%'
      }
    ]);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Trained':
      case 'Connected':
      case 'success':
        return 'success';
      case 'Not Trained':
      case 'warning':
        return 'warning';
      case 'Disconnected':
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (type, status) => {
    const icons = {
      sentiment: <SentimentSatisfied />,
      engagement: <Security />,
      popularity: <TrendingUp />,
      upload: <CloudUpload />,
      training: <Settings />
    };
    
    return icons[type] || <Info />;
  };

  const handleRefresh = () => {
    setLoading(true);
    fetchSystemStatus();
    fetchAnalytics();
    fetchRecentActivity();
    setTimeout(() => setLoading(false), 1000);
  };

  const handleQuickAction = (action) => {
    switch (action) {
      case 'upload':
        navigate('/upload');
        break;
      case 'sentiment':
        navigate('/sentiment');
        break;
      case 'popularity':
        navigate('/prediction');
        break;
      default:
        break;
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Dashboard
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tooltip title="Notifications">
            <IconButton color="inherit">
              <Badge badgeContent={notifications.length} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {systemStatus.apiStatus === 'Disconnected' && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Backend API is disconnected. Some features may not be available.
        </Alert>
      )}

      {/* System Status Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SentimentSatisfied color="primary" sx={{ mr: 2 }} />
                <Typography variant="h6">Sentiment Model</Typography>
              </Box>
              <Chip 
                label={systemStatus.sentimentModel} 
                color={getStatusColor(systemStatus.sentimentModel)}
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Last updated: {systemStatus.lastUpdated}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Security color="secondary" sx={{ mr: 2 }} />
                <Typography variant="h6">Engagement Model</Typography>
              </Box>
              <Chip 
                label={systemStatus.engagementModel} 
                color={getStatusColor(systemStatus.engagementModel)}
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Fake detection: {analytics.fakeEngagementDetected} cases
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUp color="success" sx={{ mr: 2 }} />
                <Typography variant="h6">Prediction Model</Typography>
              </Box>
              <Chip 
                label={systemStatus.popularityModel} 
                color={getStatusColor(systemStatus.popularityModel)}
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Predictions: {analytics.popularityPredictions}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Assessment color="info" sx={{ mr: 2 }} />
                <Typography variant="h6">System Health</Typography>
              </Box>
              <Chip 
                label={systemStatus.apiStatus} 
                color={getStatusColor(systemStatus.apiStatus)}
                size="small"
                sx={{ mb: 1 }}
              />
              <Typography variant="body2" color="text.secondary">
                Avg Accuracy: {analytics.avgAccuracy}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Analytics Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Sentiment Distribution
            </Typography>
            <Box sx={{ height: 320, display: 'flex', justifyContent: 'center' }}>
              <Doughnut data={sentimentData} />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Weekly Engagement Trends
            </Typography>
            <Box sx={{ height: 320 }}>
              <Bar data={engagementData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Popularity Predictions
            </Typography>
            <Box sx={{ height: 320 }}>
              <Bar data={popularityData} options={chartOptions} />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Recent Activity and Quick Actions */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <List sx={{ maxHeight: 400, overflow: 'auto' }}>
              {recentActivity.map((activity) => (
                <ListItem key={activity.id} divider>
                  <ListItemIcon>
                    {getStatusIcon(activity.type, activity.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={activity.action}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {activity.timestamp}
                        </Typography>
                        {activity.accuracy && (
                          <Chip 
                            label={`Accuracy: ${activity.accuracy}`} 
                            color="success" 
                            size="small" 
                            sx={{ mr: 1 }}
                          />
                        )}
                        {activity.authenticity && (
                          <Chip 
                            label={`Authenticity: ${activity.authenticity}`} 
                            color="warning" 
                            size="small" 
                            sx={{ mr: 1 }}
                          />
                        )}
                        {activity.avgPrediction && (
                          <Chip 
                            label={`Avg: ${activity.avgPrediction}`} 
                            color="info" 
                            size="small" 
                          />
                        )}
                      </Box>
                    }
                  />
                  <Chip
                    label={activity.status}
                    color={getStatusColor(activity.status)}
                    size="small"
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button
                variant="contained"
                startIcon={<CloudUpload />}
                onClick={() => handleQuickAction('upload')}
                fullWidth
              >
                Upload Data
              </Button>
              <Button
                variant="outlined"
                startIcon={<SentimentSatisfied />}
                onClick={() => handleQuickAction('sentiment')}
                fullWidth
              >
                Sentiment Analysis
              </Button>
              <Button
                variant="outlined"
                startIcon={<TrendingUp />}
                onClick={() => handleQuickAction('popularity')}
                fullWidth
              >
                Predict Popularity
              </Button>
              <Button
                variant="outlined"
                startIcon={<Security />}
                fullWidth
              >
                Check Engagement
              </Button>
            </Box>
            
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                System Statistics
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Total Analyses"
                    secondary={analytics.totalAnalyses.toLocaleString()}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Fake Engagement Detected"
                    secondary={analytics.fakeEngagementDetected}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Popularity Predictions"
                    secondary={analytics.popularityPredictions}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Average Accuracy"
                    secondary={`${analytics.avgAccuracy}%`}
                  />
                </ListItem>
              </List>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="refresh"
        onClick={handleRefresh}
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        disabled={loading}
      >
        <Refresh />
      </Fab>

      {loading && (
        <LinearProgress sx={{ position: 'fixed', top: 0, left: 0, right: 0 }} />
      )}
    </Box>
  );
};

export default Dashboard;
