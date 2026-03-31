/**
 * Popularity Prediction Component
 * Provides interface for content popularity prediction with Chart.js visualizations
 * Integrates sentiment analysis and engagement metrics for comprehensive predictions
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Badge,
  IconButton,
  TextField,
  Slider
} from '@mui/material';
import {
  TrendingUp,
  Upload,
  Refresh,
  CheckCircle,
  Error,
  ExpandMore,
  Assessment,
  BarChart,
  Timeline,
  PieChart,
  FilterList,
  Download,
  Share,
  Settings,
  Analytics,
  Speed,
  Visibility
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'Chart.js';
import { Bar, Line, Doughnut, Scatter } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

const PopularityPrediction = () => {
  const [tabValue, setTabValue] = useState(0);
  const [modelType, setModelType] = useState('random_forest');
  const [modelStatus, setModelStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [predictionResults, setPredictionResults] = useState([]);
  const [uploadedFiles, setUploadedFiles] = useState({
    content: null,
    sentiment: null,
    engagement: null
  });
  const [includeConfidence, setIncludeConfidence] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [contentMetrics, setContentMetrics] = useState({
    views: 1000,
    likes: 50,
    comments: 10,
    sentiment_score: 0.5,
    engagement_rate: 0.06
  });

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/models/status');
      const status = await response.json();
      setModelStatus(status.model_status);
    } catch (error) {
      setError('Failed to check model status');
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleFileUpload = (fileType, event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFiles(prev => ({ ...prev, [fileType]: file }));
    }
  };

  const runPrediction = async () => {
    if (!contentMetrics.views) {
      setError('Content metrics are required');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/predict-popularity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metrics: contentMetrics,
          include_confidence_interval: includeConfidence,
          model_type: modelType
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const result = await response.json();
      setPredictionResults([result]);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const runBatchPrediction = async () => {
    if (!uploadedFiles.content) {
      setError('Content data file is required');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('content_file', uploadedFiles.content);
      
      if (uploadedFiles.sentiment) {
        formData.append('sentiment_file', uploadedFiles.sentiment);
      }
      
      if (uploadedFiles.engagement) {
        formData.append('engagement_file', uploadedFiles.engagement);
      }
      
      formData.append('model_type', modelType);

      const response = await fetch('http://localhost:8000/batch-predict-popularity', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Batch prediction failed');
      }

      const result = await response.json();
      setPredictionResults(result.results);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/models/train-popularity', {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error('Training failed');
      }

      const result = await response.json();
      await checkModelStatus();
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const predictionData = {
    labels: predictionResults.map((_, index) => `Content ${index + 1}`),
    datasets: [
      {
        label: 'Current Views',
        data: predictionResults.map(result => result.views || 0),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
      },
      {
        label: 'Predicted Future Views',
        data: predictionResults.map(result => result.predicted_future_views),
        backgroundColor: 'rgba(255, 206, 86, 0.8)',
        borderColor: 'rgba(255, 206, 86, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Confidence intervals chart
  const confidenceData = {
    labels: predictionResults.map((_, index) => `Content ${index + 1}`),
    datasets: [
      {
        label: 'Predicted Views',
        data: predictionResults.map(result => result.predicted_future_views),
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.3,
        fill: false,
      },
      {
        label: 'Upper Bound',
        data: predictionResults.map(result => result.prediction_upper_bound || 0),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderDash: [5, 5],
        tension: 0.3,
        fill: false,
      },
      {
        label: 'Lower Bound',
        data: predictionResults.map(result => result.prediction_lower_bound || 0),
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],
        tension: 0.3,
        fill: false,
      },
    ],
  };

  // Growth rate analysis
  const growthData = {
    labels: predictionResults.map((_, index) => `Content ${index + 1}`),
    datasets: [
      {
        label: 'Growth Rate (%)',
        data: predictionResults.map(result => {
          const current = result.views || 0;
          const predicted = result.predicted_future_views;
          return current > 0 ? ((predicted - current) / current * 100) : 0;
        }),
        backgroundColor: predictionResults.map(result => {
          const current = result.views || 0;
          const predicted = result.predicted_future_views;
          const growth = current > 0 ? ((predicted - current) / current * 100) : 0;
          return growth > 50 ? 'rgba(75, 192, 192, 0.8)' : 
                 growth > 20 ? 'rgba(255, 206, 86, 0.8)' : 
                 'rgba(255, 99, 132, 0.8)';
        }),
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Engagement correlation
  const engagementCorrelation = {
    datasets: [
      {
        label: 'Engagement vs Prediction',
        data: predictionResults.map(result => ({
          x: (result.likes + result.comments) / (result.views || 1) * 100,
          y: result.predicted_future_views,
        })),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8,
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
        text: 'Content Popularity Predictions',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const scatterOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Engagement Rate vs Predicted Views',
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Engagement Rate (%)',
        },
        beginAtZero: true,
      },
      y: {
        title: {
          display: true,
          text: 'Predicted Views',
        },
        beginAtZero: true,
      },
    },
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Popularity Prediction
      </Typography>

      {/* Model Status */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <TrendingUp sx={{ mr: 2 }} />
            <Typography variant="h6">Model Status</Typography>
            <Badge 
              badgeContent={modelStatus.popularity_predictor?.trained ? "Trained" : "Not Trained"} 
              color={modelStatus.popularity_predictor?.trained ? "success" : "warning"}
              sx={{ ml: 2 }}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={includeConfidence}
                  onChange={(e) => setIncludeConfidence(e.target.checked)}
                />
              }
              label="Include Confidence Intervals"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showAdvanced}
                  onChange={(e) => setShowAdvanced(e.target.checked)}
                />
              }
              label="Advanced Options"
            />
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={checkModelStatus}
              size="small"
            >
              Refresh
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Model Configuration */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelType}
                label="Model Type"
                onChange={(e) => setModelType(e.target.value)}
              >
                <MenuItem value="random_forest">Random Forest</MenuItem>
                <MenuItem value="linear">Linear Regression</MenuItem>
                <MenuItem value="ridge">Ridge Regression</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Button
              variant="outlined"
              startIcon={<Settings />}
              onClick={trainModel}
              disabled={loading}
              fullWidth
            >
              Train Model
            </Button>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Current Model: {modelType.replace('_', ' ').charAt(0).toUpperCase() + modelType.replace('_', ' ').slice(1)}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs for different operations */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="prediction tabs"
        >
          <Tab label="Single Prediction" />
          <Tab label="Batch Prediction" />
          <Tab label="File Upload" />
        </Tabs>

        <Divider />

        {/* Single Prediction Tab */}
        {tabValue === 0 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Single Content Prediction
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Current Views"
                  type="number"
                  value={contentMetrics.views}
                  onChange={(e) => setContentMetrics(prev => ({ ...prev, views: parseInt(e.target.value) || 0 }))}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Likes"
                  type="number"
                  value={contentMetrics.likes}
                  onChange={(e) => setContentMetrics(prev => ({ ...prev, likes: parseInt(e.target.value) || 0 }))}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Comments"
                  type="number"
                  value={contentMetrics.comments}
                  onChange={(e) => setContentMetrics(prev => ({ ...prev, comments: parseInt(e.target.value) || 0 }))}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle2" gutterBottom>
                  Sentiment Score
                </Typography>
                <Slider
                  value={contentMetrics.sentiment_score}
                  onChange={(e, value) => setContentMetrics(prev => ({ ...prev, sentiment_score: value }))}
                  min={-1}
                  max={1}
                  step={0.1}
                  marks={[
                    { value: -1, label: 'Negative' },
                    { value: 0, label: 'Neutral' },
                    { value: 1, label: 'Positive' }
                  ]}
                  sx={{ mb: 3 }}
                />
                <Typography variant="subtitle2" gutterBottom>
                  Engagement Rate
                </Typography>
                <Slider
                  value={contentMetrics.engagement_rate}
                  onChange={(e, value) => setContentMetrics(prev => ({ ...prev, engagement_rate: value }))}
                  min={0}
                  max={0.2}
                  step={0.01}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 0.05, label: '5%' },
                    { value: 0.1, label: '10%' },
                    { value: 0.2, label: '20%' }
                  ]}
                />
              </Grid>
            </Grid>
            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={runPrediction}
                disabled={loading}
                startIcon={<Assessment />}
                size="large"
              >
                Predict Popularity
              </Button>
              {showAdvanced && (
                <Button
                  variant="outlined"
                  startIcon={<Settings />}
                >
                  Advanced Settings
                </Button>
              )}
            </Box>
          </Box>
        )}

        {/* Batch Prediction Tab */}
        {tabValue === 1 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Batch Prediction
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Upload multiple content items for batch prediction processing.
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <input
                accept=".csv"
                style={{ display: 'none' }}
                id="batch-content-upload"
                type="file"
                onChange={(e) => handleFileUpload('content', e)}
              />
              <label htmlFor="batch-content-upload">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<Upload />}
                >
                  Upload Content CSV
                </Button>
              </label>
              {uploadedFiles.content && (
                <Typography variant="body2">
                  Selected: {uploadedFiles.content.name}
                </Typography>
              )}
            </Box>
            <Button
              variant="contained"
              onClick={runBatchPrediction}
              disabled={loading || !uploadedFiles.content}
              startIcon={<Assessment />}
              size="large"
            >
              Run Batch Prediction
            </Button>
          </Box>
        )}

        {/* File Upload Tab */}
        {tabValue === 2 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              File Upload with Multiple Data Sources
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Content Data (Required)
                  </Typography>
                  <input
                    accept=".csv"
                    style={{ display: 'none' }}
                    id="content-file-upload"
                    type="file"
                    onChange={(e) => handleFileUpload('content', e)}
                  />
                  <label htmlFor="content-file-upload">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<Upload />}
                      fullWidth
                    >
                      Upload Content CSV
                    </Button>
                  </label>
                  {uploadedFiles.content && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Selected: {uploadedFiles.content.name}
                    </Typography>
                  )}
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Sentiment Data (Optional)
                  </Typography>
                  <input
                    accept=".csv"
                    style={{ display: 'none' }}
                    id="sentiment-file-upload"
                    type="file"
                    onChange={(e) => handleFileUpload('sentiment', e)}
                  />
                  <label htmlFor="sentiment-file-upload">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<Upload />}
                      fullWidth
                    >
                      Upload Sentiment CSV
                    </Button>
                  </label>
                  {uploadedFiles.sentiment && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Selected: {uploadedFiles.sentiment.name}
                    </Typography>
                  )}
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Engagement Data (Optional)
                  </Typography>
                  <input
                    accept=".csv"
                    style={{ display: 'none' }}
                    id="engagement-file-upload"
                    type="file"
                    onChange={(e) => handleFileUpload('engagement', e)}
                  />
                  <label htmlFor="engagement-file-upload">
                    <Button
                      variant="outlined"
                      component="span"
                      startIcon={<Upload />}
                      fullWidth
                    >
                      Upload Engagement CSV
                    </Button>
                  </label>
                  {uploadedFiles.engagement && (
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      Selected: {uploadedFiles.engagement.name}
                    </Typography>
                  )}
                </Paper>
              </Grid>
            </Grid>
            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                onClick={runBatchPrediction}
                disabled={loading || !uploadedFiles.content}
                startIcon={<Assessment />}
                size="large"
              >
                Process Files
              </Button>
            </Box>
          </Box>
        )}
      </Paper>

      {/* Loading State */}
      {loading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Processing prediction...
          </Typography>
        </Box>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Prediction Results */}
      {predictionResults.length > 0 && (
        <Grid container spacing={3}>
          {/* Charts */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Current vs Predicted Views
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar data={predictionData} options={chartOptions} />
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Growth Rate Analysis
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar data={growthData} options={chartOptions} />
              </Box>
            </Paper>
          </Grid>

          {includeConfidence && (
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Prediction Confidence Intervals
                </Typography>
                <Box sx={{ height: 300 }}>
                  <Line data={confidenceData} options={chartOptions} />
                </Box>
              </Paper>
            </Grid>
          )}

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Engagement Correlation
              </Typography>
              <Box sx={{ height: 300 }}>
                <Scatter data={engagementCorrelation} options={scatterOptions} />
              </Box>
            </Paper>
          </Grid>

          {/* Results Table */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Prediction Results ({predictionResults.length})
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <IconButton size="small">
                    <Download />
                  </IconButton>
                  <IconButton size="small">
                    <Share />
                  </IconButton>
                </Box>
              </Box>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Content ID</TableCell>
                      <TableCell>Current Views</TableCell>
                      <TableCell>Predicted Views</TableCell>
                      <TableCell>Growth Rate</TableCell>
                      {includeConfidence && <TableCell>Confidence Interval</TableCell>}
                      <TableCell>Model</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictionResults.map((result, index) => {
                      const current = result.views || 0;
                      const predicted = result.predicted_future_views;
                      const growthRate = current > 0 ? ((predicted - current) / current * 100) : 0;
                      
                      return (
                        <TableRow key={index}>
                          <TableCell>{result.content_id || `Content ${index + 1}`}</TableCell>
                          <TableCell>{current.toLocaleString()}</TableCell>
                          <TableCell>
                            <Chip
                              label={predicted.toLocaleString()}
                              color="primary"
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={`${growthRate.toFixed(1)}%`}
                              color={growthRate > 50 ? 'success' : growthRate > 20 ? 'warning' : 'default'}
                              size="small"
                            />
                          </TableCell>
                          {includeConfidence && (
                            <TableCell>
                              {result.prediction_lower_bound && result.prediction_upper_bound ? 
                                `${result.prediction_lower_bound.toLocaleString()} - ${result.prediction_upper_bound.toLocaleString()}` : 
                                'N/A'
                              }
                            </TableCell>
                          )}
                          <TableCell>
                            <Chip
                              label={result.model_type || modelType}
                              variant="outlined"
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>

          {/* Summary Statistics */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Prediction Summary
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="primary.main">
                        {predictionResults.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Predictions
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="success.main">
                        {Math.max(...predictionResults.map(r => r.predicted_future_views)).toLocaleString()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Max Predicted Views
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="warning.main">
                        {predictionResults.reduce((sum, r) => sum + r.predicted_future_views, 0) / predictionResults.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Avg Predicted Views
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="info.main">
                        {predictionResults.filter(r => {
                          const current = r.views || 0;
                          const predicted = r.predicted_future_views;
                          return current > 0 && ((predicted - current) / current * 100) > 50;
                        }).length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        High Growth (>50%)
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default PopularityPrediction;
