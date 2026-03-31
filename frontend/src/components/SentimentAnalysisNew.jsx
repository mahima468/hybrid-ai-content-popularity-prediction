/**
 * Sentiment Analysis Component
 * Provides interface for sentiment analysis of text content
 * Supports both ML model and TextBlob analysis methods with Chart.js visualizations
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
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
  IconButton
} from '@mui/material';
import {
  SentimentSatisfied,
  SentimentVeryDissatisfied,
  SentimentNeutral,
  Analytics,
  Upload,
  Refresh,
  CheckCircle,
  Error,
  ExpandMore,
  TrendingUp,
  Timeline,
  BarChart,
  PieChart,
  FilterList,
  Download,
  Share
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
import { Bar, Doughnut, Line } from 'react-chartjs-2';

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

const SentimentAnalysis = () => {
  const [tabValue, setTabValue] = useState(0);
  const [inputText, setInputText] = useState('');
  const [analysisMethod, setAnalysisMethod] = useState('logistic');
  const [analysisResults, setAnalysisResults] = useState([]);
  const [modelStatus, setModelStatus] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [batchTexts, setBatchTexts] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [filterResults, setFilterResults] = useState('all');

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

  const analyzeSentiment = async () => {
    if (!inputText.trim()) {
      setError('Please enter text to analyze');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          method: analysisMethod
        })
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const result = await response.json();
      setAnalysisResults([result]);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    if (!batchTexts.trim()) {
      setError('Please enter texts to analyze');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const texts = batchTexts.split('\n').filter(text => text.trim());
      const response = await fetch('http://localhost:8000/batch-analyze-sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          texts: texts,
          method: analysisMethod
        })
      });

      if (!response.ok) {
        throw new Error('Batch analysis failed');
      }

      const result = await response.json();
      setAnalysisResults(result.results);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('data_file', file);
      formData.append('text_column', 'text');
      formData.append('method', analysisMethod);

      const response = await fetch('http://localhost:8000/batch-analyze-sentiment', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('File analysis failed');
      }

      const result = await response.json();
      setAnalysisResults(result.results);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <SentimentSatisfied color="success" />;
      case 'negative':
        return <SentimentVeryDissatisfied color="error" />;
      case 'neutral':
        return <SentimentNeutral color="warning" />;
      default:
        return <SentimentNeutral />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'success';
      case 'negative':
        return 'error';
      case 'neutral':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Filter results based on selected filter
  const getFilteredResults = () => {
    if (filterResults === 'all') return analysisResults;
    return analysisResults.filter(result => 
      (result.sentiment === filterResults) || (result.sentiment_label === filterResults)
    );
  };

  // Prepare chart data
  const sentimentCounts = getFilteredResults().reduce((acc, result) => {
    const sentiment = result.sentiment_label || result.sentiment;
    acc[sentiment] = (acc[sentiment] || 0) + 1;
    return acc;
  }, {});

  const sentimentData = {
    labels: Object.keys(sentimentCounts),
    datasets: [
      {
        label: 'Sentiment Distribution',
        data: Object.values(sentimentCounts),
        backgroundColor: [
          'rgba(75, 192, 192, 0.8)',
          'rgba(255, 99, 132, 0.8)',
          'rgba(255, 206, 86, 0.8)',
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(255, 206, 86, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  // Confidence scores over time
  const confidenceData = {
    labels: getFilteredResults().map((_, index) => `Text ${index + 1}`),
    datasets: [
      {
        label: 'Confidence Scores',
        data: getFilteredResults().map(result => result.confidence || 0),
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.3,
        fill: true,
      },
    ],
  };

  // Sentiment score distribution
  const scoreDistribution = {
    labels: ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
    datasets: [
      {
        label: 'Confidence Score Distribution',
        data: [
          getFilteredResults().filter(r => (r.confidence || 0) <= 0.2).length,
          getFilteredResults().filter(r => (r.confidence || 0) > 0.2 && (r.confidence || 0) <= 0.4).length,
          getFilteredResults().filter(r => (r.confidence || 0) > 0.4 && (r.confidence || 0) <= 0.6).length,
          getFilteredResults().filter(r => (r.confidence || 0) > 0.6 && (r.confidence || 0) <= 0.8).length,
          getFilteredResults().filter(r => (r.confidence || 0) > 0.8).length,
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(153, 102, 255, 1)',
        ],
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
        text: 'Sentiment Analysis Results',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Sentiment Analysis
      </Typography>

      {/* Model Status */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Analytics sx={{ mr: 2 }} />
            <Typography variant="h6">Model Status</Typography>
            <Badge 
              badgeContent={modelStatus.sentiment_model?.trained ? "Ready" : "Not Ready"} 
              color={modelStatus.sentiment_model?.trained ? "success" : "warning"}
              sx={{ ml: 2 }}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
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

      {/* Analysis Method Selection */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Analysis Method</InputLabel>
              <Select
                value={analysisMethod}
                label="Analysis Method"
                onChange={(e) => setAnalysisMethod(e.target.value)}
              >
                <MenuItem value="logistic">Logistic Regression</MenuItem>
                <MenuItem value="naive_bayes">Naive Bayes</MenuItem>
                <MenuItem value="random_forest">Random Forest</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Filter Results</InputLabel>
              <Select
                value={filterResults}
                label="Filter Results"
                onChange={(e) => setFilterResults(e.target.value)}
              >
                <MenuItem value="all">All Results</MenuItem>
                <MenuItem value="positive">Positive Only</MenuItem>
                <MenuItem value="negative">Negative Only</MenuItem>
                <MenuItem value="neutral">Neutral Only</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Total Analyzed: {getFilteredResults().length}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Tabs for different input methods */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="sentiment analysis tabs"
        >
          <Tab label="Single Text" />
          <Tab label="Batch Text" />
          <Tab label="File Upload" />
        </Tabs>

        <Divider />

        {/* Single Text Tab */}
        {tabValue === 0 && (
          <Box sx={{ p: 3 }}>
            <TextField
              fullWidth
              multiline
              rows={6}
              label="Enter text to analyze"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Type or paste your text here..."
              sx={{ mb: 2 }}
            />
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={analyzeSentiment}
                disabled={loading || !inputText.trim()}
                startIcon={<Analytics />}
              >
                Analyze Sentiment
              </Button>
              {showAdvanced && (
                <Button
                  variant="outlined"
                  startIcon={<FilterList />}
                >
                  Advanced Options
                </Button>
              )}
            </Box>
          </Box>
        )}

        {/* Batch Text Tab */}
        {tabValue === 1 && (
          <Box sx={{ p: 3 }}>
            <TextField
              fullWidth
              multiline
              rows={8}
              label="Enter multiple texts (one per line)"
              value={batchTexts}
              onChange={(e) => setBatchTexts(e.target.value)}
              placeholder="Enter each text on a new line..."
              sx={{ mb: 2 }}
            />
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                onClick={analyzeBatch}
                disabled={loading || !batchTexts.trim()}
                startIcon={<Analytics />}
              >
                Analyze Batch
              </Button>
              <Button
                variant="outlined"
                startIcon={<Download />}
              >
                Export Template
              </Button>
            </Box>
          </Box>
        )}

        {/* File Upload Tab */}
        {tabValue === 2 && (
          <Box sx={{ p: 3 }}>
            <input
              accept=".csv"
              style={{ display: 'none' }}
              id="sentiment-file-upload"
              type="file"
              onChange={handleFileUpload}
            />
            <label htmlFor="sentiment-file-upload">
              <Button
                variant="outlined"
                component="span"
                startIcon={<Upload />}
                sx={{ mb: 2 }}
              >
                Upload CSV File
              </Button>
            </label>
            {uploadedFile && (
              <Typography variant="body2" sx={{ ml: 2 }}>
                Selected: {uploadedFile.name}
              </Typography>
            )}
          </Box>
        )}
      </Paper>

      {/* Loading State */}
      {loading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Analyzing sentiment...
          </Typography>
        </Box>
      )}

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Results Section */}
      {analysisResults.length > 0 && (
        <Grid container spacing={3}>
          {/* Charts */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Sentiment Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <Doughnut data={sentimentData} />
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Confidence Scores
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line data={confidenceData} options={chartOptions} />
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Score Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar data={scoreDistribution} options={chartOptions} />
              </Box>
            </Paper>
          </Grid>

          {/* Results Table */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Analysis Results ({getFilteredResults().length})
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
                      <TableCell>Text</TableCell>
                      <TableCell>Sentiment</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Method</TableCell>
                      <TableCell>Processing Time</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {getFilteredResults().map((result, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {result.text.length > 100 
                            ? `${result.text.substring(0, 100)}...` 
                            : result.text
                          }
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            {getSentimentIcon(result.sentiment_label || result.sentiment)}
                            <Box component="span" sx={{ ml: 1 }}>
                              {result.sentiment_label || result.sentiment}
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={`${(result.confidence * 100).toFixed(1)}%`}
                            color={getSentimentColor(result.sentiment_label || result.sentiment)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={result.method || analysisMethod}
                            variant="outlined"
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {result.processing_time ? `${result.processing_time.toFixed(3)}s` : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>

          {/* Summary Statistics */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Summary Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="success.main">
                        {sentimentCounts.positive || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Positive
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="warning.main">
                        {sentimentCounts.neutral || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Neutral
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="error.main">
                        {sentimentCounts.negative || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Negative
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h4" color="primary.main">
                        {getFilteredResults().length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Analyzed
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

export default SentimentAnalysis;
