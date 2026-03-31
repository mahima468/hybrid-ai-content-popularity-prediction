/**
 * Upload Data Component
 * Handles CSV data upload for the Hybrid AI Content Popularity Prediction System
 * Features file validation, preview, and data processing
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Alert,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Switch,
  FormControlLabel,
  TextField,
  MenuItem,
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import {
  CloudUpload,
  Description,
  CheckCircle,
  Error,
  Preview,
  Send,
  Delete,
  ExpandMore,
  UploadFile,
  Analytics,
  TrendingUp,
  SentimentSatisfied,
  Security
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const UploadData = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [files, setFiles] = useState({
    engagement: null,
    sentiment: null,
    popularity: null
  });
  const [previews, setPreviews] = useState({});
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadStatus, setUploadStatus] = useState({});
  const [processingResults, setProcessingResults] = useState({});
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const steps = [
    {
      label: 'Upload Files',
      description: 'Upload your CSV files containing engagement, sentiment, and popularity data'
    },
    {
      label: 'Preview & Validate',
      description: 'Review data preview and validate file formats'
    },
    {
      label: 'Process & Analyze',
      description: 'Process data and run AI analysis'
    }
  ];

  const fileTypes = [
    {
      key: 'engagement',
      title: 'Engagement Data',
      description: 'Views, likes, comments, and engagement metrics',
      icon: <Analytics />,
      required: true,
      example: 'views,likes,comments,content_id,user_id,timestamp\n1000,50,10,1,101,2024-01-15T10:30:00',
      color: 'primary'
    },
    {
      key: 'sentiment',
      title: 'Sentiment Data',
      description: 'Sentiment scores and analysis results',
      icon: <SentimentSatisfied />,
      required: false,
      example: 'content_id,sentiment_score,sentiment_label,confidence\n1,0.8,positive,0.92',
      color: 'success'
    },
    {
      key: 'popularity',
      title: 'Popularity Data',
      description: 'Content metrics and future popularity targets',
      icon: <TrendingUp />,
      required: false,
      example: 'content_id,views,likes,comments,sentiment_score,engagement_rate,future_views\n1,1000,50,10,0.8,0.06,2500',
      color: 'warning'
    }
  ];

  const onDrop = useCallback((acceptedFiles, fileType) => {
    const file = acceptedFiles[0];
    if (file && file.type === 'text/csv') {
      setFiles(prev => ({ ...prev, [fileType]: file }));
      setUploadStatus(prev => ({ ...prev, [fileType]: 'pending' }));
      setError('');
      previewFile(file, fileType);
    } else {
      setUploadStatus(prev => ({ ...prev, [fileType]: 'invalid' }));
      setError('Please upload a valid CSV file');
    }
  }, []);

  const previewFile = async (file, fileType) => {
    try {
      const text = await file.text();
      const lines = text.split('\n');
      const headers = lines[0].split(',');
      const dataRows = lines.slice(1, 6).map(row => row.split(','));
      
      setPreviews(prev => ({
        ...prev,
        [fileType]: {
          headers,
          rows: dataRows,
          totalRows: lines.length - 1,
          filename: file.name,
          size: (file.size / 1024).toFixed(2) + ' KB'
        }
      }));
    } catch (error) {
      setUploadStatus(prev => ({ ...prev, [fileType]: 'error' }));
      setError('Error reading file: ' + error.message);
    }
  };

  const removeFile = (fileType) => {
    setFiles(prev => ({ ...prev, [fileType]: null }));
    setPreviews(prev => ({ ...prev, [fileType]: null }));
    setUploadStatus(prev => ({ ...prev, [fileType]: 'none' }));
    setUploadProgress(prev => ({ ...prev, [fileType]: 0 }));
  };

  const validateFiles = () => {
    if (!files.engagement) {
      setError('Engagement data file is required');
      return false;
    }

    for (const [fileType, preview] of Object.entries(previews)) {
      if (preview && preview.totalRows < 10) {
        setError(`${fileType} file has insufficient data (minimum 10 rows required)`);
        return false;
      }
    }

    return true;
  };

  const handleNext = () => {
    if (activeStep === 0 && !validateFiles()) {
      return;
    }
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const processFiles = async () => {
    setSuccess('');
    setError('');
    
    try {
      // Process engagement data
      if (files.engagement) {
        await processFile('engagement', '/api/engagement/detect/file');
      }

      // Process sentiment data
      if (files.sentiment) {
        await processFile('sentiment', '/api/sentiment/analyze/batch');
      }

      // Process popularity data
      if (files.popularity) {
        await processFile('popularity', '/api/prediction/predict/file');
      }

      setSuccess('All files processed successfully!');
      setActiveStep(steps.length);
    } catch (error) {
      setError('Processing failed: ' + error.message);
    }
  };

  const processFile = async (fileType, endpoint) => {
    const file = files[fileType];
    if (!file) return;

    setUploadStatus(prev => ({ ...prev, [fileType]: 'uploading' }));
    setUploadProgress(prev => ({ ...prev, [fileType]: 0 }));

    const formData = new FormData();
    formData.append('data_file', file);

    try {
      const response = await axios.post(`http://localhost:8000${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(prev => ({ ...prev, [fileType]: progress }));
        },
      });

      setUploadProgress(prev => ({ ...prev, [fileType]: 100 }));
      setUploadStatus(prev => ({ ...prev, [fileType]: 'success' }));
      setProcessingResults(prev => ({ ...prev, [fileType]: response.data }));

    } catch (error) {
      setUploadStatus(prev => ({ ...prev, [fileType]: 'error' }));
      throw error;
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle color="success" />;
      case 'error':
      case 'invalid':
        return <Error color="error" />;
      case 'uploading':
        return <LinearProgress sx={{ width: 24 }} />;
      default:
        return <Description />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'error':
      case 'invalid':
        return 'error';
      case 'uploading':
        return 'warning';
      case 'pending':
        return 'info';
      default:
        return 'default';
    }
  };

  const Dropzone = ({ fileType, fileTypeInfo }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop: (acceptedFiles) => onDrop(acceptedFiles, fileType),
      accept: {
        'text/csv': ['.csv']
      },
      multiple: false
    });

    return (
      <Box
        {...getRootProps()}
        sx={{
          border: `2px dashed ${isDragActive ? '#1976d2' : '#ccc'}`,
          borderRadius: 2,
          p: 3,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? '#f5f5f5' : 'transparent',
          '&:hover': {
            backgroundColor: '#f9f9f9',
            borderColor: '#1976d2'
          }
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 48, color: '#ccc', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop the CSV file here' : `Upload ${fileTypeInfo.title}`}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Drag & drop a CSV file here, or click to select
        </Typography>
        {fileTypeInfo.required && (
          <Chip label="Required" color="error" size="small" sx={{ mt: 1 }} />
        )}
      </Box>
    );
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Data Upload & Processing
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}

      <Stepper activeStep={activeStep} orientation="vertical">
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel>
              {step.label}
            </StepLabel>
            <StepContent>
              <Typography>{step.description}</Typography>
              <Box sx={{ mb: 2 }}>
                {index === 0 && (
                  <Grid container spacing={3}>
                    {fileTypes.map((fileType) => (
                      <Grid item xs={12} md={4} key={fileType.key}>
                        <Card>
                          <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                              {fileType.icon}
                              <Typography variant="h6" sx={{ ml: 1 }}>
                                {fileType.title}
                              </Typography>
                            </Box>
                            
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                              {fileType.description}
                            </Typography>

                            {!files[fileType.key] ? (
                              <Dropzone fileType={fileType.key} fileTypeInfo={fileType} />
                            ) : (
                              <Box sx={{ mt: 2 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                                  <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center' }}>
                                    {getStatusIcon(uploadStatus[fileType.key])}
                                    <Box component="span" sx={{ ml: 1 }}>
                                      {files[fileType.key].name}
                                    </Box>
                                  </Typography>
                                  <Button
                                    size="small"
                                    onClick={() => removeFile(fileType.key)}
                                    startIcon={<Delete />}
                                  >
                                    Remove
                                  </Button>
                                </Box>

                                {uploadStatus[fileType.key] === 'uploading' && (
                                  <LinearProgress
                                    variant="determinate"
                                    value={uploadProgress[fileType.key] || 0}
                                    sx={{ mb: 1 }}
                                  />
                                )}

                                <Chip
                                  label={uploadStatus[fileType.key] || 'pending'}
                                  color={getStatusColor(uploadStatus[fileType.key])}
                                  size="small"
                                  sx={{ mb: 2 }}
                                />
                              </Box>
                            )}

                            <Accordion sx={{ mt: 2 }}>
                              <AccordionSummary expandIcon={<ExpandMore />}>
                                <Typography variant="subtitle2">Example Format</Typography>
                              </AccordionSummary>
                              <AccordionDetails>
                                <Typography variant="body2" component="pre" sx={{ 
                                  backgroundColor: 'grey.100', 
                                  p: 1, 
                                  borderRadius: 1,
                                  fontSize: '0.75rem',
                                  overflow: 'auto'
                                }}>
                                  {fileType.example}
                                </Typography>
                              </AccordionDetails>
                            </Accordion>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {index === 1 && (
                  <Grid container spacing={3}>
                    {Object.entries(previews).map(([fileType, preview]) => (
                      <Grid item xs={12} md={6} key={fileType}>
                        <Card>
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {fileType.charAt(0).toUpperCase() + fileType.slice(1)} Data Preview
                            </Typography>
                            <Box sx={{ mb: 2 }}>
                              <Typography variant="body2" color="text.secondary">
                                File: {preview.filename} ({preview.size})
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Total Rows: {preview.totalRows}
                              </Typography>
                            </Box>
                            
                            <TableContainer component={Paper} variant="outlined">
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    {preview.headers.map((header, index) => (
                                      <TableCell key={index}>{header}</TableCell>
                                    ))}
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {preview.rows.map((row, rowIndex) => (
                                    <TableRow key={rowIndex}>
                                      {row.map((cell, cellIndex) => (
                                        <TableCell key={cellIndex}>{cell}</TableCell>
                                      ))}
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {index === 2 && (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Processing Results
                    </Typography>
                    <Grid container spacing={3}>
                      {Object.entries(processingResults).map(([fileType, result]) => (
                        <Grid item xs={12} md={4} key={fileType}>
                          <Card>
                            <CardContent>
                              <Typography variant="h6" gutterBottom>
                                {fileType.charAt(0).toUpperCase() + fileType.slice(1)} Results
                              </Typography>
                              <Box sx={{ mb: 2 }}>
                                <Chip
                                  label="Completed"
                                  color="success"
                                  size="small"
                                  icon={<CheckCircle />}
                                />
                              </Box>
                              
                              {result.summary && (
                                <List dense>
                                  {Object.entries(result.summary).map(([key, value]) => (
                                    <ListItem key={key}>
                                      <ListItemText
                                        primary={key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1)}
                                        secondary={typeof value === 'object' ? JSON.stringify(value) : value}
                                      />
                                    </ListItem>
                                  ))}
                                </List>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}
              </Box>

              <Box sx={{ mb: 2 }}>
                <div>
                  <Button
                    variant="contained"
                    onClick={index === steps.length - 1 ? processFiles : handleNext}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    {index === steps.length - 1 ? 'Process Files' : 'Next'}
                  </Button>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    sx={{ mt: 1, mr: 1 }}
                  >
                    Back
                  </Button>
                </div>
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>

      {activeStep === steps.length && (
        <Paper square elevation={0} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            All steps completed successfully!
          </Typography>
          <Typography variant="body2">
            Your data has been processed and is ready for analysis. You can now navigate to the Dashboard, Sentiment Analysis, or Popularity Prediction sections to explore the results.
          </Typography>
          <Button onClick={() => setActiveStep(0)} sx={{ mt: 2, mr: 1 }}>
            Reset
          </Button>
        </Paper>
      )}
    </Box>
  );
};

export default UploadData;
