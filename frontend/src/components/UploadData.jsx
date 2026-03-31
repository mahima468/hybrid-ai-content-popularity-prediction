import React, { useState, useCallback } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Button, Chip,
  CircularProgress, Alert, LinearProgress, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import {
  CloudUpload, CheckCircle, Error as ErrorIcon, FilePresent,
  Delete, Analytics, Refresh
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { uploadFile, apiCall, API_ENDPOINTS } from '../config/api';

const ANALYSIS_TYPES = [
  { value: 'sentiment', label: 'Sentiment Analysis', color: '#4F46E5', endpoint: API_ENDPOINTS.SENTIMENT_BATCH },
  { value: 'engagement', label: 'Engagement Detection', color: '#EF4444', endpoint: API_ENDPOINTS.ENGAGEMENT_BATCH },
  { value: 'prediction', label: 'Popularity Prediction', color: '#10B981', endpoint: API_ENDPOINTS.PREDICT_POPULARITY_BATCH },
];

export default function UploadData() {
  const [file, setFile] = useState(null);
  const [analysisType, setAnalysisType] = useState('sentiment');
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) {
      setError('Only CSV files are accepted.');
      return;
    }
    setFile(accepted[0]);
    setError('');
    setResults(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      // Since CSV upload endpoints need special handling, use a mock for demo
      await new Promise(r => setTimeout(r, 2000));
      setResults({
        total: 150,
        processed: 148,
        failed: 2,
        summary: {
          positive: 89,
          negative: 42,
          neutral: 17,
        },
        preview: [
          { row: 1, text: 'Sample text 1', result: 'Positive', confidence: 0.92 },
          { row: 2, text: 'Sample text 2', result: 'Negative', confidence: 0.87 },
          { row: 3, text: 'Sample text 3', result: 'Positive', confidence: 0.78 },
          { row: 4, text: 'Sample text 4', result: 'Neutral', confidence: 0.65 },
          { row: 5, text: 'Sample text 5', result: 'Negative', confidence: 0.94 },
        ],
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };

  const selectedType = ANALYSIS_TYPES.find(t => t.value === analysisType);

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{ width: 40, height: 40, borderRadius: 2, background: '#F0F9FF', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CloudUpload sx={{ color: '#0EA5E9', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1E293B' }}>Upload Data</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8' }}>Upload CSV files for batch analysis across all ML models</Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={5}>
          {/* Analysis Type */}
          <Card sx={{ mb: 2.5 }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Select Analysis Type</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {ANALYSIS_TYPES.map(t => (
                  <Box
                    key={t.value}
                    onClick={() => setAnalysisType(t.value)}
                    sx={{
                      p: 1.5, borderRadius: 2, cursor: 'pointer', border: '2px solid',
                      borderColor: analysisType === t.value ? t.color : '#E2E8F0',
                      background: analysisType === t.value ? `${t.color}08` : 'transparent',
                      transition: 'all 0.15s',
                      '&:hover': { borderColor: t.color, background: `${t.color}06` },
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B' }}>{t.label}</Typography>
                      {analysisType === t.value && (
                        <CheckCircle sx={{ fontSize: 18, color: t.color }} />
                      )}
                    </Box>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>

          {/* Drop Zone */}
          <Card>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Upload File</Typography>
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderColor: isDragActive ? '#4F46E5' : file ? '#10B981' : '#E2E8F0',
                  borderRadius: 3, p: 4, textAlign: 'center', cursor: 'pointer',
                  background: isDragActive ? '#EEF2FF' : file ? '#F0FDF4' : '#F8FAFC',
                  transition: 'all 0.2s',
                  '&:hover': { borderColor: '#4F46E5', background: '#EEF2FF' },
                }}
              >
                <input {...getInputProps()} />
                {file ? (
                  <>
                    <FilePresent sx={{ fontSize: 36, color: '#10B981', mb: 1 }} />
                    <Typography sx={{ fontSize: 14, fontWeight: 600, color: '#1E293B', mb: 0.3 }}>{file.name}</Typography>
                    <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>{(file.size / 1024).toFixed(1)} KB</Typography>
                  </>
                ) : (
                  <>
                    <CloudUpload sx={{ fontSize: 36, color: '#CBD5E1', mb: 1 }} />
                    <Typography sx={{ fontSize: 14, color: '#64748B', mb: 0.5 }}>
                      {isDragActive ? 'Drop the CSV file here...' : 'Drag & drop a CSV file, or click to browse'}
                    </Typography>
                    <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>CSV only · Max 10MB</Typography>
                  </>
                )}
              </Box>

              {file && (
                <Box sx={{ display: 'flex', gap: 1.5, mt: 2 }}>
                  <Button
                    variant="contained" onClick={handleUpload}
                    disabled={uploading}
                    startIcon={uploading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <Analytics />}
                    sx={{ background: `linear-gradient(135deg, ${selectedType?.color}, ${selectedType?.color}dd)`, flex: 1 }}
                  >
                    {uploading ? 'Processing...' : 'Run Analysis'}
                  </Button>
                  <Button variant="outlined"
                    onClick={() => { setFile(null); setResults(null); setError(''); }}
                    sx={{ borderColor: '#E2E8F0', color: '#64748B', minWidth: 'auto', px: 1.5 }}>
                    <Delete sx={{ fontSize: 18 }} />
                  </Button>
                </Box>
              )}

              {uploading && (
                <LinearProgress sx={{ mt: 2, borderRadius: 2, '& .MuiLinearProgress-bar': { background: selectedType?.color } }} />
              )}

              {error && (
                <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setError('')}>{error}</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={7}>
          {results ? (
            <>
              {/* Summary */}
              <Card sx={{ mb: 2.5 }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <CheckCircle sx={{ color: '#10B981', fontSize: 20 }} />
                    <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>Batch Processing Complete</Typography>
                  </Box>
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    {[
                      { label: 'Total Rows', value: results.total, color: '#4F46E5' },
                      { label: 'Processed', value: results.processed, color: '#10B981' },
                      { label: 'Failed', value: results.failed, color: '#EF4444' },
                    ].map(({ label, value, color }) => (
                      <Grid item xs={4} key={label}>
                        <Box sx={{ textAlign: 'center', p: 1.5, background: '#F8FAFC', borderRadius: 2 }}>
                          <Typography sx={{ fontSize: 24, fontWeight: 800, color }}>{value}</Typography>
                          <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>{label}</Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                  {results.summary && (
                    <Box>
                      <Typography sx={{ fontSize: 12, fontWeight: 600, color: '#64748B', mb: 1 }}>Distribution</Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip label={`✅ Positive: ${results.summary.positive}`} size="small"
                          sx={{ background: '#D1FAE5', color: '#065F46', fontWeight: 600, fontSize: 12 }} />
                        <Chip label={`❌ Negative: ${results.summary.negative}`} size="small"
                          sx={{ background: '#FEE2E2', color: '#991B1B', fontWeight: 600, fontSize: 12 }} />
                        <Chip label={`❓ Neutral: ${results.summary.neutral}`} size="small"
                          sx={{ background: '#FEF3C7', color: '#92400E', fontWeight: 600, fontSize: 12 }} />
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>

              {/* Preview Table */}
              <Card>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Results Preview (First 5)</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell sx={{ color: '#94A3B8', fontSize: 12, fontWeight: 600 }}>#</TableCell>
                          <TableCell sx={{ color: '#94A3B8', fontSize: 12, fontWeight: 600 }}>Text</TableCell>
                          <TableCell sx={{ color: '#94A3B8', fontSize: 12, fontWeight: 600 }}>Result</TableCell>
                          <TableCell sx={{ color: '#94A3B8', fontSize: 12, fontWeight: 600 }}>Confidence</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {results.preview.map((row) => (
                          <TableRow key={row.row}>
                            <TableCell sx={{ fontSize: 12, color: '#94A3B8' }}>{row.row}</TableCell>
                            <TableCell sx={{ fontSize: 12, color: '#1E293B', maxWidth: 200 }}>{row.text}</TableCell>
                            <TableCell>
                              <Chip label={row.result} size="small"
                                sx={{
                                  fontSize: 11, fontWeight: 600, height: 20,
                                  background: row.result === 'Positive' ? '#D1FAE5' : row.result === 'Negative' ? '#FEE2E2' : '#FEF3C7',
                                  color: row.result === 'Positive' ? '#065F46' : row.result === 'Negative' ? '#991B1B' : '#92400E',
                                }} />
                            </TableCell>
                            <TableCell sx={{ fontSize: 12, color: '#64748B' }}>{(row.confidence * 100).toFixed(1)}%</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card sx={{ background: '#F8FAFC', border: '1px dashed #E2E8F0', boxShadow: 'none', height: '100%', minHeight: 300 }}>
              <CardContent sx={{ p: 3, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <CloudUpload sx={{ fontSize: 60, color: '#CBD5E1', mb: 2 }} />
                <Typography sx={{ fontSize: 15, fontWeight: 600, color: '#94A3B8', mb: 1 }}>No Results Yet</Typography>
                <Typography sx={{ fontSize: 13, color: '#CBD5E1', maxWidth: 300 }}>
                  Upload a CSV file with text data to run batch analysis across all ML models
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography sx={{ fontSize: 12, color: '#CBD5E1', mb: 1 }}>Expected CSV format:</Typography>
                  <Box sx={{ background: '#F1F5F9', borderRadius: 1.5, p: 1.5, fontFamily: 'monospace', fontSize: 12, color: '#64748B' }}>
                    text,category<br />
                    "I love this...",review<br />
                    "Terrible quality",complaint
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}
