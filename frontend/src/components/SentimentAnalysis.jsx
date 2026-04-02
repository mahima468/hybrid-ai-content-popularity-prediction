import React, { useState } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, TextField, Button,
  FormControl, InputLabel, Select, MenuItem, Chip, CircularProgress,
  Alert, LinearProgress, Divider, Tab, Tabs, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Paper
} from '@mui/material';
import {
  Psychology, CheckCircle, Cancel, HelpOutline,
  Send, DeleteOutline, BarChart, FileUpload, Refresh
} from '@mui/icons-material';
import {
  BarChart as ReBarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend
} from 'recharts';
import { apiCall, uploadFile, logAnalysis, API_ENDPOINTS } from '../config/api';

const METHODS = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'naive_bayes', label: 'Naive Bayes' },
  { value: 'random_forest', label: 'Random Forest' },
];

const sentColor = (s) => s === 'Positive' ? '#10B981' : s === 'Negative' ? '#EF4444' : '#F59E0B';
const sentBg = (s) => s === 'Positive' ? '#D1FAE5' : s === 'Negative' ? '#FEE2E2' : '#FEF3C7';
const sentIcon = (s) => s === 'Positive' ? <CheckCircle sx={{ fontSize: 16 }} /> : s === 'Negative' ? <Cancel sx={{ fontSize: 16 }} /> : <HelpOutline sx={{ fontSize: 16 }} />;

function ConfidenceBar({ value, color }) {
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography sx={{ fontSize: 12, color: '#64748B' }}>Confidence</Typography>
        <Typography sx={{ fontSize: 12, fontWeight: 700, color }}>{(value * 100).toFixed(1)}%</Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 6, borderRadius: 3,
          background: '#F1F5F9',
          '& .MuiLinearProgress-bar': { borderRadius: 3, background: color },
        }}
      />
    </Box>
  );
}

export default function SentimentAnalysis() {
  const [tab, setTab] = useState(0);
  const [method, setMethod] = useState('logistic');
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [batchTexts, setBatchTexts] = useState('');
  const [batchResults, setBatchResults] = useState([]);
  const [batchLoading, setBatchLoading] = useState(false);

  const [history, setHistory] = useState([]);

  const handleSingle = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await apiCall(API_ENDPOINTS.SENTIMENT_ANALYZE, {
        method: 'POST',
        body: JSON.stringify({ text: text.trim(), method }),
      });
      setResult(data);
      setHistory(prev => [{ ...data, inputText: text.trim() }, ...prev].slice(0, 10));
      logAnalysis('sentiment', data.sentiment, data.confidence);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    const lines = batchTexts.split('\n').map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    setBatchLoading(true);
    setError('');
    try {
      const data = await apiCall(API_ENDPOINTS.SENTIMENT_BATCH, {
        method: 'POST',
        body: JSON.stringify({ texts: lines, method }),
      });
      setBatchResults(data.results || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setBatchLoading(false);
    }
  };

  const sampleTexts = [
    'This product is absolutely amazing! The quality exceeded my expectations.',
    'Terrible experience. The product broke after one day and customer service was unhelpful.',
    'It\'s okay, not great but not bad either. Gets the job done.',
  ];

  const batchStats = batchResults.length
    ? {
        positive: batchResults.filter(r => r.sentiment === 'Positive').length,
        negative: batchResults.filter(r => r.sentiment === 'Negative').length,
        neutral: batchResults.filter(r => r.sentiment !== 'Positive' && r.sentiment !== 'Negative').length,
      }
    : null;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{ width: 40, height: 40, borderRadius: 2, background: '#EEF2FF', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Psychology sx={{ color: '#4F46E5', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1E293B' }}>Sentiment Analysis</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8' }}>ML-powered sentiment classification using TF-IDF + multiple classifiers</Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={2.5}>
        {/* Left: Input */}
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent sx={{ p: 2.5 }}>
              {/* Method selector */}
              <Box sx={{ display: 'flex', gap: 2, mb: 2.5, alignItems: 'center', flexWrap: 'wrap' }}>
                <FormControl size="small" sx={{ minWidth: 200 }}>
                  <InputLabel>Analysis Method</InputLabel>
                  <Select value={method} label="Analysis Method" onChange={e => setMethod(e.target.value)}
                    sx={{ borderRadius: 2 }}>
                    {METHODS.map(m => (
                      <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{
                  '& .MuiTab-root': { fontSize: 13, textTransform: 'none', minHeight: 36, py: 0, minWidth: 90 },
                  '& .MuiTabs-indicator': { background: '#4F46E5' },
                }}>
                  <Tab label="Single" />
                  <Tab label="Batch" />
                </Tabs>
              </Box>

              {tab === 0 && (
                <>
                  <TextField
                    fullWidth
                    multiline
                    rows={5}
                    placeholder="Enter text to analyze sentiment... (e.g., product review, social media post, comment)"
                    value={text}
                    onChange={e => setText(e.target.value)}
                    sx={{
                      mb: 2,
                      '& .MuiOutlinedInput-root': {
                        borderRadius: 2,
                        '&:hover fieldset': { borderColor: '#6366F1' },
                        '&.Mui-focused fieldset': { borderColor: '#4F46E5' },
                      },
                    }}
                  />
                  <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                    {sampleTexts.map((t, i) => (
                      <Chip key={i} label={`Sample ${i + 1}`} size="small" onClick={() => setText(t)}
                        sx={{ cursor: 'pointer', fontSize: 12, background: '#F8FAFC', '&:hover': { background: '#EEF2FF' } }} />
                    ))}
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1.5 }}>
                    <Button variant="contained" onClick={handleSingle}
                      disabled={loading || !text.trim()} endIcon={loading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <Send sx={{ fontSize: 16 }} />}
                      sx={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', px: 3 }}>
                      {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                    </Button>
                    <Button variant="outlined" onClick={() => { setText(''); setResult(null); setError(''); }}
                      startIcon={<DeleteOutline sx={{ fontSize: 16 }} />}
                      sx={{ borderColor: '#E2E8F0', color: '#64748B' }}>
                      Clear
                    </Button>
                  </Box>
                </>
              )}

              {tab === 1 && (
                <>
                  <Typography sx={{ fontSize: 13, color: '#64748B', mb: 1 }}>
                    Enter one text per line (up to 100 texts)
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={6}
                    placeholder={"I love this product!\nThis is terrible quality.\nNot sure about it yet."}
                    value={batchTexts}
                    onChange={e => setBatchTexts(e.target.value)}
                    sx={{
                      mb: 2,
                      '& .MuiOutlinedInput-root': { borderRadius: 2 },
                    }}
                  />
                  <Button variant="contained" onClick={handleBatch}
                    disabled={batchLoading || !batchTexts.trim()}
                    endIcon={batchLoading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <BarChart sx={{ fontSize: 16 }} />}
                    sx={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', px: 3 }}>
                    {batchLoading ? 'Processing...' : 'Analyze All'}
                  </Button>
                </>
              )}

              {error && (
                <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setError('')}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Right: Result */}
        <Grid item xs={12} md={5}>
          {tab === 0 && result && !result.error && (
            <Card sx={{ mb: 2.5 }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#94A3B8', letterSpacing: 0.5, mb: 2 }}>
                  ANALYSIS RESULT
                </Typography>
                <Box sx={{
                  textAlign: 'center', py: 2, mb: 2.5,
                  background: sentBg(result.sentiment), borderRadius: 2,
                }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Box sx={{ color: sentColor(result.sentiment), fontSize: 40 }}>
                      {result.sentiment === 'Positive' ? '😊' : result.sentiment === 'Negative' ? '😞' : '😐'}
                    </Box>
                  </Box>
                  <Chip
                    label={result.sentiment}
                    icon={sentIcon(result.sentiment)}
                    sx={{
                      background: sentColor(result.sentiment),
                      color: 'white',
                      fontWeight: 700,
                      fontSize: 15,
                      height: 34,
                      '& .MuiChip-icon': { color: 'white' },
                    }}
                  />
                </Box>
                <ConfidenceBar value={result.confidence || 0.8} color={sentColor(result.sentiment)} />
                <Divider sx={{ my: 2 }} />
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>METHOD</Typography>
                    <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B', textTransform: 'capitalize' }}>
                      {result.method?.replace('_', ' ') || '—'}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>PROCESSING TIME</Typography>
                    <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B' }}>
                      {result.processing_time ? `${(result.processing_time * 1000).toFixed(1)}ms` : '—'}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* History */}
          {history.length > 0 && tab === 0 && (
            <Card>
              <CardContent sx={{ p: 2.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B' }}>Analysis History</Typography>
                  <Button size="small" onClick={() => setHistory([])}
                    startIcon={<Refresh sx={{ fontSize: 14 }} />}
                    sx={{ color: '#94A3B8', fontSize: 12, p: 0, minWidth: 'auto' }}>
                    Clear
                  </Button>
                </Box>
                {history.slice(0, 5).map((h, i) => (
                  <Box key={i} sx={{ display: 'flex', gap: 1.5, alignItems: 'center', py: 1 }}>
                    <Chip
                      label={h.sentiment}
                      size="small"
                      sx={{ background: sentBg(h.sentiment), color: sentColor(h.sentiment), fontWeight: 600, fontSize: 11, flexShrink: 0 }}
                    />
                    <Typography sx={{ fontSize: 12, color: '#64748B', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {h.inputText}
                    </Typography>
                    <Typography sx={{ fontSize: 11, color: '#CBD5E1', flexShrink: 0 }}>
                      {h.confidence ? `${(h.confidence * 100).toFixed(0)}%` : ''}
                    </Typography>
                  </Box>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Batch Results */}
          {tab === 1 && batchResults.length > 0 && (
            <Card>
              <CardContent sx={{ p: 2.5 }}>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B', mb: 1.5 }}>Batch Results</Typography>
                {batchStats && (
                  <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                    <Chip label={`✅ Positive: ${batchStats.positive}`} size="small" sx={{ background: '#D1FAE5', color: '#065F46', fontWeight: 600, fontSize: 12 }} />
                    <Chip label={`❌ Negative: ${batchStats.negative}`} size="small" sx={{ background: '#FEE2E2', color: '#991B1B', fontWeight: 600, fontSize: 12 }} />
                    <Chip label={`❓ Neutral: ${batchStats.neutral}`} size="small" sx={{ background: '#FEF3C7', color: '#92400E', fontWeight: 600, fontSize: 12 }} />
                  </Box>
                )}
                <Box sx={{ maxHeight: 280, overflow: 'auto' }}>
                  {batchResults.map((r, i) => (
                    <Box key={i} sx={{ display: 'flex', gap: 1, alignItems: 'center', py: 0.8 }}>
                      <Typography sx={{ fontSize: 11, color: '#CBD5E1', minWidth: 20 }}>#{i + 1}</Typography>
                      <Chip label={r.sentiment} size="small"
                        sx={{ background: sentBg(r.sentiment), color: sentColor(r.sentiment), fontWeight: 600, fontSize: 11, flexShrink: 0 }} />
                      <Typography sx={{ fontSize: 12, color: '#64748B', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {r.text || r.inputText || '—'}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          )}

          {tab === 0 && !result && (
            <Card sx={{ background: '#F8FAFC', border: '1px dashed #E2E8F0', boxShadow: 'none' }}>
              <CardContent sx={{ p: 3, textAlign: 'center' }}>
                <Psychology sx={{ fontSize: 48, color: '#CBD5E1', mb: 1.5 }} />
                <Typography sx={{ fontSize: 14, color: '#94A3B8', mb: 0.5 }}>No analysis yet</Typography>
                <Typography sx={{ fontSize: 12, color: '#CBD5E1' }}>Enter text and click Analyze Sentiment</Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Model Info */}
      <Card sx={{ mt: 2.5, background: 'linear-gradient(135deg, #F8FAFF, #EEF2FF)' }}>
        <CardContent sx={{ p: 2.5 }}>
          <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B', mb: 1.5 }}>About the Model</Typography>
          <Grid container spacing={3}>
            {[
              { label: 'Training Data', value: 'IMDB Reviews + Twitter Sentiment' },
              { label: 'Feature Extraction', value: 'TF-IDF (5,000 features)' },
              { label: 'Preprocessing', value: 'HTML strip, lemmatization, stopwords' },
              { label: 'Model Accuracy', value: '~94.2% on test set' },
            ].map(({ label, value }) => (
              <Grid item xs={12} sm={6} md={3} key={label}>
                <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</Typography>
                <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#4F46E5' }}>{value}</Typography>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}
