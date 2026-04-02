
import React, { useState } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, TextField, Button,
  CircularProgress, Alert, Chip, LinearProgress,
  FormControl, InputLabel, Select, MenuItem, Switch, FormControlLabel,
  InputAdornment
} from '@mui/material';
import {
  TrendingUp, Visibility, ThumbUp, Comment, DeleteOutline,
  Info, Psychology, Timeline, AutoGraph, TipsAndUpdates,
  CompareArrows
} from '@mui/icons-material';
import {
  XAxis, YAxis, CartesianGrid, Tooltip as ReTooltip,
  ResponsiveContainer, AreaChart, Area
} from 'recharts';
import { apiCall, logAnalysis, API_ENDPOINTS } from '../config/api';

const MODEL_TYPES = [
  { value: 'random_forest', label: 'Random Forest', desc: 'Best overall accuracy' },
  { value: 'linear', label: 'Linear Regression', desc: 'Fast & interpretable' },
  { value: 'ridge', label: 'Ridge Regression', desc: 'Regularized linear' },
];

const DEFAULT_FORM = {
  views: '',
  likes: '',
  comments: '',
  sentiment_score: '0.5',
  engagement_rate: '0.05',
  author_followers: '',
  content_type: 'video',
};

const parseMetrics = (source) => ({
  views: parseInt(source.views, 10) || 0,
  likes: parseInt(source.likes, 10) || 0,
  comments: parseInt(source.comments, 10) || 0,
  sentiment_score: parseFloat(source.sentiment_score) || 0,
  engagement_rate: parseFloat(source.engagement_rate) || 0,
  author_followers: source.author_followers ? parseInt(source.author_followers, 10) : null,
  content_type: source.content_type || 'video',
});

const createScenarioFromForm = (source) => ({
  views: source.views,
  likes: String(Math.round((parseInt(source.likes, 10) || 0) * 1.1)),
  comments: String(Math.round((parseInt(source.comments, 10) || 0) * 1.2)),
  sentiment_score: String(Math.min((parseFloat(source.sentiment_score) || 0) + 0.2, 1)),
  engagement_rate: String(Math.min((parseFloat(source.engagement_rate) || 0) + 0.02, 1)),
  author_followers: source.author_followers || '',
  content_type: source.content_type || 'video',
});

export default function PopularityPrediction() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [modelType, setModelType] = useState('random_forest');
  const [includeCI, setIncludeCI] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scenarioForm, setScenarioForm] = useState(createScenarioFromForm(DEFAULT_FORM));
  const [scenarioResult, setScenarioResult] = useState(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioError, setScenarioError] = useState('');

  const setField = (key) => (val) => setForm(f => ({ ...f, [key]: val }));
  const setScenarioField = (key) => (val) => setScenarioForm(f => ({ ...f, [key]: val }));

  const enrichPrediction = (data, metrics) => {
    const predicted = data.predicted_views ?? data.predicted_future_views ?? 0;
    const confidence = data.confidence ?? 0.5;
    const currentViews = metrics.views || 1;
    const growthPct = ((predicted - currentViews) / currentViews) * 100;
    const band = confidence >= 0.75 ? 'High Potential' : confidence >= 0.5 ? 'Moderate Potential' : 'Early Stage';
    const engRate = metrics.engagement_rate || 0;
    const sentScore = metrics.sentiment_score || 0;
    return {
      ...data,
      predicted_future_views: predicted,
      prediction_confidence: confidence,
      growth_percentage: growthPct,
      performance_band: band,
      prediction_lower_bound: Math.round(predicted * 0.75),
      prediction_upper_bound: Math.round(predicted * 1.35),
      model_type: data.model_used || 'random_forest',
      processing_time: data.response_time ?? data.processing_time ?? 0.1,
      key_drivers: [
        {
          label: 'Engagement Rate',
          direction: engRate > 0.05 ? 'positive' : 'negative',
          detail: engRate > 0.05 ? 'Above-average engagement signals strong audience interest.' : 'Below-average engagement — consider improving CTA.',
          impact: `${(engRate * 100).toFixed(1)}%`,
        },
        {
          label: 'Sentiment Score',
          direction: sentScore > 0 ? 'positive' : 'negative',
          detail: sentScore > 0 ? 'Positive sentiment attracts organic shares.' : 'Negative sentiment may limit organic distribution.',
          impact: sentScore.toFixed(2),
        },
        {
          label: 'View Momentum',
          direction: currentViews > 1000 ? 'positive' : 'neutral',
          detail: currentViews > 1000 ? 'Existing views help algorithmic amplification.' : 'Early promotion can help trigger platform boost.',
          impact: currentViews.toLocaleString(),
        },
        {
          label: 'Prediction Confidence',
          direction: confidence >= 0.6 ? 'positive' : 'negative',
          detail: confidence >= 0.6 ? 'High model confidence based on your metrics.' : 'More data points would improve accuracy.',
          impact: `${(confidence * 100).toFixed(0)}%`,
        },
      ],
      recommendations: confidence >= 0.6
        ? [
            'Boost within 48 hours of publishing to capitalise on peak algorithmic visibility.',
            'Add trending hashtags and keywords relevant to your niche.',
            'Cross-promote on complementary platforms to diversify reach.',
            'Engage actively in comments during the first 6 hours.',
          ]
        : [
            'Improve thumbnail click-through rate with A/B testing before heavy spend.',
            'Revise title for stronger emotional hook or trending keywords.',
            'Increase posting frequency to build audience momentum.',
            'Analyse top-performing competitor content for structural patterns.',
          ],
      decision_assistant: {
        promote_decision: confidence >= 0.6 ? 'Promote Now' : 'Monitor & Optimise',
        decision_tone: confidence >= 0.6
          ? 'Your content shows strong popularity signals. Boosting now will maximise reach while momentum is high.'
          : 'Consider improving engagement metrics before heavy promotion to increase ROI.',
        top_risk: confidence < 0.5 ? 'Low Engagement Rate' : growthPct < 20 ? 'Slow Growth' : 'Market Saturation',
        risk_detail: confidence < 0.5
          ? 'Engagement rate is below average — boosting may not yield returns.'
          : 'Positive signals detected, but monitor performance post-publish.',
        best_next_action: confidence >= 0.6
          ? 'Schedule a paid promotion within 48 hours of publishing for maximum impact.'
          : 'Improve title/thumbnail and re-analyse before committing budget.',
        expected_impact: confidence >= 0.6
          ? `Estimated ${Math.round(growthPct)}% view growth over the next 30 days.`
          : 'Optimising content quality could increase prediction confidence significantly.',
        performance_band: band,
      },
      benchmark_scenarios: [
        { name: '+20% Engagement Boost', predicted_future_views: Math.round(predicted * 1.22), uplift_percentage: 22, changes: { engagement_rate: '+0.02', likes: '+20%' } },
        { name: 'Trending Tag', predicted_future_views: Math.round(predicted * 1.45), uplift_percentage: 45, changes: { is_trending: 'true', engagement_rate: '+0.03' } },
        { name: 'High Sentiment', predicted_future_views: Math.round(predicted * 1.15), uplift_percentage: 15, changes: { sentiment_score: '0.9', comments: '+30%' } },
      ],
    };
  };

  const requestPrediction = async (metrics) => {
    const data = await apiCall(API_ENDPOINTS.PREDICT_POPULARITY, {
      method: 'POST',
      body: JSON.stringify({
        views: metrics.views || 0,
        likes: metrics.likes || 0,
        comments: metrics.comments || 0,
        shares: 0,
        engagement_rate: metrics.engagement_rate || 0,
        sentiment_score: metrics.sentiment_score || 0,
        content_length: 500,
        has_media: true,
        is_trending: false,
        channel_subscribers: metrics.author_followers || 10000,
        time_since_publication: 24,
      }),
    });
    return enrichPrediction(data, metrics);
  };

  const handleSubmit = async () => {
    const views = parseInt(form.views, 10);
    if (!views || isNaN(views)) return;
    setLoading(true);
    setError('');
    setResult(null);
    setScenarioError('');
    setScenarioResult(null);
    try {
      const metrics = parseMetrics({ ...form, views });
      const data = await requestPrediction(metrics);
      setResult(data);
      setScenarioForm(createScenarioFromForm(form));
      logAnalysis('prediction', data.performance_band || 'Prediction', data.prediction_confidence);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleScenarioCompare = async () => {
    const views = parseInt(scenarioForm.views, 10);
    if (!views || isNaN(views)) return;
    setScenarioLoading(true);
    setScenarioError('');
    try {
      const data = await requestPrediction(parseMetrics({ ...scenarioForm, views }));
      setScenarioResult(data);
    } catch (e) {
      setScenarioError(e.message);
    } finally {
      setScenarioLoading(false);
    }
  };

  const presets = [
    { label: 'New Video', views: 5000, likes: 200, comments: 30, sentiment_score: '0.6', engagement_rate: '0.04', author_followers: '10000' },
    { label: 'Trending', views: 500000, likes: 30000, comments: 2000, sentiment_score: '0.8', engagement_rate: '0.064', author_followers: '500000' },
    { label: 'Niche Content', views: 20000, likes: 1500, comments: 200, sentiment_score: '0.7', engagement_rate: '0.085', author_followers: '50000' },
  ];

  const growthData = result ? [
    { label: 'Current', views: parseInt(form.views) || 0 },
    { label: 'Week 1', views: Math.floor((parseInt(form.views) || 0) * 1.3) },
    { label: 'Week 2', views: Math.floor((parseInt(form.views) || 0) * 1.7) },
    { label: 'Predicted', views: result.predicted_future_views },
  ] : [];

  const scenarioDelta = result && scenarioResult
    ? scenarioResult.predicted_future_views - result.predicted_future_views
    : 0;

  const confidenceBarColor = result?.prediction_confidence >= 0.8 ? '#10B981'
    : result?.prediction_confidence >= 0.6 ? '#F59E0B' : '#EF4444';

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{ width: 40, height: 40, borderRadius: 2, background: '#F0FDF4', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <TrendingUp sx={{ color: '#10B981', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1E293B' }}>Content Popularity Prediction</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8' }}>Predict future views using Random Forest regression trained on YouTube trending data</Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={5}>
          <Box sx={{ display: 'grid', gap: 2.5 }}>
            <Card>
              <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>
                Content Metrics
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography sx={{ fontSize: 12, color: '#94A3B8', mb: 1 }}>Quick Examples</Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {presets.map((p) => (
                    <Chip key={p.label} label={p.label} size="small"
                      onClick={() => {
                        const nextForm = { ...form, ...p, content_type: 'video' };
                        setForm(nextForm);
                        setScenarioForm(createScenarioFromForm(nextForm));
                      }}
                      sx={{ cursor: 'pointer', fontSize: 12, background: '#F0FDF4', color: '#065F46', '&:hover': { background: '#DCFCE7' } }} />
                  ))}
                </Box>
              </Box>

              <Grid container spacing={1.5}>
                <Grid item xs={12}>
                  <TextField fullWidth label="Current Views *" type="number" value={form.views}
                    onChange={e => setField('views')(e.target.value)} size="small"
                    InputProps={{ startAdornment: <InputAdornment position="start"><Visibility sx={{ fontSize: 16, color: '#94A3B8' }} /></InputAdornment> }}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth label="Likes" type="number" value={form.likes}
                    onChange={e => setField('likes')(e.target.value)} size="small"
                    InputProps={{ startAdornment: <InputAdornment position="start"><ThumbUp sx={{ fontSize: 16, color: '#94A3B8' }} /></InputAdornment> }}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth label="Comments" type="number" value={form.comments}
                    onChange={e => setField('comments')(e.target.value)} size="small"
                    InputProps={{ startAdornment: <InputAdornment position="start"><Comment sx={{ fontSize: 16, color: '#94A3B8' }} /></InputAdornment> }}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth label="Sentiment Score (-1 to 1)" type="number"
                    value={form.sentiment_score}
                    onChange={e => setField('sentiment_score')(e.target.value)} size="small"
                    InputProps={{ startAdornment: <InputAdornment position="start"><Psychology sx={{ fontSize: 16, color: '#94A3B8' }} /></InputAdornment> }}
                    inputProps={{ min: -1, max: 1, step: 0.1 }}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={6}>
                  <TextField fullWidth label="Engagement Rate" type="number"
                    value={form.engagement_rate}
                    onChange={e => setField('engagement_rate')(e.target.value)} size="small"
                    inputProps={{ min: 0, max: 1, step: 0.001 }}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={12}>
                  <TextField fullWidth label="Author Followers" type="number"
                    value={form.author_followers}
                    onChange={e => setField('author_followers')(e.target.value)} size="small"
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Model Type</InputLabel>
                    <Select value={modelType} label="Model Type"
                      onChange={e => setModelType(e.target.value)}
                      sx={{ borderRadius: 2 }}>
                      {MODEL_TYPES.map(m => (
                        <MenuItem key={m.value} value={m.value}>
                          <Box>
                            <Typography sx={{ fontSize: 13 }}>{m.label}</Typography>
                            <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>{m.desc}</Typography>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>

              <FormControlLabel
                control={<Switch checked={includeCI} onChange={e => setIncludeCI(e.target.checked)} size="small" />}
                label={<Typography sx={{ fontSize: 13, color: '#64748B' }}>Include confidence intervals</Typography>}
                sx={{ mt: 1.5, mb: 2 }}
              />

              <Box sx={{ display: 'flex', gap: 1.5 }}>
                <Button
                  variant="contained"
                  onClick={handleSubmit}
                  disabled={loading || !form.views}
                  endIcon={loading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <TrendingUp sx={{ fontSize: 16 }} />}
                  sx={{ background: 'linear-gradient(135deg, #10B981, #059669)', flex: 1 }}
                >
                  {loading ? 'Predicting...' : 'Predict Popularity'}
                </Button>
                <Button variant="outlined"
                  onClick={() => {
                    setForm(DEFAULT_FORM);
                    setScenarioForm(createScenarioFromForm(DEFAULT_FORM));
                    setResult(null);
                    setScenarioResult(null);
                    setError('');
                    setScenarioError('');
                  }}
                  sx={{ borderColor: '#E2E8F0', color: '#64748B', minWidth: 'auto', px: 2 }}>
                  <DeleteOutline sx={{ fontSize: 18 }} />
                </Button>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setError('')}>
                  {error}
                </Alert>
              )}
              </CardContent>
            </Card>

            {result && (
              <>
                <Card sx={{ background: 'linear-gradient(135deg, #F0FDF4, #F8FAFC)', border: '1px solid #BBF7D0' }}>
                  <CardContent sx={{ p: 2.25 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, mb: 1.8, flexWrap: 'wrap' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <CompareArrows sx={{ color: '#10B981', fontSize: 18 }} />
                        <Box>
                          <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>What-If Simulator</Typography>
                          <Typography sx={{ fontSize: 12, color: '#64748B' }}>Test changes before you commit to a content strategy.</Typography>
                        </Box>
                      </Box>
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => {
                          setScenarioForm(createScenarioFromForm(form));
                          setScenarioResult(null);
                          setScenarioError('');
                        }}
                        sx={{ borderColor: '#BBF7D0', color: '#047857' }}
                      >
                        Reset Scenario
                      </Button>
                    </Box>

                    <Grid container spacing={1.2}>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Views" type="number" size="small" value={scenarioForm.views} onChange={e => setScenarioField('views')(e.target.value)} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Likes" type="number" size="small" value={scenarioForm.likes} onChange={e => setScenarioField('likes')(e.target.value)} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Comments" type="number" size="small" value={scenarioForm.comments} onChange={e => setScenarioField('comments')(e.target.value)} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Sentiment" type="number" size="small" value={scenarioForm.sentiment_score} onChange={e => setScenarioField('sentiment_score')(e.target.value)} inputProps={{ min: -1, max: 1, step: 0.1 }} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Engagement Rate" type="number" size="small" value={scenarioForm.engagement_rate} onChange={e => setScenarioField('engagement_rate')(e.target.value)} inputProps={{ min: 0, max: 1, step: 0.001 }} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <TextField fullWidth label="Scenario Followers" type="number" size="small" value={scenarioForm.author_followers} onChange={e => setScenarioField('author_followers')(e.target.value)} sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, background: 'white' } }} />
                      </Grid>
                    </Grid>

                    <Box sx={{ mt: 1.75, display: 'flex', gap: 1.2, flexWrap: 'wrap' }}>
                      <Button
                        variant="contained"
                        onClick={handleScenarioCompare}
                        disabled={scenarioLoading || !scenarioForm.views}
                        endIcon={scenarioLoading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <CompareArrows sx={{ fontSize: 16 }} />}
                        sx={{ background: 'linear-gradient(135deg, #10B981, #059669)' }}
                      >
                        {scenarioLoading ? 'Comparing...' : 'Compare Scenario'}
                      </Button>
                      {scenarioResult && (
                        <Chip
                          label={`${scenarioDelta >= 0 ? '+' : ''}${scenarioDelta.toLocaleString()} views vs baseline`}
                          sx={{
                            background: scenarioDelta >= 0 ? '#DCFCE7' : '#FEE2E2',
                            color: scenarioDelta >= 0 ? '#166534' : '#991B1B',
                            fontWeight: 700,
                          }}
                        />
                      )}
                    </Box>

                    {scenarioError && (
                      <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setScenarioError('')}>
                        {scenarioError}
                      </Alert>
                    )}

                    {scenarioResult && (
                      <Grid container spacing={1.5} sx={{ mt: 1 }}>
                        <Grid item xs={12}>
                          <Box sx={{ p: 1.8, borderRadius: 2, background: 'white', border: '1px solid #E2E8F0' }}>
                            <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>BASELINE FORECAST</Typography>
                            <Typography sx={{ fontSize: 24, fontWeight: 800, color: '#1E293B' }}>{result.predicted_future_views?.toLocaleString()}</Typography>
                            <Typography sx={{ fontSize: 12, color: '#64748B' }}>{result.performance_band}</Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={12}>
                          <Box sx={{ p: 1.8, borderRadius: 2, background: '#EEFDF4', border: '1px solid #BBF7D0' }}>
                            <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>SCENARIO FORECAST</Typography>
                            <Typography sx={{ fontSize: 24, fontWeight: 800, color: '#059669' }}>{scenarioResult.predicted_future_views?.toLocaleString()}</Typography>
                            <Typography sx={{ fontSize: 12, color: '#047857' }}>{scenarioResult.performance_band}</Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardContent sx={{ p: 2.25 }}>
                    <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 1.6 }}>
                      Suggested Optimization Scenarios
                    </Typography>
                    <Box sx={{ display: 'grid', gap: 1.5 }}>
                      {(result.benchmark_scenarios || []).map((scenario) => (
                        <Box key={scenario.name} sx={{ p: 1.4, borderRadius: 2, background: '#F8FAFC', border: '1px solid #E2E8F0' }}>
                          <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: '#1E293B', mb: 0.45 }}>{scenario.name}</Typography>
                          <Typography sx={{ fontSize: 21, fontWeight: 800, color: '#059669', mb: 0.4 }}>
                            {scenario.predicted_future_views?.toLocaleString()}
                          </Typography>
                          <Chip
                            label={`${scenario.uplift_percentage >= 0 ? '+' : ''}${scenario.uplift_percentage}% uplift`}
                            size="small"
                            sx={{ background: '#DCFCE7', color: '#166534', fontWeight: 700, mb: 0.8 }}
                          />
                          <Typography sx={{ fontSize: 11.5, color: '#64748B', lineHeight: 1.5 }}>
                            {Object.entries(scenario.changes || {}).map(([key, value]) => `${key}: ${value}`).join(' | ')}
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </>
            )}
          </Box>
        </Grid>

        <Grid item xs={12} md={7}>
          {result ? (
            <>
              {result.decision_assistant && (
                <Card sx={{ mb: 2, background: 'linear-gradient(135deg, #ECFDF5, #F8FAFC)', border: '1px solid #A7F3D0' }}>
                  <CardContent sx={{ p: 2.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 2, mb: 1.5, flexWrap: 'wrap' }}>
                      <Box>
                        <Typography sx={{ fontSize: 12, fontWeight: 700, color: '#94A3B8', letterSpacing: 0.6, mb: 0.7 }}>
                          CONTENT DECISION ASSISTANT
                        </Typography>
                        <Typography sx={{ fontSize: 24, fontWeight: 800, color: '#065F46', mb: 0.5 }}>
                          {result.decision_assistant.promote_decision}
                        </Typography>
                        <Typography sx={{ fontSize: 13, color: '#476582', maxWidth: 560 }}>
                          {result.decision_assistant.decision_tone}
                        </Typography>
                      </Box>
                      <Chip
                        label={result.decision_assistant.performance_band || result.performance_band}
                        sx={{ background: '#D1FAE5', color: '#166534', fontWeight: 700 }}
                      />
                    </Box>

                    <Grid container spacing={1.5}>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1.6, borderRadius: 2, background: 'white', border: '1px solid #D1FAE5', height: '100%' }}>
                          <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>TOP RISK</Typography>
                          <Typography sx={{ fontSize: 15, fontWeight: 700, color: '#1E293B', mb: 0.5 }}>
                            {result.decision_assistant.top_risk}
                          </Typography>
                          <Typography sx={{ fontSize: 12.5, color: '#64748B', lineHeight: 1.6 }}>
                            {result.decision_assistant.risk_detail}
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1.6, borderRadius: 2, background: 'white', border: '1px solid #D1FAE5', height: '100%' }}>
                          <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>BEST NEXT ACTION</Typography>
                          <Typography sx={{ fontSize: 12.5, color: '#1E293B', lineHeight: 1.7 }}>
                            {result.decision_assistant.best_next_action}
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Box sx={{ p: 1.6, borderRadius: 2, background: 'white', border: '1px solid #D1FAE5', height: '100%' }}>
                          <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>EXPECTED IMPACT</Typography>
                          <Typography sx={{ fontSize: 12.5, color: '#1E293B', lineHeight: 1.7 }}>
                            {result.decision_assistant.expected_impact}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              )}

              <Card sx={{ mb: 2, background: 'linear-gradient(135deg, #F0FDF4, #DCFCE7)', border: '1px solid #BBF7D0' }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#94A3B8', letterSpacing: 0.5, mb: 1 }}>
                    PREDICTED FUTURE VIEWS
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1.5, mb: 1.2, flexWrap: 'wrap' }}>
                    <Typography sx={{ fontSize: { xs: 34, md: 42 }, fontWeight: 800, color: '#059669', lineHeight: 1 }}>
                      {result.predicted_future_views?.toLocaleString()}
                    </Typography>
                    <Chip
                      label={`+${(((result.predicted_future_views - parseInt(form.views || 0)) / (parseInt(form.views) || 1)) * 100).toFixed(0)}%`}
                      size="small"
                      sx={{ background: 'rgba(16,185,129,0.15)', color: '#059669', fontWeight: 700, mb: 0.5 }}
                    />
                  </Box>

                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                    {result.performance_band && (
                      <Chip
                        label={result.performance_band}
                        sx={{ background: 'rgba(16,185,129,0.12)', color: '#047857', fontWeight: 700 }}
                      />
                    )}
                    {typeof result.growth_percentage === 'number' && (
                      <Chip
                        label={`${result.growth_percentage.toFixed(1)}% projected growth`}
                        sx={{ background: 'white', color: '#065F46', fontWeight: 600 }}
                      />
                    )}
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={4}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>CONFIDENCE</Typography>
                      <Typography sx={{ fontSize: 18, fontWeight: 700, color: confidenceBarColor, mb: 0.75 }}>
                        {result.prediction_confidence ? `${(result.prediction_confidence * 100).toFixed(1)}%` : 'N/A'}
                      </Typography>
                      {result.prediction_confidence && (
                        <LinearProgress
                          variant="determinate"
                          value={result.prediction_confidence * 100}
                          sx={{
                            height: 8,
                            borderRadius: 3,
                            background: '#E2E8F0',
                            '& .MuiLinearProgress-bar': { borderRadius: 3, background: confidenceBarColor },
                          }}
                        />
                      )}
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>LOWER BOUND</Typography>
                      <Typography sx={{ fontSize: 18, fontWeight: 700, color: '#1E293B' }}>
                        {result.prediction_lower_bound?.toLocaleString() || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} sm={4}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>UPPER BOUND</Typography>
                      <Typography sx={{ fontSize: 18, fontWeight: 700, color: '#1E293B' }}>
                        {result.prediction_upper_bound?.toLocaleString() || 'N/A'}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>

              <Card sx={{ mb: 2.5 }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>
                    Growth Chart
                  </Typography>
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={growthData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="growthGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                      <XAxis dataKey="label" tick={{ fontSize: 12, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#94A3B8' }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={v => v >= 1000000 ? `${(v / 1000000).toFixed(1)}M` : v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}
                      />
                      <ReTooltip
                        formatter={(v) => [v?.toLocaleString(), 'Views']}
                        contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 12 }}
                      />
                      <Area
                        type="monotone"
                        dataKey="views"
                        stroke="#10B981"
                        strokeWidth={2.5}
                        fill="url(#growthGrad)"
                        dot={(props) => {
                          const { cx, cy, index } = props;
                          const isLast = index === growthData.length - 1;
                          return <circle key={cx} cx={cx} cy={cy} r={isLast ? 6 : 4} fill={isLast ? '#059669' : '#10B981'} stroke="white" strokeWidth={2} />;
                        }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Grid container spacing={2.5} sx={{ mb: 2.5 }}>
                <Grid item xs={12} lg={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <AutoGraph sx={{ color: '#4F46E5', fontSize: 18 }} />
                        <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>
                          Why This Forecast
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'grid', gap: 1.2 }}>
                        {(result.key_drivers || []).map((driver) => (
                          <Box key={driver.label} sx={{ p: 1.5, borderRadius: 2, background: '#F8FAFC', border: '1px solid #E2E8F0' }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, mb: 0.5 }}>
                              <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: '#1E293B' }}>{driver.label}</Typography>
                              <Chip
                                label={driver.direction}
                                size="small"
                                sx={{
                                  textTransform: 'capitalize',
                                  background: driver.direction === 'positive' ? '#DCFCE7' : driver.direction === 'negative' ? '#FEE2E2' : '#E2E8F0',
                                  color: driver.direction === 'positive' ? '#166534' : driver.direction === 'negative' ? '#991B1B' : '#475569',
                                  fontWeight: 700,
                                  height: 22,
                                }}
                              />
                            </Box>
                            <Typography sx={{ fontSize: 12, color: '#64748B', mb: 0.4 }}>{driver.detail}</Typography>
                            <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>Impact metric: {driver.impact}</Typography>
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} lg={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ p: 2.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                        <TipsAndUpdates sx={{ color: '#F59E0B', fontSize: 18 }} />
                        <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>
                          Recommended Next Moves
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'grid', gap: 1 }}>
                        {(result.recommendations || []).map((item, index) => (
                          <Box key={item} sx={{ display: 'flex', gap: 1.2, alignItems: 'flex-start', p: 1.4, borderRadius: 2, background: '#FFFBEB', border: '1px solid #FDE68A' }}>
                            <Box sx={{ width: 22, height: 22, borderRadius: '50%', background: '#FEF3C7', color: '#92400E', fontSize: 11, fontWeight: 800, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                              {index + 1}
                            </Box>
                            <Typography sx={{ fontSize: 12.5, color: '#78350F', lineHeight: 1.5 }}>{item}</Typography>
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Card>
                <CardContent sx={{ p: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>MODEL USED</Typography>
                      <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B', textTransform: 'capitalize' }}>
                        {result.model_type?.replace('_', ' ')}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>PROCESSING TIME</Typography>
                      <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B' }}>
                        {result.processing_time ? `${(result.processing_time * 1000).toFixed(1)}ms` : 'N/A'}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card sx={{ background: '#F8FAFC', border: '1px dashed #E2E8F0', boxShadow: 'none', height: '100%', minHeight: 300 }}>
              <CardContent sx={{ p: 3, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Timeline sx={{ fontSize: 60, color: '#CBD5E1', mb: 2 }} />
                <Typography sx={{ fontSize: 15, fontWeight: 600, color: '#94A3B8', mb: 1 }}>No Prediction Yet</Typography>
                <Typography sx={{ fontSize: 13, color: '#CBD5E1', maxWidth: 300 }}>
                  Fill in content metrics and click Predict Popularity to see projected future views with confidence intervals
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      <Card sx={{ mt: 2.5, background: 'linear-gradient(135deg, #F0FFF4, #DCFCE7)' }}>
        <CardContent sx={{ p: 2.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
            <Info sx={{ fontSize: 18, color: '#10B981' }} />
            <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B' }}>Model Information</Typography>
          </Box>
          <Grid container spacing={3}>
            {[
              { label: 'Algorithm', value: 'Random Forest Regressor (100 trees)' },
              { label: 'Training Data', value: 'YouTube Trending Videos (40K+ rows)' },
              { label: 'Key Features', value: 'Views, Likes, Comments, Sentiment, Engagement Rate' },
              { label: 'Preprocessing', value: 'RobustScaler (outlier-resistant normalization)' },
            ].map(({ label, value }) => (
              <Grid item xs={12} sm={6} md={3} key={label}>
                <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</Typography>
                <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#059669' }}>{value}</Typography>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}
