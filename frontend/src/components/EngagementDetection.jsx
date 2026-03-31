import React, { useState } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, TextField, Button,
  CircularProgress, Alert, Chip, LinearProgress, Divider, Switch,
  FormControlLabel, Slider, InputAdornment
} from '@mui/material';
import {
  Security, CheckCircle, Warning, Error as ErrorIcon,
  Visibility, ThumbUp, Comment, Send, DeleteOutline, Info
} from '@mui/icons-material';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell
} from 'recharts';
import { apiCall, API_ENDPOINTS } from '../config/api';

const getAuthLevel = (score) => {
  if (score >= 80) return { label: 'Very High', color: '#10B981', bg: '#D1FAE5' };
  if (score >= 60) return { label: 'High', color: '#3B82F6', bg: '#DBEAFE' };
  if (score >= 40) return { label: 'Medium', color: '#F59E0B', bg: '#FEF3C7' };
  if (score >= 20) return { label: 'Low', color: '#F97316', bg: '#FFEDD5' };
  return { label: 'Very Low', color: '#EF4444', bg: '#FEE2E2' };
};

function GaugeChart({ score }) {
  const level = getAuthLevel(score);
  const rotation = (score / 100) * 180 - 90;
  return (
    <Box sx={{ textAlign: 'center', py: 2 }}>
      <Box sx={{ position: 'relative', display: 'inline-block', width: 180, height: 100 }}>
        {/* Track */}
        <svg width="180" height="100" viewBox="0 0 180 100">
          <path d="M 15 90 A 75 75 0 0 1 165 90" fill="none" stroke="#F1F5F9" strokeWidth="16" strokeLinecap="round" />
          <path d="M 15 90 A 75 75 0 0 1 165 90" fill="none" stroke={level.color}
            strokeWidth="16" strokeLinecap="round"
            strokeDasharray={`${(score / 100) * 235} 235`} opacity="0.8" />
          {/* Needle */}
          <g transform={`translate(90, 90) rotate(${rotation})`}>
            <line x1="0" y1="0" x2="0" y2="-60" stroke="#1E293B" strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="0" cy="0" r="5" fill="#1E293B" />
          </g>
        </svg>
        <Box sx={{ position: 'absolute', bottom: 0, left: '50%', transform: 'translateX(-50%)' }}>
          <Typography sx={{ fontSize: 24, fontWeight: 800, color: level.color, lineHeight: 1 }}>
            {score.toFixed(0)}
          </Typography>
          <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>/ 100</Typography>
        </Box>
      </Box>
      <Chip
        label={`${level.label} Authenticity`}
        sx={{ background: level.bg, color: level.color, fontWeight: 700, mt: 1 }}
        icon={score >= 60 ? <CheckCircle sx={{ fontSize: '16px !important' }} /> : <Warning sx={{ fontSize: '16px !important' }} />}
      />
    </Box>
  );
}

export default function EngagementDetection() {
  const [form, setForm] = useState({ views: '', likes: '', comments: '' });
  const [includeDetail, setIncludeDetail] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    const views = parseInt(form.views);
    const likes = parseInt(form.likes);
    const comments = parseInt(form.comments);
    if (!views || isNaN(views)) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await apiCall(API_ENDPOINTS.ENGAGEMENT_DETECT, {
        method: 'POST',
        body: JSON.stringify({
          metrics: { views, likes: likes || 0, comments: comments || 0 },
          include_detailed_analysis: includeDetail,
        }),
      });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const presets = [
    { label: 'Normal Content', views: 10000, likes: 500, comments: 80 },
    { label: 'Viral Video', views: 1000000, likes: 85000, comments: 4500 },
    { label: 'Suspicious (Bot)', views: 50000, likes: 48000, comments: 20 },
    { label: 'Low Engagement', views: 5000, likes: 30, comments: 5 },
  ];

  const radarData = result ? [
    { subject: 'Like Ratio', value: Math.min(100, ((parseInt(form.likes) / parseInt(form.views)) * 100) || 0) },
    { subject: 'Comment Rate', value: Math.min(100, ((parseInt(form.comments) / parseInt(form.views)) * 1000) || 0) },
    { subject: 'Authenticity', value: result.engagement_authenticity_score || 0 },
    { subject: 'Engagement', value: Math.min(100, (((parseInt(form.likes) + parseInt(form.comments)) / parseInt(form.views)) * 100) || 0) },
    { subject: 'Normal Score', value: result.is_fake_engagement ? 20 : 80 },
  ] : [];

  const level = result ? getAuthLevel(result.engagement_authenticity_score) : null;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{ width: 40, height: 40, borderRadius: 2, background: '#FEF2F2', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Security sx={{ color: '#EF4444', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1E293B' }}>Fake Engagement Detection</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8' }}>Isolation Forest anomaly detection to identify suspicious engagement patterns</Typography>
          </Box>
        </Box>
      </Box>

      <Grid container spacing={2.5}>
        {/* Input */}
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>
                Enter Engagement Metrics
              </Typography>

              {/* Presets */}
              <Box sx={{ mb: 2 }}>
                <Typography sx={{ fontSize: 12, color: '#94A3B8', mb: 1 }}>Quick Presets</Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {presets.map((p) => (
                    <Chip key={p.label} label={p.label} size="small"
                      onClick={() => setForm({ views: String(p.views), likes: String(p.likes), comments: String(p.comments) })}
                      sx={{ cursor: 'pointer', fontSize: 12, background: '#F8FAFC', '&:hover': { background: '#EEF2FF' } }} />
                  ))}
                </Box>
              </Box>

              <TextField
                fullWidth
                label="Views"
                type="number"
                value={form.views}
                onChange={e => setForm({ ...form, views: e.target.value })}
                InputProps={{ startAdornment: <InputAdornment position="start"><Visibility sx={{ fontSize: 18, color: '#94A3B8' }} /></InputAdornment> }}
                sx={{ mb: 2, '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
              />
              <TextField
                fullWidth
                label="Likes"
                type="number"
                value={form.likes}
                onChange={e => setForm({ ...form, likes: e.target.value })}
                InputProps={{ startAdornment: <InputAdornment position="start"><ThumbUp sx={{ fontSize: 18, color: '#94A3B8' }} /></InputAdornment> }}
                sx={{ mb: 2, '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
              />
              <TextField
                fullWidth
                label="Comments"
                type="number"
                value={form.comments}
                onChange={e => setForm({ ...form, comments: e.target.value })}
                InputProps={{ startAdornment: <InputAdornment position="start"><Comment sx={{ fontSize: 18, color: '#94A3B8' }} /></InputAdornment> }}
                sx={{ mb: 2, '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
              />

              <FormControlLabel
                control={<Switch checked={includeDetail} onChange={e => setIncludeDetail(e.target.checked)} size="small" />}
                label={<Typography sx={{ fontSize: 13, color: '#64748B' }}>Include detailed analysis</Typography>}
                sx={{ mb: 2 }}
              />

              <Box sx={{ display: 'flex', gap: 1.5 }}>
                <Button
                  variant="contained"
                  onClick={handleSubmit}
                  disabled={loading || !form.views}
                  endIcon={loading ? <CircularProgress size={14} sx={{ color: 'white' }} /> : <Send sx={{ fontSize: 16 }} />}
                  sx={{ background: 'linear-gradient(135deg, #EF4444, #DC2626)', px: 3 }}
                >
                  {loading ? 'Detecting...' : 'Detect'}
                </Button>
                <Button variant="outlined"
                  onClick={() => { setForm({ views: '', likes: '', comments: '' }); setResult(null); setError(''); }}
                  startIcon={<DeleteOutline sx={{ fontSize: 16 }} />}
                  sx={{ borderColor: '#E2E8F0', color: '#64748B' }}>
                  Clear
                </Button>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mt: 2, borderRadius: 2 }} onClose={() => setError('')}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Ratio Preview */}
          {form.views && (
            <Card sx={{ mt: 2 }}>
              <CardContent sx={{ p: 2 }}>
                <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B', mb: 1.5 }}>Live Ratios</Typography>
                {[
                  { label: 'Like Rate', value: form.views ? ((parseInt(form.likes || 0) / parseInt(form.views)) * 100).toFixed(2) : 0, max: 20 },
                  { label: 'Comment Rate', value: form.views ? ((parseInt(form.comments || 0) / parseInt(form.views)) * 100).toFixed(2) : 0, max: 5 },
                ].map(({ label, value, max }) => (
                  <Box key={label} sx={{ mb: 1.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography sx={{ fontSize: 12, color: '#64748B' }}>{label}</Typography>
                      <Typography sx={{ fontSize: 12, fontWeight: 700, color: parseFloat(value) > max ? '#EF4444' : '#10B981' }}>
                        {value}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(100, (parseFloat(value) / (max * 5)) * 100)}
                      sx={{
                        height: 5, borderRadius: 3, background: '#F1F5F9',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 3,
                          background: parseFloat(value) > max ? '#EF4444' : '#10B981',
                        },
                      }}
                    />
                  </Box>
                ))}
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Result */}
        <Grid item xs={12} md={7}>
          {result ? (
            <>
              <Card sx={{ mb: 2.5 }}>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#94A3B8', letterSpacing: 0.5, mb: 2 }}>
                    DETECTION RESULT
                  </Typography>
                  <GaugeChart score={result.engagement_authenticity_score} />

                  <Divider sx={{ my: 2 }} />

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>STATUS</Typography>
                      <Chip
                        label={result.is_fake_engagement ? 'FAKE ENGAGEMENT' : 'AUTHENTIC'}
                        icon={result.is_fake_engagement ? <ErrorIcon sx={{ fontSize: '14px !important' }} /> : <CheckCircle sx={{ fontSize: '14px !important' }} />}
                        sx={{
                          background: result.is_fake_engagement ? '#FEE2E2' : '#D1FAE5',
                          color: result.is_fake_engagement ? '#991B1B' : '#065F46',
                          fontWeight: 700,
                          '& .MuiChip-icon': { color: result.is_fake_engagement ? '#EF4444' : '#10B981' },
                        }}
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>SUSPICION LEVEL</Typography>
                      <Chip label={result.suspicion_level}
                        sx={{ background: '#FEF3C7', color: '#92400E', fontWeight: 600 }} />
                    </Grid>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>ANOMALY SCORE</Typography>
                      <Typography sx={{ fontSize: 16, fontWeight: 700, color: '#1E293B' }}>
                        {result.anomaly_score?.toFixed(4)}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>PROCESSING TIME</Typography>
                      <Typography sx={{ fontSize: 16, fontWeight: 700, color: '#1E293B' }}>
                        {result.processing_time ? `${(result.processing_time * 1000).toFixed(1)}ms` : '—'}
                      </Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>

              {/* Radar Chart */}
              <Card>
                <CardContent sx={{ p: 2.5 }}>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 1 }}>
                    Engagement Analysis Radar
                  </Typography>
                  <ResponsiveContainer width="100%" height={220}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#F1F5F9" />
                      <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12, fill: '#94A3B8' }} />
                      <Radar name="Score" dataKey="value" stroke={level?.color || '#4F46E5'}
                        fill={level?.color || '#4F46E5'} fillOpacity={0.25} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                  {result.detailed_analysis && (
                    <Box sx={{ mt: 1.5 }}>
                      <Typography sx={{ fontSize: 12, fontWeight: 600, color: '#64748B', mb: 1 }}>Detailed Features</Typography>
                      <Grid container spacing={1}>
                        {Object.entries(result.detailed_analysis).map(([k, v]) => (
                          <Grid item xs={6} key={k}>
                            <Box sx={{ background: '#F8FAFC', borderRadius: 1.5, p: 1 }}>
                              <Typography sx={{ fontSize: 11, color: '#94A3B8', textTransform: 'capitalize' }}>
                                {k.replace(/_/g, ' ')}
                              </Typography>
                              <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B' }}>
                                {typeof v === 'number' ? v.toFixed(4) : String(v)}
                              </Typography>
                            </Box>
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </>
          ) : (
            <Card sx={{ background: '#F8FAFC', border: '1px dashed #E2E8F0', boxShadow: 'none', height: '100%', minHeight: 300 }}>
              <CardContent sx={{ p: 3, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Security sx={{ fontSize: 60, color: '#CBD5E1', mb: 2 }} />
                <Typography sx={{ fontSize: 15, fontWeight: 600, color: '#94A3B8', mb: 1 }}>No Detection Yet</Typography>
                <Typography sx={{ fontSize: 13, color: '#CBD5E1', maxWidth: 300 }}>
                  Enter engagement metrics (views, likes, comments) and click Detect to analyze authenticity
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>

      {/* Detection Info */}
      <Card sx={{ mt: 2.5, background: 'linear-gradient(135deg, #FFF5F5, #FEF2F2)' }}>
        <CardContent sx={{ p: 2.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
            <Info sx={{ fontSize: 18, color: '#EF4444' }} />
            <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#1E293B' }}>Detection Algorithm Info</Typography>
          </Box>
          <Grid container spacing={3}>
            {[
              { label: 'Algorithm', value: 'Isolation Forest (Unsupervised)' },
              { label: 'Contamination Rate', value: '10% (expected anomaly rate)' },
              { label: 'Features Used', value: 'Likes/Views ratio, Comment rate, Engagement velocity' },
              { label: 'Detection Rules', value: 'Likes > 80% views, Low comments + high likes, ER > 0.5' },
            ].map(({ label, value }) => (
              <Grid item xs={12} sm={6} md={3} key={label}>
                <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</Typography>
                <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#EF4444' }}>{value}</Typography>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}
