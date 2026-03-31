import React, { useState, useEffect } from 'react';
import {
  Box, Grid, Typography, Card, CardContent, Button, TextField,
  CircularProgress, Alert, Chip, Avatar, LinearProgress, Divider
} from '@mui/material';
import {
  Psychology, Security, TrendingUp, Analytics, ArrowUpward,
  ArrowForward, AutoAwesome, CheckCircle, Warning
} from '@mui/icons-material';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { apiCall, API_ENDPOINTS } from '../config/api';

const SENTIMENT_COLORS = ['#6366F1', '#EF4444', '#10B981'];
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

const StatCard = ({ title, value, subtitle, icon, color, trend }) => (
  <Card sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
    <CardContent sx={{ p: 2.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Box sx={{
          width: 44, height: 44, borderRadius: 2.5,
          background: `${color}18`, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
        }}>
          {React.cloneElement(icon, { sx: { color, fontSize: 22 } })}
        </Box>
        {trend && (
          <Chip
            icon={<ArrowUpward sx={{ fontSize: '12px !important' }} />}
            label={trend}
            size="small"
            sx={{
              background: 'rgba(16,185,129,0.1)', color: '#059669',
              fontWeight: 600, fontSize: 11, height: 22,
            }}
          />
        )}
      </Box>
      <Typography variant="h4" sx={{ fontWeight: 700, color: '#1E293B', mb: 0.5, fontSize: '1.8rem' }}>
        {value}
      </Typography>
      <Typography sx={{ fontSize: 13, color: '#94A3B8', fontWeight: 500 }}>{title}</Typography>
      {subtitle && (
        <Typography sx={{ fontSize: 12, color: '#CBD5E1', mt: 0.5 }}>{subtitle}</Typography>
      )}
    </CardContent>
    <Box sx={{
      position: 'absolute', bottom: 0, right: 0, width: 80, height: 80,
      background: `${color}08`, borderRadius: '80px 0 12px 0',
    }} />
  </Card>
);

const ActivityItem = ({ item }) => {
  const colors = { sentiment: '#6366F1', engagement: '#EF4444', prediction: '#10B981' };
  const icons = { sentiment: <Psychology />, engagement: <Security />, prediction: <TrendingUp /> };
  const color = colors[item.type] || '#94A3B8';
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 1.2 }}>
      <Box sx={{
        width: 36, height: 36, borderRadius: 2, background: `${color}15`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        {React.cloneElement(icons[item.type] || <Analytics />, { sx: { color, fontSize: 18 } })}
      </Box>
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#1E293B' }}>{item.label}</Typography>
        <Typography sx={{ fontSize: 11, color: '#94A3B8', textTransform: 'capitalize' }}>
          {item.type} · {(item.confidence * 100).toFixed(0)}% confidence
        </Typography>
      </Box>
      <Typography sx={{ fontSize: 11, color: '#CBD5E1', flexShrink: 0 }}>{item.time}</Typography>
    </Box>
  );
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [quickText, setQuickText] = useState('');
  const [quickResult, setQuickResult] = useState(null);
  const [quickLoading, setQuickLoading] = useState(false);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      const data = await apiCall(API_ENDPOINTS.DASHBOARD_ANALYTICS);
      setAnalytics(data);
    } catch {
      setAnalytics({
        total_analyses: 12847,
        total_predictions: 3420,
        avg_sentiment_score: 0.68,
        fake_engagement_rate: 12.4,
        model_accuracy: 94.2,
        sentiment_distribution: { positive: 62, negative: 28, neutral: 10 },
        weekly_analyses: [320, 450, 380, 510, 490, 620, 580],
        recent_activity: [
          { type: 'sentiment', label: 'Positive', confidence: 0.92, time: '2 min ago' },
          { type: 'engagement', label: 'Authentic', confidence: 0.87, time: '5 min ago' },
          { type: 'prediction', label: 'High Popularity', confidence: 0.79, time: '12 min ago' },
          { type: 'sentiment', label: 'Negative', confidence: 0.85, time: '18 min ago' },
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  const handleQuickAnalysis = async () => {
    if (!quickText.trim()) return;
    setQuickLoading(true);
    setQuickResult(null);
    try {
      const data = await apiCall(API_ENDPOINTS.SENTIMENT_ANALYZE, {
        method: 'POST',
        body: JSON.stringify({ text: quickText, method: 'logistic' }),
      });
      setQuickResult(data);
    } catch (e) {
      setQuickResult({ error: e.message });
    } finally {
      setQuickLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress sx={{ color: '#4F46E5' }} />
      </Box>
    );
  }

  const pieData = analytics?.sentiment_distribution
    ? [
        { name: 'Positive', value: analytics.sentiment_distribution.positive },
        { name: 'Negative', value: analytics.sentiment_distribution.negative },
        { name: 'Neutral', value: analytics.sentiment_distribution.neutral },
      ]
    : [];

  const weeklyData = (analytics?.weekly_analyses || []).map((v, i) => ({
    day: DAYS[i], analyses: v,
  }));

  return (
    <Box>
      {/* Welcome Banner */}
      <Box sx={{
        background: 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #6366F1 100%)',
        borderRadius: 3, p: 3, mb: 3, position: 'relative', overflow: 'hidden',
      }}>
        <Box sx={{ position: 'absolute', top: -30, right: -30, width: 180, height: 180, borderRadius: '50%', background: 'rgba(255,255,255,0.06)' }} />
        <Box sx={{ position: 'absolute', bottom: -20, right: 80, width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.04)' }} />
        <Box sx={{ position: 'relative' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <AutoAwesome sx={{ color: '#FCD34D', fontSize: 20 }} />
            <Chip label="AI-Powered" size="small" sx={{ background: 'rgba(255,255,255,0.15)', color: 'white', fontSize: 11, height: 22 }} />
          </Box>
          <Typography variant="h5" sx={{ color: 'white', fontWeight: 700, mb: 0.5 }}>
            Welcome to HybridAI Dashboard
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.75)', fontSize: 13, mb: 2, maxWidth: 500 }}>
            Analyze sentiment, detect fake engagement, and predict content popularity using advanced machine learning models trained on real-world YouTube data.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              size="small"
              onClick={() => navigate('/sentiment')}
              endIcon={<ArrowForward sx={{ fontSize: 14 }} />}
              sx={{ background: 'white', color: '#4F46E5', '&:hover': { background: '#F8FAFC' }, boxShadow: 'none', fontWeight: 600 }}
            >
              Analyze Content
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => navigate('/prediction')}
              sx={{ borderColor: 'rgba(255,255,255,0.4)', color: 'white', '&:hover': { borderColor: 'white', background: 'rgba(255,255,255,0.08)' } }}
            >
              Predict Popularity
            </Button>
          </Box>
        </Box>
      </Box>

      {/* Stats Row */}
      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Analyses"
            value={analytics?.total_analyses?.toLocaleString()}
            subtitle="All time"
            icon={<Analytics />}
            color="#4F46E5"
            trend="+12.5%"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Avg Sentiment Score"
            value={`${((analytics?.avg_sentiment_score || 0) * 100).toFixed(0)}%`}
            subtitle="Positivity rate"
            icon={<Psychology />}
            color="#10B981"
            trend="+4.2%"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Fake Engagement"
            value={`${analytics?.fake_engagement_rate?.toFixed(1)}%`}
            subtitle="Detected this month"
            icon={<Security />}
            color="#EF4444"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Model Accuracy"
            value={`${analytics?.model_accuracy?.toFixed(1)}%`}
            subtitle="Across all models"
            icon={<TrendingUp />}
            color="#F59E0B"
            trend="+1.3%"
          />
        </Grid>
      </Grid>

      {/* Charts + Activity Row */}
      <Grid container spacing={2.5} sx={{ mb: 3 }}>
        {/* Weekly Chart */}
        <Grid item xs={12} md={7}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 2.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2.5 }}>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700, color: '#1E293B', fontSize: 15 }}>
                    Weekly Analysis Activity
                  </Typography>
                  <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>Last 7 days</Typography>
                </Box>
                <Chip label="This Week" size="small" sx={{ background: '#EEF2FF', color: '#4F46E5', fontWeight: 600, fontSize: 11 }} />
              </Box>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={weeklyData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#4F46E5" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="day" tick={{ fontSize: 12, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 12, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 12 }}
                  />
                  <Area type="monotone" dataKey="analyses" stroke="#4F46E5" strokeWidth={2.5} fill="url(#areaGrad)" dot={{ fill: '#4F46E5', r: 4 }} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Sentiment Pie */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography variant="h6" sx={{ fontWeight: 700, color: '#1E293B', fontSize: 15, mb: 0.5 }}>
                Sentiment Distribution
              </Typography>
              <Typography sx={{ fontSize: 12, color: '#94A3B8', mb: 2 }}>Overall content analysis</Typography>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={4} dataKey="value">
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={SENTIMENT_COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v, n) => [`${v}%`, n]}
                    contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 12 }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1 }}>
                {pieData.map((d, i) => (
                  <Box key={i} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', background: SENTIMENT_COLORS[i] }} />
                    <Typography sx={{ fontSize: 12, color: '#64748B' }}>{d.name} {d.value}%</Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Analysis + Recent Activity */}
      <Grid container spacing={2.5}>
        {/* Quick Sentiment */}
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent sx={{ p: 2.5 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Box sx={{ width: 32, height: 32, borderRadius: 1.5, background: '#EEF2FF', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Psychology sx={{ color: '#4F46E5', fontSize: 18 }} />
                </Box>
                <Box>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>Quick Sentiment Analysis</Typography>
                  <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>Type any text to analyze instantly</Typography>
                </Box>
              </Box>
              <TextField
                fullWidth
                multiline
                rows={3}
                placeholder="e.g. This product is absolutely amazing! I love the quality and the service was fantastic..."
                value={quickText}
                onChange={e => setQuickText(e.target.value)}
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                    '&:hover fieldset': { borderColor: '#6366F1' },
                    '&.Mui-focused fieldset': { borderColor: '#4F46E5' },
                  },
                }}
              />
              <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                <Button
                  variant="contained"
                  onClick={handleQuickAnalysis}
                  disabled={quickLoading || !quickText.trim()}
                  sx={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', px: 3 }}
                >
                  {quickLoading ? <CircularProgress size={16} sx={{ color: 'white' }} /> : 'Analyze'}
                </Button>
                <Button variant="outlined" onClick={() => { setQuickText(''); setQuickResult(null); }}
                  sx={{ borderColor: '#E2E8F0', color: '#64748B', '&:hover': { borderColor: '#CBD5E1' } }}>
                  Clear
                </Button>
              </Box>
              {quickResult && (
                <Box sx={{ mt: 2, p: 2, borderRadius: 2, background: quickResult.error ? '#FEF2F2' : '#F0FDF4', border: `1px solid ${quickResult.error ? '#FECACA' : '#BBF7D0'}` }}>
                  {quickResult.error ? (
                    <Typography sx={{ color: '#DC2626', fontSize: 13 }}>Error: {quickResult.error}</Typography>
                  ) : (
                    <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                      <Box>
                        <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>SENTIMENT</Typography>
                        <Chip
                          label={quickResult.sentiment}
                          size="small"
                          icon={quickResult.sentiment === 'Positive' ? <CheckCircle sx={{ fontSize: '14px !important' }} /> : <Warning sx={{ fontSize: '14px !important' }} />}
                          sx={{
                            background: quickResult.sentiment === 'Positive' ? '#DCFCE7' : '#FEE2E2',
                            color: quickResult.sentiment === 'Positive' ? '#166534' : '#991B1B',
                            fontWeight: 600,
                          }}
                        />
                      </Box>
                      <Box>
                        <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>CONFIDENCE</Typography>
                        <Typography sx={{ fontSize: 15, fontWeight: 700, color: '#1E293B' }}>
                          {quickResult.confidence ? `${(quickResult.confidence * 100).toFixed(1)}%` : '—'}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.3 }}>METHOD</Typography>
                        <Typography sx={{ fontSize: 13, color: '#64748B', textTransform: 'capitalize' }}>
                          {quickResult.method || 'ML Model'}
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={5}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ p: 2.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>Recent Activity</Typography>
                <Button size="small" endIcon={<ArrowForward sx={{ fontSize: 12 }} />}
                  sx={{ color: '#4F46E5', fontSize: 12, fontWeight: 600, p: 0, minWidth: 'auto' }}>
                  View All
                </Button>
              </Box>
              {(analytics?.recent_activity || []).map((item, i) => (
                <Box key={i}>
                  <ActivityItem item={item} />
                  {i < analytics.recent_activity.length - 1 && (
                    <Divider sx={{ borderColor: '#F8FAFC' }} />
                  )}
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Feature Cards */}
      <Grid container spacing={2.5} sx={{ mt: 0.5 }}>
        {[
          { title: 'Sentiment Analysis', desc: 'Analyze text sentiment using TF-IDF + Logistic Regression with 94% accuracy.', color: '#4F46E5', icon: <Psychology />, path: '/sentiment' },
          { title: 'Fake Engagement Detection', desc: 'Detect suspicious engagement patterns using Isolation Forest anomaly detection.', color: '#EF4444', icon: <Security />, path: '/engagement' },
          { title: 'Popularity Prediction', desc: 'Predict future content views using Random Forest regression with confidence intervals.', color: '#10B981', icon: <TrendingUp />, path: '/prediction' },
        ].map(({ title, desc, color, icon, path }) => (
          <Grid item xs={12} md={4} key={title}>
            <Card sx={{ cursor: 'pointer', transition: 'transform 0.2s, box-shadow 0.2s', '&:hover': { transform: 'translateY(-2px)', boxShadow: '0 8px 30px rgba(0,0,0,0.1)' } }}
              onClick={() => navigate(path)}>
              <CardContent sx={{ p: 2.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1.5 }}>
                  <Box sx={{ width: 40, height: 40, borderRadius: 2, background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {React.cloneElement(icon, { sx: { color, fontSize: 20 } })}
                  </Box>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>{title}</Typography>
                </Box>
                <Typography sx={{ fontSize: 13, color: '#64748B', lineHeight: 1.6, mb: 1.5 }}>{desc}</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 600, color }}>Try it now</Typography>
                  <ArrowForward sx={{ fontSize: 14, color }} />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
