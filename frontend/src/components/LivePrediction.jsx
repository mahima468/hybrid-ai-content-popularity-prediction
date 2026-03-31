import React, { useState, useEffect, useRef } from 'react';
import {
  Box, Grid, Card, CardContent, Typography, Button, Chip,
  LinearProgress, TextField, Alert
} from '@mui/material';
import {
  FlashOn, Stop, CheckCircle, Warning, RadioButtonChecked
} from '@mui/icons-material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import { apiCall, API_ENDPOINTS } from '../config/api';

const DEMO_TEXTS = [
  'This video is absolutely incredible! Best content I have watched all year.',
  'Terrible waste of time. Do not recommend at all.',
  'Pretty good overall, some parts were boring but mostly enjoyable.',
  'Amazing cinematography and storytelling. A masterpiece!',
  'The production quality is lacking and the content is outdated.',
  'Interesting perspective, I learned something new today.',
  'Completely unoriginal and copied from better creators.',
  'This changed my life! Everyone needs to see this.',
];

export default function LivePrediction() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState([]);
  const [currentText, setCurrentText] = useState('');
  const [stats, setStats] = useState({ positive: 0, negative: 0, total: 0 });
  const [customText, setCustomText] = useState('');
  const intervalRef = useRef(null);
  const indexRef = useRef(0);

  const runOnce = async (text) => {
    try {
      const data = await apiCall(API_ENDPOINTS.SENTIMENT_ANALYZE, {
        method: 'POST',
        body: JSON.stringify({ text, method: 'logistic' }),
      });
      const entry = {
        id: Date.now(),
        text: text.slice(0, 60) + (text.length > 60 ? '...' : ''),
        sentiment: data.sentiment,
        confidence: data.confidence || 0.8,
        time: new Date().toLocaleTimeString(),
      };
      setResults(prev => [entry, ...prev].slice(0, 20));
      setStats(prev => ({
        total: prev.total + 1,
        positive: prev.positive + (data.sentiment === 'Positive' ? 1 : 0),
        negative: prev.negative + (data.sentiment === 'Negative' ? 1 : 0),
      }));
    } catch { }
  };

  const startLive = () => {
    setRunning(true);
    setResults([]);
    setStats({ positive: 0, negative: 0, total: 0 });
    indexRef.current = 0;
    intervalRef.current = setInterval(async () => {
      const text = DEMO_TEXTS[indexRef.current % DEMO_TEXTS.length];
      indexRef.current += 1;
      setCurrentText(text);
      await runOnce(text);
    }, 1800);
  };

  const stopLive = () => {
    setRunning(false);
    setCurrentText('');
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  const chartData = results.slice(0, 15).reverse().map((r, i) => ({
    n: i + 1,
    score: r.confidence * (r.sentiment === 'Positive' ? 1 : -1),
  }));

  const positiveRate = stats.total > 0 ? ((stats.positive / stats.total) * 100).toFixed(0) : 0;

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{ width: 40, height: 40, borderRadius: 2, background: '#FFF7ED', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FlashOn sx={{ color: '#F59E0B', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#1E293B' }}>Live Prediction</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8' }}>Real-time streaming sentiment analysis demo</Typography>
          </Box>
          {running && (
            <Chip
              icon={<RadioButtonChecked sx={{ fontSize: '14px !important' }} />}
              label="LIVE"
              size="small"
              sx={{ background: '#FEE2E2', color: '#EF4444', fontWeight: 700, ml: 1 }}
            />
          )}
        </Box>
      </Box>

      <Grid container spacing={2.5}>
        <Grid item xs={12} md={4}>
          <Card sx={{ mb: 2.5 }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Stream Control</Typography>
              <Button variant="contained" fullWidth onClick={running ? stopLive : startLive}
                startIcon={running ? <Stop /> : <FlashOn />}
                sx={{
                  background: running ? 'linear-gradient(135deg, #EF4444, #DC2626)' : 'linear-gradient(135deg, #F59E0B, #D97706)',
                  py: 1.2, mb: 2,
                }}>
                {running ? 'Stop Stream' : 'Start Live Stream'}
              </Button>
              {running && currentText && (
                <Box sx={{ p: 1.5, background: '#FFFBEB', borderRadius: 2, border: '1px solid #FEF3C7' }}>
                  <Typography sx={{ fontSize: 11, color: '#94A3B8', mb: 0.5 }}>ANALYZING NOW</Typography>
                  <Typography sx={{ fontSize: 12, color: '#92400E', lineHeight: 1.5 }}>"{currentText}"</Typography>
                  <LinearProgress sx={{ mt: 1, borderRadius: 2, '& .MuiLinearProgress-bar': { background: '#F59E0B' } }} />
                </Box>
              )}
            </CardContent>
          </Card>

          <Card sx={{ mb: 2.5 }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Live Stats</Typography>
              <Grid container spacing={1.5}>
                {[
                  { label: 'Total', value: stats.total, color: '#1E293B', bg: '#F8FAFC' },
                  { label: 'Positive', value: stats.positive, color: '#10B981', bg: '#F0FDF4' },
                  { label: 'Negative', value: stats.negative, color: '#EF4444', bg: '#FEF2F2' },
                ].map(({ label, value, color, bg }) => (
                  <Grid item xs={4} key={label}>
                    <Box sx={{ textAlign: 'center', p: 1, background: bg, borderRadius: 2 }}>
                      <Typography sx={{ fontSize: 22, fontWeight: 800, color }}>{value}</Typography>
                      <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>{label}</Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography sx={{ fontSize: 12, color: '#64748B' }}>Positive Rate</Typography>
                  <Typography sx={{ fontSize: 12, fontWeight: 700, color: '#10B981' }}>{positiveRate}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={parseFloat(positiveRate)}
                  sx={{ height: 6, borderRadius: 3, background: '#FEE2E2', '& .MuiLinearProgress-bar': { borderRadius: 3, background: '#10B981' } }} />
              </Box>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 1.5 }}>Manual Input</Typography>
              <TextField fullWidth multiline rows={3} size="small"
                placeholder="Type any text to analyze..."
                value={customText} onChange={e => setCustomText(e.target.value)}
                sx={{ mb: 1.5, '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
              <Button fullWidth variant="outlined"
                onClick={() => { if (customText.trim()) { setCurrentText(customText); runOnce(customText); setCustomText(''); } }}
                disabled={!customText.trim()}
                sx={{ borderColor: '#E2E8F0', color: '#64748B', '&:hover': { borderColor: '#4F46E5', color: '#4F46E5' } }}>
                Analyze Once
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 2.5 }}>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 0.5 }}>Real-time Sentiment Stream</Typography>
              <Typography sx={{ fontSize: 12, color: '#94A3B8', mb: 2 }}>Positive values = positive sentiment · Negative = negative</Typography>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                  <XAxis dataKey="n" tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[-1, 1]} tick={{ fontSize: 11, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                  <Tooltip formatter={v => [v.toFixed(3), 'Sentiment']}
                    contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)', fontSize: 12 }} />
                  <ReferenceLine y={0} stroke="#E2E8F0" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="score" stroke="#4F46E5" strokeWidth={2.5}
                    dot={false} activeDot={{ r: 5, fill: '#4F46E5' }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardContent sx={{ p: 2.5 }}>
              <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B', mb: 2 }}>Live Analysis Feed</Typography>
              {results.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <FlashOn sx={{ fontSize: 40, color: '#CBD5E1', mb: 1 }} />
                  <Typography sx={{ fontSize: 14, color: '#94A3B8' }}>Start the live stream to see real-time analyses</Typography>
                </Box>
              ) : (
                <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                  {results.map((r) => (
                    <Box key={r.id} sx={{ display: 'flex', gap: 1.5, alignItems: 'center', py: 1 }}>
                      <Box sx={{
                        width: 28, height: 28, borderRadius: 1.5, flexShrink: 0,
                        background: r.sentiment === 'Positive' ? '#D1FAE5' : '#FEE2E2',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        {r.sentiment === 'Positive'
                          ? <CheckCircle sx={{ fontSize: 16, color: '#10B981' }} />
                          : <Warning sx={{ fontSize: 16, color: '#EF4444' }} />}
                      </Box>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography sx={{ fontSize: 12, color: '#1E293B', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {r.text}
                        </Typography>
                        <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>{r.time}</Typography>
                      </Box>
                      <Box sx={{ textAlign: 'right', flexShrink: 0 }}>
                        <Chip label={r.sentiment} size="small"
                          sx={{
                            background: r.sentiment === 'Positive' ? '#D1FAE5' : '#FEE2E2',
                            color: r.sentiment === 'Positive' ? '#065F46' : '#991B1B',
                            fontWeight: 600, fontSize: 11, height: 20,
                          }} />
                        <Typography sx={{ fontSize: 11, color: '#CBD5E1', mt: 0.3 }}>
                          {(r.confidence * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
