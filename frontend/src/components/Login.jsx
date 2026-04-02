import React, { useState } from 'react';
import {
  Box, Card, CardContent, Typography, TextField, Button,
  CircularProgress, Alert, InputAdornment, IconButton, Divider
} from '@mui/material';
import { Email, Lock, Visibility, VisibilityOff, SmartToy } from '@mui/icons-material';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const API = `${process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000'}/api/auth`;

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: '', password: '' });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Login failed');
      login(data.token, data.user);
      navigate('/');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #1E1B4B 100%)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', p: 2,
    }}>
      <Box sx={{ width: '100%', maxWidth: 420 }}>
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Box sx={{
            width: 56, height: 56, borderRadius: 3,
            background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center', mb: 1.5,
          }}>
            <SmartToy sx={{ color: 'white', fontSize: 28 }} />
          </Box>
          <Typography sx={{ fontWeight: 700, fontSize: 22, color: 'white' }}>HybridAI</Typography>
          <Typography sx={{ fontSize: 13, color: 'rgba(255,255,255,0.55)' }}>Content Analytics Platform</Typography>
        </Box>

        <Card>
          <CardContent sx={{ p: 3.5 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#1E293B', mb: 0.5 }}>Welcome back</Typography>
            <Typography sx={{ fontSize: 13, color: '#94A3B8', mb: 3 }}>Sign in to your account</Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2, borderRadius: 2 }} onClose={() => setError('')}>
                {error}
              </Alert>
            )}

            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth label="Email" type="email" value={form.email}
                onChange={e => setForm({ ...form, email: e.target.value })}
                required autoFocus
                InputProps={{ startAdornment: <InputAdornment position="start"><Email sx={{ fontSize: 18, color: '#94A3B8' }} /></InputAdornment> }}
                sx={{ mb: 2, '& .MuiOutlinedInput-root': { borderRadius: 2, '&:hover fieldset': { borderColor: '#6366F1' }, '&.Mui-focused fieldset': { borderColor: '#4F46E5' } } }}
              />
              <TextField
                fullWidth label="Password" type={showPw ? 'text' : 'password'} value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                required
                InputProps={{
                  startAdornment: <InputAdornment position="start"><Lock sx={{ fontSize: 18, color: '#94A3B8' }} /></InputAdornment>,
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton size="small" onClick={() => setShowPw(v => !v)}>
                        {showPw ? <VisibilityOff sx={{ fontSize: 18 }} /> : <Visibility sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 2.5, '& .MuiOutlinedInput-root': { borderRadius: 2, '&:hover fieldset': { borderColor: '#6366F1' }, '&.Mui-focused fieldset': { borderColor: '#4F46E5' } } }}
              />
              <Button
                fullWidth type="submit" variant="contained" disabled={loading}
                sx={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', py: 1.3, borderRadius: 2, fontWeight: 700, fontSize: 15, boxShadow: '0 4px 14px rgba(79,70,229,0.35)' }}
              >
                {loading ? <CircularProgress size={20} sx={{ color: 'white' }} /> : 'Sign In'}
              </Button>
            </form>

            <Divider sx={{ my: 2.5 }}>
              <Typography sx={{ fontSize: 12, color: '#CBD5E1' }}>OR</Typography>
            </Divider>

            <Box sx={{ textAlign: 'center' }}>
              <Typography sx={{ fontSize: 13, color: '#64748B' }}>
                Don't have an account?{' '}
                <Link to="/register" style={{ color: '#4F46E5', fontWeight: 600, textDecoration: 'none' }}>
                  Create one
                </Link>
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}
