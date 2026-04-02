import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  CssBaseline, Box, Drawer, AppBar, Toolbar, Typography, List,
  ListItem, ListItemButton, ListItemIcon, ListItemText, IconButton,
  useMediaQuery, Avatar, Menu, MenuItem, Divider, Tooltip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology,
  Security,
  TrendingUp,
  CloudUpload,
  FlashOn,
  Menu as MenuIcon,
  SmartToy,
  Logout,
} from '@mui/icons-material';
import Dashboard from './components/Dashboard';
import SentimentAnalysis from './components/SentimentAnalysis';
import EngagementDetection from './components/EngagementDetection';
import PopularityPrediction from './components/PopularityPrediction';
import UploadData from './components/UploadData';
import LivePrediction from './components/LivePrediction';
import Chatbot from './components/Chatbot';
import Login from './components/Login';
import Register from './components/Register';
import { AuthProvider, useAuth } from './context/AuthContext';

const DRAWER_WIDTH = 260;

const theme = createTheme({
  palette: {
    primary: { main: '#4F46E5' },
    secondary: { main: '#7C3AED' },
    background: { default: '#F1F5F9', paper: '#FFFFFF' },
    text: { primary: '#1E293B', secondary: '#64748B' },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", sans-serif',
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04)',
        },
      },
    },
    MuiPaper: { styleOverrides: { root: { borderRadius: 16 } } },
    MuiButton: {
      styleOverrides: {
        root: { borderRadius: 10, textTransform: 'none', fontWeight: 600 },
        contained: {
          boxShadow: '0 4px 14px rgba(79,70,229,0.25)',
          '&:hover': { boxShadow: '0 6px 20px rgba(79,70,229,0.35)' },
        },
      },
    },
  },
});

const NAV_ITEMS = [
  { label: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { label: 'Sentiment Analysis', path: '/sentiment', icon: <Psychology /> },
  { label: 'Engagement Detection', path: '/engagement', icon: <Security /> },
  { label: 'Popularity Prediction', path: '/prediction', icon: <TrendingUp /> },
  { label: 'Live Prediction', path: '/live-prediction', icon: <FlashOn /> },
  { label: 'Upload Data', path: '/upload', icon: <CloudUpload /> },
  { label: 'AI Chatbot', path: '/chatbot', icon: <SmartToy /> },
];

function Sidebar({ mobileOpen, onClose }) {
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleNav = (path) => {
    navigate(path);
    if (isMobile) onClose();
  };

  const content = (
    <Box sx={{
      height: '100%',
      background: 'linear-gradient(180deg, #1E1B4B 0%, #312E81 50%, #1E1B4B 100%)',
      color: 'white',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <Box sx={{ p: 3, pb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
          <Box sx={{
            width: 40, height: 40, borderRadius: 2,
            background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <SmartToy sx={{ color: 'white', fontSize: 22 }} />
          </Box>
          <Box>
            <Typography sx={{ fontWeight: 700, fontSize: 15, lineHeight: 1.2, color: 'white' }}>
              HybridAI
            </Typography>
            <Typography sx={{ fontSize: 11, color: 'rgba(255,255,255,0.55)', lineHeight: 1 }}>
              Content Analytics
            </Typography>
          </Box>
        </Box>
      </Box>

      <Box sx={{ px: 1.5, flex: 1 }}>
        <Typography sx={{ fontSize: 10, fontWeight: 600, color: 'rgba(255,255,255,0.4)', px: 1.5, mb: 1, letterSpacing: 1 }}>
          MAIN MENU
        </Typography>
        <List dense disablePadding>
          {NAV_ITEMS.map(({ label, path, icon }) => {
            const active = location.pathname === path;
            return (
              <ListItem key={path} disablePadding sx={{ mb: 0.5 }}>
                <ListItemButton
                  onClick={() => handleNav(path)}
                  sx={{
                    borderRadius: 2, py: 1.2, px: 1.5,
                    background: active ? 'rgba(99,102,241,0.25)' : 'transparent',
                    border: active ? '1px solid rgba(99,102,241,0.4)' : '1px solid transparent',
                    '&:hover': { background: 'rgba(255,255,255,0.08)' },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36, color: active ? '#A5B4FC' : 'rgba(255,255,255,0.5)' }}>
                    {React.cloneElement(icon, { sx: { fontSize: 20 } })}
                  </ListItemIcon>
                  <ListItemText
                    primary={label}
                    primaryTypographyProps={{
                      fontSize: 13.5,
                      fontWeight: active ? 600 : 400,
                      color: active ? 'white' : 'rgba(255,255,255,0.65)',
                    }}
                  />
                  {active && (
                    <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: '#6366F1' }} />
                  )}
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Box>
    </Box>
  );

  if (isMobile) {
    return (
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onClose}
        ModalProps={{ keepMounted: true }}
        sx={{ '& .MuiDrawer-paper': { width: DRAWER_WIDTH, border: 'none' } }}
      >
        {content}
      </Drawer>
    );
  }

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: DRAWER_WIDTH,
        flexShrink: 0,
        '& .MuiDrawer-paper': { width: DRAWER_WIDTH, border: 'none', position: 'relative' },
      }}
    >
      {content}
    </Drawer>
  );
}

function PageTitle() {
  const location = useLocation();
  const item = NAV_ITEMS.find(n => n.path === location.pathname);
  return item?.label || 'Dashboard';
}

function UserMenu() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [anchor, setAnchor] = useState(null);

  const initials = user?.name
    ? user.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : '?';

  return (
    <>
      <Tooltip title={user?.name || 'Account'}>
        <IconButton onClick={e => setAnchor(e.currentTarget)} size="small" sx={{ ml: 1 }}>
          <Avatar sx={{
            width: 36, height: 36,
            background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
            fontSize: 14, fontWeight: 700,
          }}>
            {initials}
          </Avatar>
        </IconButton>
      </Tooltip>
      <Menu
        anchorEl={anchor}
        open={Boolean(anchor)}
        onClose={() => setAnchor(null)}
        PaperProps={{ sx: { mt: 1, minWidth: 190, borderRadius: 2 } }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography sx={{ fontWeight: 700, fontSize: 14, color: '#1E293B' }}>{user?.name}</Typography>
          <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>{user?.email}</Typography>
        </Box>
        <Divider />
        <MenuItem
          onClick={() => { setAnchor(null); logout(); navigate('/login'); }}
          sx={{ gap: 1.5, color: '#EF4444', py: 1.2 }}
        >
          <Logout sx={{ fontSize: 18 }} />
          <Typography sx={{ fontSize: 13, fontWeight: 600 }}>Sign Out</Typography>
        </MenuItem>
      </Menu>
    </>
  );
}

function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const location = useLocation();
  const mainRef = useRef(null);

  useEffect(() => {
    if (mainRef.current) mainRef.current.scrollTop = 0;
  }, [location.pathname]);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', background: '#F1F5F9' }}>
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />

      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <AppBar
          position="sticky" elevation={0}
          sx={{
            background: 'rgba(255,255,255,0.9)',
            backdropFilter: 'blur(12px)',
            borderBottom: '1px solid rgba(0,0,0,0.06)',
            color: '#1E293B',
          }}
        >
          <Toolbar sx={{ gap: 2, minHeight: '64px !important' }}>
            {isMobile && (
              <IconButton onClick={() => setMobileOpen(true)} edge="start">
                <MenuIcon />
              </IconButton>
            )}
            <Box sx={{ flex: 1 }}>
              <Typography variant="h6" sx={{ fontWeight: 700, color: '#1E293B', fontSize: 17 }}>
                <PageTitle />
              </Typography>
              <Typography sx={{ fontSize: 12, color: '#94A3B8' }}>
                Hybrid AI Content Popularity Prediction System
              </Typography>
            </Box>
            <Tooltip title="AI Chatbot">
              <IconButton
                onClick={() => navigate('/chatbot')}
                sx={{
                  width: 36, height: 36,
                  background: 'linear-gradient(135deg, #6366F1, #8B5CF6)',
                  '&:hover': { background: 'linear-gradient(135deg, #4F46E5, #7C3AED)' },
                  borderRadius: 2,
                }}
              >
                <SmartToy sx={{ color: 'white', fontSize: 18 }} />
              </IconButton>
            </Tooltip>
            <UserMenu />
          </Toolbar>
        </AppBar>

        <Box ref={mainRef} component="main" sx={{ flex: 1, p: { xs: 2, md: 3 }, overflow: 'auto' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/sentiment" element={<SentimentAnalysis />} />
            <Route path="/engagement" element={<EngagementDetection />} />
            <Route path="/prediction" element={<PopularityPrediction />} />
            <Route path="/live-prediction" element={<LivePrediction />} />
            <Route path="/upload" element={<UploadData />} />
            <Route path="/chatbot" element={<Chatbot />} />
          </Routes>
        </Box>
      </Box>
    </Box>
  );
}

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return null;
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/*" element={
              <ProtectedRoute>
                <AppLayout />
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}
