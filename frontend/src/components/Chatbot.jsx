import React, { useState, useRef, useEffect } from 'react';
import {
  Box, Card, CardContent, Typography, TextField, Button,
  CircularProgress, Avatar, Chip, IconButton
} from '@mui/material';
import {
  SmartToy, Person, Send, DeleteOutline, AutoAwesome
} from '@mui/icons-material';

const CHAT_API_URL = `${process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000'}/api/chat`;

const WELCOME_MESSAGE = {
  role: 'bot',
  text: "Hi! I'm your HybridAI assistant. Ask me anything about content analytics — sentiment analysis, engagement detection, popularity prediction, or how to use this dashboard.",
  time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
};

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <Box sx={{
      display: 'flex',
      flexDirection: isUser ? 'row-reverse' : 'row',
      alignItems: 'flex-end',
      gap: 1,
      mb: 2,
    }}>
      <Avatar sx={{
        width: 32, height: 32, flexShrink: 0,
        background: isUser
          ? 'linear-gradient(135deg, #4F46E5, #7C3AED)'
          : 'linear-gradient(135deg, #6366F1, #8B5CF6)',
        fontSize: 14,
      }}>
        {isUser ? <Person sx={{ fontSize: 18 }} /> : <SmartToy sx={{ fontSize: 18 }} />}
      </Avatar>
      <Box sx={{ maxWidth: '72%' }}>
        <Box sx={{
          px: 2, py: 1.25,
          borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
          background: isUser ? 'linear-gradient(135deg, #4F46E5, #7C3AED)' : '#FFFFFF',
          border: isUser ? 'none' : '1px solid #E2E8F0',
          boxShadow: isUser ? '0 4px 14px rgba(79,70,229,0.25)' : '0 1px 3px rgba(0,0,0,0.06)',
        }}>
          <Typography sx={{
            fontSize: 13.5, lineHeight: 1.65,
            color: isUser ? 'white' : '#1E293B',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            {msg.text}
          </Typography>
        </Box>
        <Typography sx={{
          fontSize: 11, color: '#94A3B8', mt: 0.4,
          textAlign: isUser ? 'right' : 'left', px: 0.5,
        }}>
          {msg.time}
        </Typography>
      </Box>
    </Box>
  );
}

function TypingIndicator() {
  return (
    <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1, mb: 2 }}>
      <Avatar sx={{ width: 32, height: 32, flexShrink: 0, background: 'linear-gradient(135deg, #6366F1, #8B5CF6)', fontSize: 14 }}>
        <SmartToy sx={{ fontSize: 18 }} />
      </Avatar>
      <Box sx={{
        px: 2, py: 1.5,
        borderRadius: '4px 16px 16px 16px',
        background: '#FFFFFF', border: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        display: 'flex', gap: 0.5, alignItems: 'center',
      }}>
        {[0, 1, 2].map(i => (
          <Box key={i} sx={{
            width: 7, height: 7, borderRadius: '50%', background: '#94A3B8',
            animation: 'bounce 1.2s infinite',
            animationDelay: `${i * 0.2}s`,
            '@keyframes bounce': {
              '0%, 60%, 100%': { transform: 'translateY(0)' },
              '30%': { transform: 'translateY(-6px)' },
            },
          }} />
        ))}
      </Box>
    </Box>
  );
}

const SUGGESTIONS = [
  'What is sentiment analysis?',
  'How is fake engagement detected?',
  'How accurate is the model?',
  'What does popularity prediction do?',
];

export default function Chatbot() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesBoxRef = useRef(null);

  useEffect(() => {
    // Only scroll the inner messages box, not the page
    if (messagesBoxRef.current) {
      messagesBoxRef.current.scrollTop = messagesBoxRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const userText = (text || input).trim();
    if (!userText || loading) return;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMsg = { role: 'user', text: userText, time };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);
    setError('');
    try {
      const res = await fetch(CHAT_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedMessages.map(m => ({ role: m.role, text: m.text })),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Error ${res.status}`);
      setMessages(prev => [...prev, {
        role: 'bot',
        text: data.reply || "I'm not sure about that. Could you rephrase?",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }]);
    } catch (e) {
      setError(e.message.includes('loading') ? 'Model is loading, please retry in a moment.' : e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const clearChat = () => { setMessages([WELCOME_MESSAGE]); setError(''); setInput(''); };

  return (
    <Box>
      {/* Header Banner */}
      <Box sx={{
        background: 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #6366F1 100%)',
        borderRadius: 3, p: 3, mb: 3, position: 'relative', overflow: 'hidden',
      }}>
        <Box sx={{ position: 'absolute', top: -30, right: -30, width: 180, height: 180, borderRadius: '50%', background: 'rgba(255,255,255,0.06)' }} />
        <Box sx={{ position: 'absolute', bottom: -20, right: 80, width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.04)' }} />
        <Box sx={{ position: 'relative' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <AutoAwesome sx={{ color: '#FCD34D', fontSize: 20 }} />
            <Chip label="AI Assistant" size="small" sx={{ background: 'rgba(255,255,255,0.15)', color: 'white', fontSize: 11, height: 22 }} />
          </Box>
          <Typography variant="h5" sx={{ color: 'white', fontWeight: 700, mb: 0.5 }}>
            HybridAI Chat Assistant
          </Typography>
          <Typography sx={{ color: 'rgba(255,255,255,0.75)', fontSize: 13, maxWidth: 500 }}>
            Ask questions about content analytics, sentiment analysis, engagement detection, or how to interpret your results.
          </Typography>
        </Box>
      </Box>

      {/* Chat Card */}
      <Card>
        <CardContent sx={{ p: 0 }}>
          {/* Card Header */}
          <Box sx={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            px: 2.5, py: 2, borderBottom: '1px solid #F1F5F9',
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 32, height: 32, borderRadius: 1.5, background: '#EEF2FF', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <SmartToy sx={{ color: '#4F46E5', fontSize: 18 }} />
              </Box>
              <Box>
                <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#1E293B' }}>AI Assistant</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: '#10B981' }} />
                  <Typography sx={{ fontSize: 11, color: '#94A3B8' }}>Online</Typography>
                </Box>
              </Box>
            </Box>
            <IconButton size="small" onClick={clearChat}
              sx={{ color: '#94A3B8', '&:hover': { color: '#EF4444', background: '#FEF2F2' }, borderRadius: 1.5 }}
              title="Clear chat">
              <DeleteOutline sx={{ fontSize: 18 }} />
            </IconButton>
          </Box>

          {/* Messages Area */}
          <Box ref={messagesBoxRef} sx={{
            height: 420, overflowY: 'auto', px: 2.5, py: 2, background: '#F8FAFC',
            '&::-webkit-scrollbar': { width: 4 },
            '&::-webkit-scrollbar-track': { background: 'transparent' },
            '&::-webkit-scrollbar-thumb': { background: '#E2E8F0', borderRadius: 4 },
          }}>
            {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
            {loading && <TypingIndicator />}
          </Box>

          {/* Error */}
          {error && (
            <Box sx={{ px: 2.5, py: 1, background: '#FEF2F2', borderTop: '1px solid #FECACA' }}>
              <Typography sx={{ fontSize: 12, color: '#DC2626' }}>{error}</Typography>
            </Box>
          )}

          {/* Suggestion Chips */}
          {messages.length === 1 && (
            <Box sx={{ px: 2.5, py: 1.5, borderTop: '1px solid #F1F5F9', display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {SUGGESTIONS.map(s => (
                <Chip key={s} label={s} size="small" onClick={() => sendMessage(s)}
                  sx={{ cursor: 'pointer', fontSize: 12, '&:hover': { background: '#EEF2FF', color: '#4F46E5' } }} />
              ))}
            </Box>
          )}

          {/* Input */}
          <Box sx={{ px: 2.5, py: 2, borderTop: '1px solid #F1F5F9', display: 'flex', gap: 1.5, alignItems: 'flex-end' }}>
            <TextField
              fullWidth multiline maxRows={4} placeholder="Ask anything about content analytics..."
              value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKeyDown}
              disabled={loading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2.5, fontSize: 13.5,
                  '&:hover fieldset': { borderColor: '#6366F1' },
                  '&.Mui-focused fieldset': { borderColor: '#4F46E5' },
                },
              }}
            />
            <Button
              variant="contained" onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              sx={{
                minWidth: 48, width: 48, height: 48, borderRadius: 2.5, p: 0, flexShrink: 0,
                background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
                boxShadow: '0 4px 14px rgba(79,70,229,0.3)',
              }}
            >
              {loading ? <CircularProgress size={18} sx={{ color: 'white' }} /> : <Send sx={{ fontSize: 18 }} />}
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
