/**
 * React Application Entry Point
 * Renders the main App component into the DOM
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Import Material-UI Roboto font
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

// Create root element and render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
