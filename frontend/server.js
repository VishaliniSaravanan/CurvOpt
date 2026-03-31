import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import httpProxy from 'http-proxy';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';

console.log(`Backend URL configured as: ${BACKEND_URL}`);

// Create proxy for API requests
const apiProxy = httpProxy.createProxyServer({
  target: BACKEND_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api': ''
  }
});

// Proxy API requests
app.use('/api', (req, res) => {
  apiProxy.web(req, res, (err) => {
    if (err) {
      console.error('Proxy error:', err);
      res.status(503).json({ error: 'Backend service unavailable' });
    }
  });
});

// Serve static files from dist
app.use(express.static(path.join(__dirname, 'dist')));

// SPA fallback: serve index.html for all non-file routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Listen on 0.0.0.0 (required for Render)
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}`);
  console.log(`API proxy configured for backend: ${BACKEND_URL}`);
  console.log(`Application available on port ${PORT}`);
});
