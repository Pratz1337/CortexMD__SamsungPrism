const { createServer } = require('https');
const { parse } = require('url');
const next = require('next');
const fs = require('fs');
const path = require('path');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

const httpsOptions = {
  key: fs.readFileSync(path.join(__dirname, 'ssl_certs', 'key.pem')),
  cert: fs.readFileSync(path.join(__dirname, 'ssl_certs', 'cert.pem')),
};

app.prepare().then(() => {
  createServer(httpsOptions, (req, res) => {
    const parsedUrl = parse(req.url, true);
    handle(req, res, parsedUrl);
  }).listen(3000, '0.0.0.0', (err) => {
    if (err) throw err;
    console.log('🔐 HTTPS Next.js server ready!');
    console.log('📱 Mobile access: https://192.168.1.6:3000');
    console.log('💻 Local access: https://localhost:3000');
  });
});
