const fs = require('fs');
const path = require('path');
const https = require('https');
const { exec } = require('child_process');

// Copy SSL certificates from backend to frontend
const backendCertDir = path.join(__dirname, '..', 'backend', 'ssl_certs');
const frontendCertDir = path.join(__dirname, 'ssl_certs');

// Create frontend cert directory
if (!fs.existsSync(frontendCertDir)) {
    fs.mkdirSync(frontendCertDir, { recursive: true });
}

// Copy certificates
const certFiles = ['cert.pem', 'key.pem'];
certFiles.forEach(file => {
    const srcPath = path.join(backendCertDir, file);
    const destPath = path.join(frontendCertDir, file);
    
    if (fs.existsSync(srcPath)) {
        fs.copyFileSync(srcPath, destPath);
        console.log(`✅ Copied ${file} to frontend`);
    } else {
        console.log(`❌ ${file} not found in backend`);
    }
});

console.log('🔐 SSL certificates copied to frontend');
console.log('📁 Certificates location: ssl_certs/');
console.log('🚀 Ready for HTTPS frontend server');
