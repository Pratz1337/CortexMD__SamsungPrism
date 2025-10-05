# CortexMD Deployment Configuration - GCP

## Server Details
- **GCP External IP**: `34.133.136.240`
- **Backend Port**: `5000` (HTTP)
- **Frontend Port**: `3000` (HTTP)

## Changes Made

### Frontend Configuration

**File: `frontend/.env.local`**
```bash
NEXT_PUBLIC_BACKEND_URL=http://34.133.136.240:5000
NEXT_PUBLIC_API_URL=http://34.133.136.240:5000
```

**File: `frontend/.env`**
```bash
NEXT_PUBLIC_API_URL=http://34.133.136.240:5000
NEXT_PUBLIC_BACKEND_URL=http://34.133.136.240:5000
```

### Backend Configuration

**File: `backend/.env`**
```bash
# Flask Configuration
FLASK_HOST=0.0.0.0  # Binds to all network interfaces
FLASK_PORT=5000

# Frontend Configuration
FRONTEND_URL=http://34.133.136.240:3000
ALLOWED_ORIGINS=http://34.133.136.240:3000,https://34.133.136.240:3000,http://localhost:3000,https://localhost:3000
```

**File: `backend/core/app.py`**
- Added GCP IP to CORS allowed origins:
  - `http://34.133.136.240:3000` (frontend)
  - `https://34.133.136.240:3000` (frontend HTTPS)
  - `http://34.133.136.240:5000` (backend)
  - `https://34.133.136.240:5000` (backend HTTPS)

## Why HTTP Instead of HTTPS?

**Problem**: When using HTTPS with self-signed certificates on GCP, the frontend couldn't connect to the backend due to:
- Certificate validation errors
- Browser security blocking untrusted certificates
- CORS preflight failures

**Solution**: Use HTTP for development/testing
- No certificate issues
- Direct connection
- CORS works properly

## Starting the Services

### On GCP Server:

**Backend:**
```bash
cd /path/to/CortexMD/backend
python app.py
# Or use gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Frontend:**
```bash
cd /path/to/CortexMD/frontend
npm run build
npm start
# Or for development:
npm run dev
```

### Local Development:

You can still develop locally - the CORS configuration allows both:
- `http://localhost:3000` ✅
- `http://34.133.136.240:3000` ✅

## Testing the Connection

### 1. Test Backend Health:
```bash
curl http://34.133.136.240:5000/health
```

Expected response:
```json
{"status": "healthy"}
```

### 2. Test Frontend Connection:
Open in browser:
```
http://34.133.136.240:3000
```

### 3. Test Backend from Frontend:
In browser console:
```javascript
fetch('http://34.133.136.240:5000/health')
  .then(r => r.json())
  .then(console.log)
```

## Firewall Rules Required

Make sure GCP firewall allows:
- **Port 5000** (Backend) - TCP Ingress
- **Port 3000** (Frontend) - TCP Ingress
- **Port 5432** (PostgreSQL) - TCP Ingress (if external)
- **Port 6379** (Redis) - TCP Ingress (if external)

### GCP Firewall Command:
```bash
gcloud compute firewall-rules create allow-cortexmd \
  --allow tcp:3000,tcp:5000 \
  --source-ranges 0.0.0.0/0 \
  --description "Allow CortexMD frontend and backend"
```

## Environment Variables Summary

### Frontend (.env.local)
```bash
NEXT_PUBLIC_BACKEND_URL=http://34.133.136.240:5000
NEXT_PUBLIC_API_URL=http://34.133.136.240:5000
```

### Backend (.env)
```bash
# Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Frontend
FRONTEND_URL=http://34.133.136.240:3000
ALLOWED_ORIGINS=http://34.133.136.240:3000,http://localhost:3000

# Database (already configured)
DATABASE_URL=postgresql://postgres:xi6REKcZ3g33qwEk@pgnode305-mum-1.database.excloud.co.in:5432/cortexmd

# Redis (if needed)
REDIS_URL=redis://localhost:6379/0

# API Keys (already configured)
GOOGLE_API_KEY=AIzaSyAYNH2sZ6S334iEHA8IRM7t2g7QP9eWSU8
GROQ_API_KEY=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
```

## Troubleshooting

### Frontend can't reach backend:

1. **Check backend is running:**
   ```bash
   curl http://34.133.136.240:5000/health
   ```

2. **Check CORS headers:**
   ```bash
   curl -H "Origin: http://34.133.136.240:3000" \
        -H "Access-Control-Request-Method: POST" \
        -X OPTIONS http://34.133.136.240:5000/diagnose -v
   ```

3. **Check GCP firewall:**
   ```bash
   gcloud compute firewall-rules list | grep 5000
   ```

4. **Check if backend is binding to 0.0.0.0:**
   ```bash
   # On server
   netstat -tuln | grep 5000
   # Should show: 0.0.0.0:5000
   ```

### CORS Issues:

If you see CORS errors in browser console:
1. Verify backend CORS configuration includes GCP IP
2. Check `Access-Control-Allow-Origin` header in response
3. Ensure backend is using HTTP not HTTPS

### Connection Timeout:

1. Check GCP firewall rules
2. Verify backend is running
3. Check if port 5000 is open:
   ```bash
   nc -zv 34.133.136.240 5000
   ```

## Production Notes

For production deployment with HTTPS:
1. Get proper SSL certificates (Let's Encrypt)
2. Use nginx as reverse proxy
3. Configure proper HTTPS on both frontend and backend
4. Update CORS to use HTTPS URLs

Example nginx config:
```nginx
server {
    listen 80;
    server_name 34.133.136.240;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name 34.133.136.240;
    
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;
    
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Status

✅ Frontend configured to connect to GCP backend (HTTP)
✅ Backend CORS configured to accept GCP frontend requests
✅ HTTP protocol used to avoid certificate issues
✅ Both local and GCP deployment supported

**Ready to deploy!** 🚀
