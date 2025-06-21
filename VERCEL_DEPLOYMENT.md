# Vercel Deployment Guide for Heal Ayur

## 🚀 Vercel Deployment Configuration

### Files Added for Vercel:
- `vercel.json` - Vercel configuration
- `api/index.py` - Vercel entry point
- `requirements-vercel.txt` - Simplified dependencies
- `.vercelignore` - Files to exclude from deployment

### Environment Variables to Set in Vercel Dashboard:

1. **GEMINI_API_KEY** (Required for AI features)
   - Value: `AIzaSyDhsUnlg85rpkGoiJlKnJHNXAl_0FfIeh0`

2. **FACE_PLUS_API_KEY** (Required for advanced analysis)
   - Value: `8f0FzOLKgxrCCai5NzAkiGp_D-5XRd-f`

3. **FACE_PLUS_API_SECRET** (Required for advanced analysis)
   - Value: `8YbEz4LCUZ2-Furl5Oohw0RonwgV18CX`

4. **SECRET_KEY** (Required for sessions)
   - Value: `heal-ayur-secret-key-2024-vercel`

5. **FLASK_ENV** (Set automatically)
   - Value: `production`

### Deployment Steps:

1. **Connect GitHub Repository**
   - Go to Vercel Dashboard
   - Import project from GitHub: `thatrasunil/healayur`

2. **Configure Build Settings**
   - Framework Preset: Other
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
   - Install Command: `pip install -r requirements-vercel.txt`

3. **Set Environment Variables**
   - Add all the environment variables listed above

4. **Deploy**
   - Click "Deploy" button
   - Wait for deployment to complete

### Troubleshooting:

1. **404 Error**: Usually means the routing is not configured correctly
   - Check that `api/index.py` exists
   - Verify `vercel.json` configuration

2. **Import Errors**: Dependencies not found
   - Check `requirements-vercel.txt`
   - Ensure all required packages are listed

3. **Memory/Timeout Issues**: 
   - Increase memory in `vercel.json`
   - Optimize heavy dependencies

4. **Database Issues**:
   - SQLite may not work on Vercel
   - Consider using Vercel KV or external database

### Features Available on Vercel:
✅ Basic skin analysis
✅ AI chatbot with Gemini
✅ User authentication
✅ Real-time features (limited)
✅ Mobile-optimized UI

### Features Limited on Vercel:
⚠️ File uploads (temporary storage)
⚠️ SQLite database (use external DB)
⚠️ WebSocket connections (may be limited)
⚠️ Large file processing

### Alternative Deployment Options:
- **Render.com** (Recommended for full features)
- **Railway**
- **Heroku**
- **DigitalOcean App Platform**

For full functionality, Render.com is still recommended as it supports:
- Persistent file storage
- Full WebSocket support
- SQLite database
- Longer execution times
