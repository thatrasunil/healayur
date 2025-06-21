# 🔐 OAuth Setup Guide for Heal Ayur

This guide will help you set up Google and GitHub OAuth authentication for your Heal Ayur application.

## 🚀 Quick Setup

1. **Copy Environment File**
   ```bash
   cp .env.example .env
   ```

2. **Configure OAuth Providers** (follow sections below)

3. **Update .env file** with your credentials

4. **Restart the application**

---

## 🔵 Google OAuth Setup

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google+ API

### Step 2: Configure OAuth Consent Screen
1. Go to **APIs & Services** → **OAuth consent screen**
2. Choose **External** user type
3. Fill in required information:
   - **App name**: Heal Ayur
   - **User support email**: Your email
   - **Developer contact**: Your email
4. Add scopes: `email`, `profile`, `openid`
5. Save and continue

### Step 3: Create OAuth Credentials
1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth 2.0 Client IDs**
3. Choose **Web application**
4. Set **Authorized redirect URIs**:
   - `http://localhost:5000/auth/google/callback`
   - `https://yourdomain.com/auth/google/callback` (for production)
5. Copy **Client ID** and **Client Secret**

### Step 4: Update .env File
```env
GOOGLE_CLIENT_ID=your-actual-google-client-id
GOOGLE_CLIENT_SECRET=your-actual-google-client-secret
```

---

## ⚫ GitHub OAuth Setup

### Step 1: Create GitHub OAuth App
1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **New OAuth App**
3. Fill in the details:
   - **Application name**: Heal Ayur
   - **Homepage URL**: `http://localhost:5000` (or your domain)
   - **Authorization callback URL**: `http://localhost:5000/auth/github/callback`
4. Click **Register application**

### Step 2: Get Credentials
1. Copy the **Client ID**
2. Generate a new **Client Secret**
3. Copy the **Client Secret**

### Step 3: Update .env File
```env
GITHUB_CLIENT_ID=your-actual-github-client-id
GITHUB_CLIENT_SECRET=your-actual-github-client-secret
```

---

## 🔧 Testing OAuth

### Test Google OAuth
1. Go to `http://localhost:5000/login`
2. Click **Google** button
3. You should be redirected to Google login
4. After authorization, you'll be redirected back to Heal Ayur

### Test GitHub OAuth
1. Go to `http://localhost:5000/login`
2. Click **GitHub** button
3. You should be redirected to GitHub login
4. After authorization, you'll be redirected back to Heal Ayur

---

## 🛠️ Troubleshooting

### Common Issues

**1. "redirect_uri_mismatch" Error**
- Check that your redirect URIs match exactly in OAuth settings
- Make sure to include the correct protocol (http/https)

**2. "invalid_client" Error**
- Verify your Client ID and Client Secret are correct
- Check that the OAuth app is properly configured

**3. "access_denied" Error**
- User cancelled the OAuth flow
- Check OAuth consent screen configuration

**4. "scope" Errors**
- Verify the requested scopes are approved in OAuth settings
- For Google: ensure email, profile, openid scopes are enabled

### Debug Mode
Set `FLASK_ENV=development` in your `.env` file for detailed error messages.

---

## 🔒 Security Notes

1. **Never commit .env file** - it contains sensitive credentials
2. **Use HTTPS in production** - OAuth requires secure connections
3. **Rotate secrets regularly** - update OAuth credentials periodically
4. **Limit OAuth scopes** - only request necessary permissions

---

## 📝 Production Deployment

For production deployment:

1. **Update redirect URIs** to use your production domain
2. **Use environment variables** instead of .env file
3. **Enable HTTPS** for secure OAuth flows
4. **Set proper CORS settings** if using different domains

Example production redirect URIs:
- Google: `https://yourdomain.com/auth/google/callback`
- GitHub: `https://yourdomain.com/auth/github/callback`

---

## ✅ Verification

After setup, you should see:
- ✅ Google OAuth button works on login/signup pages
- ✅ GitHub OAuth button works on login/signup pages
- ✅ Users can login with their Google/GitHub accounts
- ✅ User profiles are created automatically
- ✅ OAuth users can access all authenticated features

---

## 📞 Support

If you encounter issues:
1. Check the application logs for detailed error messages
2. Verify your OAuth app configurations
3. Test with a fresh browser session (clear cookies)
4. Ensure all redirect URIs are correctly configured

Happy coding! 🌿✨
