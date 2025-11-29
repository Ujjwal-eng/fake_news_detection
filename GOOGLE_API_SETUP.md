# Google Fact Check API Setup Guide

## Overview
The Google Fact Check API provides access to fact-check claims from verified publishers like Snopes, PolitiFact, FactCheck.org, and others. This integration enhances this fake news detector by cross-referencing claims against professional fact-checkers.

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Get a Google API Key

1. **Go to Google Cloud Console**
   - Visit: https://console.cloud.google.com/

2. **Create a New Project** (if you don't have one)
   - Click "Select a project" → "New Project"
   - Name: `fake-news-detector` (or any name)
   - Click "Create"

3. **Enable the Fact Check Tools API**
   - In the search bar, type "Fact Check Tools API"
   - Click on "Fact Check Tools API"
   - Click "Enable"

4. **Create API Credentials**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API key"
   - **Copy the API key** (you'll need this!)
   - Optional: Click "Restrict Key" to limit it to Fact Check Tools API only

### Step 2: Set the API Key as Environment Variable

**Windows (PowerShell):**
```powershell
# Temporary (current session only)
$env:GOOGLE_FACT_CHECK_API_KEY="YOUR_API_KEY_HERE"

# Permanent (add to shell profile)
[System.Environment]::SetEnvironmentVariable('GOOGLE_FACT_CHECK_API_KEY', 'YOUR_API_KEY_HERE', 'User')
```

**Windows (Command Prompt):**
```cmd
setx GOOGLE_FACT_CHECK_API_KEY "YOUR_API_KEY_HERE"
```

**Linux/Mac:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export GOOGLE_FACT_CHECK_API_KEY="YOUR_API_KEY_HERE"

# Reload
source ~/.bashrc  # or source ~/.zshrc
```

### Step 3: Restart the Application

```bash
python app.py
```

Expected output:
```
✓ Google Fact Check API enabled
```

Instead of:
```
ℹ️ Google Fact Check API not configured (optional)
```

---

## 🧪 Testing the Integration

### Test with a Known False Claim

Try analyzing this text (known false claim):
```
"COVID-19 vaccines contain microchips that allow the government to track you"
```

If the Google API is working, the output will include warnings like:
```
🌐 Snopes: "COVID-19 vaccines contain microchips" rated as False
```

---

## 📊 How It Works

1. **Text Analysis**: When you submit an article, the fact checker extracts the first 500 characters
2. **API Query**: Sends the text to Google Fact Check API
3. **Match Claims**: Google searches for matching fact-checked claims
4. **Display Results**: Shows ratings from verified publishers (Snopes, PolitiFact, etc.)
5. **Adjust Confidence**: Reduces confidence by 20% for each false/misleading claim found

---

## 💰 API Pricing

**Free Tier:**
- 10,000 requests per day
- No credit card required
- Perfect for educational/personal projects

**Note**: This fake news detector makes 1 API call per article analysis, allowing analysis of 10,000 articles per day for free!

---

## 🔧 Troubleshooting

### "Google API returned status 403"
- **Cause**: API not enabled or invalid key
- **Fix**: Make sure you enabled "Fact Check Tools API" in Google Cloud Console

### "Google Fact Check API not configured"
- **Cause**: Environment variable not set
- **Fix**: Set `GOOGLE_FACT_CHECK_API_KEY` and restart the app

### No Google warnings showing
- **Cause**: The claim might not be in Google's database yet
- **Note**: Google API only contains claims that have been fact-checked by verified publishers
- **Try**: Test with known false claims like COVID/election misinformation

### API quota exceeded
- **Cause**: Used more than 10,000 requests today
- **Fix**: Wait until tomorrow or upgrade to a paid plan

---

## 🎯 Example Output

**Without Google API:**
```
📚 Wikipedia Mode

⚠️ Issues Detected:
📊 Pattern: Invalid percentage: 200%
📊 Pattern: Scam language detected: "share this"
```

**With Google API:**
```
🌐 Google API Active

⚠️ Issues Detected:
🌐 Snopes: "Drinking hot water cures cancer" rated as False
🌐 FactCheck.org: "Government announces 200% tax" rated as Misleading
📊 Pattern: Invalid percentage: 200%
📊 Pattern: Scam language detected: "share this"
```

---

## 🔒 Security Best Practices

1. **Never commit API keys to Git**
   - Add to `.gitignore`: `*.env`, `.env`
   
2. **Restrict the API key** (recommended)
   - Go to Google Cloud Console → Credentials
   - Edit the API key
   - Under "API restrictions", select "Restrict key"
   - Choose "Fact Check Tools API"

3. **Monitor usage**
   - Check Google Cloud Console regularly
   - Set up alerts for unusual activity

---

## 🆘 Support

If you encounter issues:
1. Check the console output when starting the app
2. Verify the API key is correct
3. Ensure the Fact Check Tools API is enabled
4. Check internet connection

**Still not working?** The app will fall back to Wikipedia + pattern matching (still effective!).

---

## 📚 Additional Resources

- [Google Fact Check Tools API Documentation](https://developers.google.com/fact-check/tools/api)
- [Google Cloud Console](https://console.cloud.google.com/)
- [List of Verified Fact-Check Publishers](https://toolbox.google.com/factcheck/explorer)

---

**Note**: The Google Fact Check API is **optional**. The fake news detector works perfectly fine without it using Wikipedia verification and pattern matching. The Google API adds an extra layer of verification from professional fact-checkers.
