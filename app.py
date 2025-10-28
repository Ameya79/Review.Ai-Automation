import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import time
import re

# Configure Gemini API
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

def get_available_models():
    """Get all available Gemini models that support generateContent"""
    try:
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
        return available
    except Exception as e:
        return ["models/gemini-2.5-flash", "models/gemini-2.0-flash-lite"]

def fetch_website_content(url):
    """Fetch and parse website content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:40000]  # Larger content for comprehensive analysis
    except Exception as e:
        return f"Error fetching website: {str(e)}"

def analyze_website_with_gemini(url, content, selected_model):
    """Analyze website content using Gemini with retry logic"""
    
    # Try multiple models if rate limited
    models_to_try = [selected_model, "models/gemini-2.0-flash-lite", "models/gemini-2.5-flash-lite"]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            
            # Detailed, comprehensive prompt
            prompt = f"""Analyze the following website from {url} in extreme detail and provide a comprehensive, elaborate report.

Extract and present EVERYTHING about this website in the following format:

═══════════════════════════════════════════════════════════════

WEBSITE NAME:
[Provide the complete, official name of the website/company/service]

═══════════════════════════════════════════════════════════════

WEBSITE OVERVIEW:
[Write a detailed, multi-paragraph description covering:
- What the website is about
- The main purpose and mission
- Core value proposition
- What makes it unique or different
- Brief history or background if mentioned
- Overall positioning in the market]

═══════════════════════════════════════════════════════════════

TARGET AUDIENCE:
[Provide detailed information about who this website is for:
- Primary target demographics
- Secondary audiences
- User personas or customer types
- Industries or sectors served
- Geographic targeting if mentioned
- Skill level or expertise expected]

═══════════════════════════════════════════════════════════════

KEY FEATURES & SERVICES:
[List and describe in detail ALL features, services, products, or offerings:
- Main features with full descriptions
- Secondary features
- Tools or resources provided
- Integrations or partnerships
- Technology stack if mentioned
- Unique capabilities
Be thorough and include everything mentioned]

═══════════════════════════════════════════════════════════════

PRICING INFORMATION:
[If pricing is mentioned, provide complete details:
- All pricing tiers/plans with exact costs
- What's included in each plan
- Free trial information
- Payment terms and billing cycles
- Any discounts or promotions
- Enterprise or custom pricing options
- Comparison between plans
If NOT mentioned: "Pricing information is not available on this page"]

═══════════════════════════════════════════════════════════════

CONTENT SECTIONS & NAVIGATION:
[List all major sections, pages, and content areas:
- Main navigation menu items
- Footer sections
- Key landing pages or content hubs
- Resource sections (blog, docs, guides, etc.)
- Support or help sections
- Community or social features]

═══════════════════════════════════════════════════════════════

COMPANY INFORMATION:
[Include any company/organization details:
- Company name and background
- Location or headquarters
- Team information or leadership
- Company size or stats
- Founding date or history
- Mission and values]

═══════════════════════════════════════════════════════════════

SOCIAL PROOF & CREDIBILITY:
[List all trust indicators:
- Customer testimonials or reviews
- Case studies or success stories
- Client logos or partnerships
- Awards or certifications
- Press mentions or media coverage
- Statistics or metrics shared
- User counts or community size]

═══════════════════════════════════════════════════════════════

CONTACT & SUPPORT:
[Provide all contact information:
- Email addresses
- Phone numbers
- Physical addresses
- Support channels (chat, email, phone)
- Social media links
- Contact forms or support portals]

═══════════════════════════════════════════════════════════════

CALLS-TO-ACTION:
[List the main actions the website encourages:
- Primary CTAs (sign up, get started, etc.)
- Secondary CTAs
- Lead magnets or free offerings
- Newsletter or email signup
- Demo or trial requests]

═══════════════════════════════════════════════════════════════

TECHNICAL & ADDITIONAL DETAILS:
[Include any other relevant information:
- Technology or platform details
- Mobile app availability
- API or developer resources
- Privacy policy highlights
- Terms of service key points
- Compliance or certifications
- Any unique or notable features not covered above]

═══════════════════════════════════════════════════════════════

OVERALL IMPRESSION:
[Provide a brief summary of the website's quality, design, and effectiveness]

═══════════════════════════════════════════════════════════════

Website Content to Analyze:
{content}

INSTRUCTIONS:
- Be extremely thorough and detailed in every section
- Extract ALL available information from the content
- Use PLAIN TEXT formatting only - NO markdown, NO asterisks, NO special formatting
- Use simple line breaks and spacing for structure
- Use CAPITAL LETTERS for section headers
- Use hyphens (-) or numbers (1, 2, 3) for lists
- If information for a section is not available, state "Information not available on this page"
- Make the output copy-paste friendly with proper formatting
- Include specific details, numbers, and quotes where relevant
- DO NOT use ** for bold, * for italic, or # for headers - use plain text only"""

            response = model.generate_content(prompt)
            
            # Clean up the response text - remove markdown formatting
            clean_text = response.text
            # Remove bold markdown
            clean_text = clean_text.replace('**', '')
            # Remove italic markdown
            clean_text = clean_text.replace('*', '')
            # Remove header markdown
            clean_text = clean_text.replace('###', '')
            clean_text = clean_text.replace('##', '')
            clean_text = clean_text.replace('#', '')
            
            return f"Analysis completed using: {model_name}\n\n{clean_text}"
            
        except Exception as e:
            error_msg = str(e)
            
            # If rate limit, try next model
            if "429" in error_msg or "quota" in error_msg.lower():
                st.warning(f"Rate limit on {model_name}, trying alternative...")
                continue
            else:
                return f"Error with {model_name}: {error_msg}"
    
    # All models failed
    return """⚠️ ALL MODELS RATE LIMITED

Your API key has exhausted its quota across all available models.

SOLUTIONS:
1. Wait 1-2 hours for quota to reset
2. Create a new API key at: https://aistudio.google.com/apikey
3. Check your usage: https://ai.dev/usage

Free tier limits:
- 15 requests per minute
- 1500 requests per day
- Limited tokens per day"""

# Streamlit UI
st.set_page_config(
    page_title="Website Analyzer - Review.Ai",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header with branding
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #6366f1; margin-bottom: 0.5rem;'>🤖 Website Analyzer</h1>
        <p style='color: #6b7280; font-size: 1.1rem; margin: 0;'>Internal Automation Tool by <strong>Review.Ai</strong></p>
        <hr style='margin-top: 1.5rem; margin-bottom: 0;'>
    </div>
""", unsafe_allow_html=True)

# How to Use Section
with st.expander("📖 How to Use This Tool - Complete Guide", expanded=False):
    st.markdown("""
### 🎯 Purpose
This tool automatically analyzes any website and extracts comprehensive information including:
- Website overview and purpose
- Target audience analysis
- Features and services
- Pricing information
- Company details
- Contact information
- And much more!

---

### 📝 Step-by-Step Instructions

**1. Enter Website URL**
- Paste the complete URL of the website you want to analyze
- Make sure to include `https://` or `http://`
- Example: `https://example.com`

**2. Click "Analyze Website"**
- The tool will fetch the website content
- Gemini AI will analyze all available information
- This may take 10-30 seconds depending on website size

**3. View Results**
- Analysis appears in a formatted code block
- All information is organized into clear sections
- Everything is in plain text format (no markdown symbols)

**4. Copy the Results**
- **Method 1 (Recommended):** Click inside the code block → Press `Ctrl+A` (or `Cmd+A` on Mac) → Press `Ctrl+C` (or `Cmd+C` on Mac)
- **Method 2:** Click "View in Text Area" expander and copy from there
- **Method 3:** Click "💾 Download as TXT" to save as a file

---

### ⚙️ Advanced Settings

**API Key Management:**
- If you encounter rate limits, you can use a different API key
- Go to [Google AI Studio](https://aistudio.google.com/apikey) to create new keys
- Paste your new key in the "⚙️ API Settings" section

**Model Selection:**
- Different models have different quotas
- If one model is rate limited, try selecting a lighter model
- The tool will automatically try fallback models if needed

---

### 🚨 Troubleshooting

**Rate Limit Errors:**
- Free tier has limits: 15 requests/minute, 1500 requests/day
- Solution 1: Wait 1-2 hours for quota to reset
- Solution 2: Create a new API key
- Solution 3: Try selecting a lighter model

**Website Not Loading:**
- Check if the URL is correct and accessible
- Some websites may block automated access
- Try accessing the website in your browser first

**Incomplete Analysis:**
- Some websites have limited public information
- The tool can only analyze visible content on the page
- Dynamic content loaded by JavaScript may not be captured

---

### 💡 Tips for Best Results

1. **Use complete URLs** - Always include the full path if analyzing a specific page
2. **Analyze landing pages** - Homepage or main product pages usually have the most info
3. **Check multiple pages** - For comprehensive analysis, run the tool on different pages
4. **Save your results** - Use the download button to keep records of analyses
5. **Respect quotas** - Space out your requests to avoid hitting rate limits

---

### 🔒 Internal Use Only
This tool is designed for Review.Ai team members for competitive analysis, research, and content gathering purposes.
    """)

st.markdown("---")

# API Key option
with st.expander("⚙️ API Settings"):
    custom_api_key = st.text_input(
        "Use a different API key (optional):",
        type="password",
        help="Paste a new API key if current one is rate limited"
    )
    if custom_api_key:
        genai.configure(api_key=custom_api_key)
        st.success("✅ Using custom API key")
    
    # Model selection
    available_models = get_available_models()
    if available_models:
        selected_model = st.selectbox(
            "Select Model (try lighter models if rate limited):",
            available_models,
            index=0
        )
    else:
        selected_model = "models/gemini-2.5-flash-lite"
        st.warning("Could not load models, using default")

# URL Input
url = st.text_input("Enter Website URL:", placeholder="https://example.com")

if url:
    if st.button("Analyze Website", type="primary"):
        with st.spinner("Fetching website content..."):
            content = fetch_website_content(url)
        
        if content.startswith("Error"):
            st.error(content)
        else:
            with st.spinner("Analyzing with Gemini AI (trying multiple models if needed)..."):
                analysis = analyze_website_with_gemini(url, content, selected_model)
            
            # Store in session state
            st.session_state['analysis'] = analysis
            st.session_state['url'] = url

# Display results if available
if 'analysis' in st.session_state:
    st.markdown("---")
    st.subheader(f"Analysis for: {st.session_state['url']}")
    
    # Check if it's a rate limit error
    if "⚠️" in st.session_state['analysis'] and "RATE LIMITED" in st.session_state['analysis']:
        st.error(st.session_state['analysis'])
        st.info("💡 Try creating a fresh API key or wait for quota reset")
    else:
        # Display analysis in markdown for better formatting
        st.markdown("### 📄 Analysis Results")
        
        # Show in a code block for better readability and copying
        st.code(st.session_state['analysis'], language=None)
        
        # Alternative text area view
        with st.expander("📝 View in Text Area (for manual selection)"):
            st.text_area(
                "Copy from here:", 
                value=st.session_state['analysis'], 
                height=400,
                key="analysis_display"
            )
        
        # Copy and Download buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            # Simple instruction instead of buggy button
            st.info("💡 To copy: Click in the code block above, press Ctrl+A (Cmd+A on Mac), then Ctrl+C (Cmd+C on Mac)")
        
        with col2:
            # Download button
            st.download_button(
                label="💾 Download as TXT",
                data=st.session_state['analysis'],
                file_name=f"website_analysis_{st.session_state['url'].replace('https://', '').replace('http://', '').replace('/', '_')}.txt",
                mime="text/plain",
                type="primary"
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem 0; color: #6b7280;'>
        <p style='margin: 0;'>Powered by Google Gemini AI • Built for <strong style='color: #6366f1;'>Review.Ai</strong></p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.875rem;'>Internal Automation Tool © 2025</p>
    </div>
""", unsafe_allow_html=True)
