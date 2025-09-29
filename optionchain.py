import streamlit as st
import requests
import pandas as pd
import json
import pytz
import time as time_module
from datetime import datetime, time, timedelta
import urllib.parse
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Optional auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORFR = True
except ImportError:
    AUTORFR = False

# Set up Indian timezone
IST = pytz.timezone('Asia/Kolkata')
def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def format_ist_time(dt):
    """Format datetime in IST with AM/PM"""
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    elif dt.tzinfo != IST:
        dt = dt.astimezone(IST)
    return dt.strftime('%I:%M:%S %p')

def get_credentials():
    """Get pre-configured developer credentials"""
    try:
        # Check if secrets file exists
        secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
        if not os.path.exists(secrets_path):
            st.error(f"secrets.toml file not found at: {secrets_path}")
            return None
            
        # Verify secrets are loaded
        if not hasattr(st, 'secrets'):
            st.error("Streamlit secrets not initialized")
            return None
            
        # Check if upstox section exists
        if 'upstox' not in st.secrets:
            st.error("'upstox' section not found in secrets.toml")
            st.error(f"Available sections: {list(st.secrets.keys())}")
            return None
            
        # Get credentials
        try:
            credentials = {
                'access_token': st.secrets["upstox"]["access_token"],
                'api_key': st.secrets["upstox"]["api_key"],
                'api_secret': st.secrets["upstox"]["api_secret"],
                'redirect_uri': st.secrets["upstox"]["redirect_uri"]
            }
            return credentials
        except KeyError as e:
            st.error(f"Missing required credential: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error getting credentials: {str(e)}")
        return None
        if not hasattr(st, 'secrets'):
            st.error("st.secrets is not available. Make sure you're running this with streamlit run")
        elif 'upstox' not in st.secrets:
            st.error("'upstox' section not found in secrets.toml")
            st.write("Current sections in secrets:", list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else "None")
        else:
            st.error(f"Error loading credentials: {str(e)}")
        return None

# Upstox API endpoints
BASE_URL = "https://api.upstox.com/v2"
AUTH_URL = "https://api-v2.upstox.com/login/authorization/dialog"
TOKEN_URL = "https://api-v2.upstox.com/login/authorization/token"

class UpstoxAPI:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        
    def get_auth_url(self, api_key, redirect_uri):
        """Generate authorization URL with proper encoding"""
        rurl = urllib.parse.quote(redirect_uri, safe="")
        uri = f'{AUTH_URL}?response_type=code&client_id={api_key}&redirect_uri={rurl}'
        return uri
    
    def get_access_token(self, auth_code, api_key, api_secret, redirect_uri):
        """Exchange authorization code for access token"""
        try:
            payload = {
                'code': auth_code,
                'client_id': api_key,
                'client_secret': api_secret,
                'redirect_uri': redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            response = requests.post(TOKEN_URL, data=payload, headers=headers)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                self.refresh_token = token_data.get('refresh_token')
                return True, token_data
            else:
                return False, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def refresh_access_token(self, api_key, api_secret, refresh_token):
        """Refresh access token using refresh token"""
        try:
            payload = {
                'refresh_token': refresh_token,
                'client_id': api_key,
                'client_secret': api_secret,
                'grant_type': 'refresh_token'
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            response = requests.post(TOKEN_URL, data=payload, headers=headers)
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                new_refresh_token = token_data.get('refresh_token')
                if new_refresh_token:
                    self.refresh_token = new_refresh_token
                return True, token_data
            else:
                return False, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def get_option_contracts(self, instrument_key, expiry_date=None):
        """Get option contracts for an underlying symbol"""
        if not self.access_token:
            return None, "Access token not available"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            url = f"{BASE_URL}/option/contract"
            params = {'instrument_key': instrument_key}
            
            if expiry_date:
                params['expiry_date'] = expiry_date
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def get_pc_option_chain(self, instrument_key, expiry_date):
        """Get Put-Call option chain data using the correct endpoint"""
        if not self.access_token:
            return None, "Access token not available"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            url = f"{BASE_URL}/option/chain"
            params = {
                'instrument_key': instrument_key,
                'expiry_date': expiry_date
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def get_option_greeks(self, instrument_keys):
        """Get option Greeks for specific instruments"""
        if not self.access_token:
            return None, "Access token not available"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            url = f"{BASE_URL}/option/greek"
            if isinstance(instrument_keys, str):
                instrument_keys = [instrument_keys]
            
            params = {'instrument_key': ','.join(instrument_keys)}
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def get_market_data_feed(self, instrument_key, interval='1minute'):
        """Get market data feed"""
        if not self.access_token:
            return None, "Access token not available"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            url = f"{BASE_URL}/market-quote/ohlc"
            params = {
                'instrument_key': instrument_key,
                'interval': interval
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)
    
    def get_profile(self):
        """Get user profile to test token"""
        if not self.access_token:
            return None, "Access token not available"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            url = f"{BASE_URL}/user/profile"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, response.json() if response.text else f"HTTP {response.status_code}"
        except Exception as e:
            return None, str(e)

def format_number(num: float) -> str:
    """Format large numbers for summary display"""
    if num >= 10000000:  # 1 crore
        return f"{num/10000000:.2f}Cr"
    elif num >= 100000:  # 1 lakh
        return f"{num/100000:.2f}L"
    elif num >= 1000:
        return f"{num/1000:.2f}K"
    else:
        return str(int(num)) if num == int(num) else f"{num:.2f}"

def get_position_signal(ltp: float, change: float, chg_oi: float) -> str:
    """Determine position type based on price change and change in OI"""
    if change == 0 and chg_oi == 0:
        return "No Change"
    
    price_up = change > 0
    price_down = change < 0
    oi_increase = chg_oi > 0
    oi_decrease = chg_oi < 0
    
    if price_up and oi_increase:
        return "Long Build"
    elif price_down and oi_decrease:
        return "Long Unwinding"
    elif price_down and oi_increase:
        return "Short Buildup"
    elif price_up and oi_decrease:
        return "Short Covering"
    elif oi_increase and change == 0:
        return "Fresh Positions"
    elif oi_decrease and change == 0:
        return "Position Unwinding"
    else:
        return "Mixed Activity"

def get_position_color(position: str) -> str:
    """Get color for position type"""
    colors = {
        "Long Build": "#4caf50",
        "Long Unwinding": "#ff5722", 
        "Short Buildup": "#f44336",
        "Short Covering": "#2196f3",
        "Fresh Positions": "#9c27b0",
        "Position Unwinding": "#ff9800",
        "Mixed Activity": "#795548",
        "No Change": "#6b7280"
    }
    return colors.get(position, "#6b7280")

def calculate_pcr(put_value: float, call_value: float) -> float:
    """Calculate Put-Call Ratio"""
    return put_value / call_value if call_value != 0 else 0

def get_pcr_signal(pcr_value: float, metric_type: str = "OI") -> str:
    """Get PCR signal based on value and metric type"""
    if metric_type == "OI":
        if pcr_value > 1.2:
            return "Bearish"
        elif pcr_value < 0.8:
            return "Bullish"
        else:
            return "Neutral"
    else:  # Volume
        if pcr_value > 1.3:
            return "Bearish"
        elif pcr_value < 0.7:
            return "Bullish"
        else:
            return "Neutral"

def calculate_comprehensive_sentiment_score(table_data, bucket_summary, pcr_data, spot_price) -> dict:
    """Comprehensive multi-factor sentiment analysis with weighted scoring"""
    
    scores = {
        "price_action": 0,
        "open_interest": 0, 
        "fresh_activity": 0,
        "position_distribution": 0
    }
    
    # 1. PRICE ACTION ANALYSIS (25% weight)
    price_score = 0
    atm_strike = table_data.loc[table_data["Strike"].sub(spot_price).abs().idxmin(), "Strike"]
    strikes_above_spot = len(table_data[table_data["Strike"] > spot_price])
    strikes_below_spot = len(table_data[table_data["Strike"] < spot_price])
    
    if strikes_above_spot > strikes_below_spot:
        price_score += 20
    elif strikes_below_spot > strikes_above_spot:
        price_score -= 20
    
    max_pain_strike = table_data.loc[table_data["CE_OI"].add(table_data["PE_OI"]).idxmax(), "Strike"]
    price_vs_max_pain = (spot_price - max_pain_strike) / max_pain_strike * 100
    
    if price_vs_max_pain > 2:
        price_score -= 30
    elif price_vs_max_pain < -2:
        price_score += 30
        
    scores["price_action"] = max(-100, min(100, price_score))
    
    # 2. OPEN INTEREST ANALYSIS (30% weight)
    oi_score = 0
    oi_pcr = pcr_data['OVERALL_PCR_OI']
    
    if oi_pcr < 0.6:
        oi_score += 40
    elif oi_pcr < 0.8:
        oi_score += 20
    elif oi_pcr > 1.4:
        oi_score -= 40
    elif oi_pcr > 1.2:
        oi_score -= 20
    
    scores["open_interest"] = max(-100, min(100, oi_score))
    
    # 3. FRESH ACTIVITY ANALYSIS (25% weight)
    activity_score = 0
    chgoi_pcr = pcr_data['OVERALL_PCR_CHGOI']
    
    if chgoi_pcr > 2.0:
        activity_score += 50
    elif chgoi_pcr > 1.5:
        activity_score += 30
    elif chgoi_pcr < 0.3:
        activity_score -= 50
    elif chgoi_pcr < 0.6:
        activity_score -= 30
    
    scores["fresh_activity"] = max(-100, min(100, activity_score))
    
    # 4. POSITION DISTRIBUTION ANALYSIS (20% weight)
    position_score = 0
    ce_positions = table_data['CE_Position'].value_counts()
    pe_positions = table_data['PE_Position'].value_counts()
    
    bullish_ce = ce_positions.get("Long Build", 0) + ce_positions.get("Short Covering", 0)
    bullish_pe = pe_positions.get("Long Unwinding", 0) + pe_positions.get("Short Buildup", 0)
    bearish_ce = ce_positions.get("Short Buildup", 0) + ce_positions.get("Long Unwinding", 0)
    bearish_pe = pe_positions.get("Long Build", 0) + pe_positions.get("Short Covering", 0)
    
    total_strikes = len(table_data)
    net_bullish_activity = (bullish_ce - bearish_ce) + (bullish_pe - bearish_pe)
    position_bias_pct = (net_bullish_activity / total_strikes) * 100
    position_score = max(-100, min(100, position_bias_pct * 10))
    
    scores["position_distribution"] = position_score
    
    # Calculate weighted final score
    weights = {
        "price_action": 0.25,
        "open_interest": 0.30,
        "fresh_activity": 0.25, 
        "position_distribution": 0.20
    }
    
    final_score = sum(scores[key] * weights[key] for key in scores.keys())
    
    # Determine sentiment category and confidence
    if final_score >= 60:
        sentiment = "STRONG BULLISH"
        confidence = "HIGH"
    elif final_score >= 30:
        sentiment = "BULLISH"
        confidence = "HIGH"
    elif final_score >= 15:
        sentiment = "BULLISH BIAS"
        confidence = "MEDIUM"
    elif final_score <= -60:
        sentiment = "STRONG BEARISH"
        confidence = "HIGH" 
    elif final_score <= -30:
        sentiment = "BEARISH"
        confidence = "HIGH"
    elif final_score <= -15:
        sentiment = "BEARISH BIAS"
        confidence = "MEDIUM"
    else:
        sentiment = "NEUTRAL"
        confidence = "MEDIUM"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "final_score": final_score,
        "component_scores": scores
    }

def get_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    try:
        expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
        now = datetime.now()
        days_to_expiry = (expiry - now).days
        return max(days_to_expiry / 365.0, 1/365)
    except:
        return 1/365

# IST timezone for market operations
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current datetime in IST"""
    return datetime.now(IST)

def is_market_open(now=None):
    """Check if NSE market is open (IST timezone)"""
    if now is None:
        now = get_ist_now()
    elif now.tzinfo is None:
        now = IST.localize(now)
    elif now.tzinfo != IST:
        now = now.astimezone(IST)
    
    # NSE is closed on weekends
    if now.weekday() >= 5:
        return False
    
    market_open = time(9, 15)
    market_close = time(15, 30)
    current_time = now.time()
    return market_open <= current_time <= market_close

def setup_page():
    """Set up page configuration and CSS styles"""
    st.set_page_config(page_title="Enhanced Upstox F&O Option Chain", layout="wide")
    
    # Add responsive CSS
    st.markdown("""
    <style>
        @media (max-width: 640px) {
            .element-container {
                width: 100% !important;
                padding: 0 !important;
            }
            .stDataFrame {
                width: 100% !important;
                font-size: 12px !important;
            }
            .stDataFrame > div {
                overflow-x: auto !important;
            }
            .css-1xarl3l {
                font-size: 16px !important;
            }
        }
        .stDataFrame > div {
            max-width: 100% !important;
        }
        .dataframe td {
            min-width: 70px !important;
            white-space: nowrap !important;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Set wide layout for the entire app
    st.set_page_config(
        page_title="Enhanced Upstox F&O Option Chain Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    setup_page()
    
    st.title("Enhanced Upstox F&O Option Chain Dashboard")
    st.markdown("Advanced options data analysis with Greeks and sentiment tracking")
    
    # Initialize session state with all required variables
    if 'upstox_api' not in st.session_state:
        st.session_state.upstox_api = UpstoxAPI()
    if 'token_data' not in st.session_state:
        st.session_state.token_data = None
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = "NIFTY"
    if 'selected_expiry' not in st.session_state:
        st.session_state.selected_expiry = None
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if 'last_data_update' not in st.session_state:
        st.session_state.last_data_update = None
    if 'option_chain_data' not in st.session_state:
        st.session_state.option_chain_data = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
        
    # Get pre-configured developer credentials
    try:
        # Check if .streamlit/secrets.toml exists
        secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
        if not os.path.exists(secrets_path):
            st.error("secrets.toml file not found!")
            st.info(f"Expected location: {secrets_path}")
            st.info("Please create the file with your Upstox credentials")
            return
            
        credentials = get_credentials()
        if not credentials:
            st.error("Failed to load credentials from secrets.toml")
            st.info("""
            Your secrets.toml should look like this:
            ```toml
            [upstox]
            access_token="your_token"
            api_key="your_api_key"
            api_secret="your_secret"
            redirect_uri="your_uri"
            ```
            Make sure there are no spaces around the = signs
            """)
            return
            
        # Verify all required fields are present
        required_fields = ['access_token', 'api_key', 'api_secret', 'redirect_uri']
        missing_fields = [field for field in required_fields if not credentials.get(field)]
        if missing_fields:
            st.error(f"Missing required fields in secrets.toml: {', '.join(missing_fields)}")
            return
            
    except Exception as e:
        st.error(f"Error accessing credentials: {str(e)}")
        st.info("Please verify that your secrets.toml file is properly formatted")
        return
    
    # Set the pre-configured access token
    st.session_state.upstox_api.access_token = credentials['access_token']
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("API Configuration")
        st.info("Using pre-configured developer credentials")
        
        # Set credentials from the pre-configured values
        api_key = credentials['api_key']
        api_secret = credentials['api_secret']
        redirect_uri = credentials['redirect_uri']
        
        # Analysis settings
        st.subheader("Analysis Settings")
        itm_count = st.radio("ITM Strikes", [3, 5], index=1, key="itm_count_radio")
        risk_free_rate = st.number_input("Risk-free Rate (%)", value=5.84, min_value=0.0, max_value=15.0, step=0.1) / 100
        
        # Auto-refresh settings
        st.markdown("---")
        st.subheader("Real-time Data Settings")
        
        # Auto-refresh checkbox - Fixed to prevent page reload
        auto_refresh_enabled = st.checkbox(
            "Enable Auto-Refresh", 
            value=st.session_state.auto_refresh_enabled,
            key="auto_refresh_checkbox",
            help="Automatically refresh option chain data every 30 seconds during market hours"
        )
        
        # Update session state only if changed
        if auto_refresh_enabled != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh_enabled
        
        refresh_interval = 30  # Default value
        if st.session_state.auto_refresh_enabled:
            refresh_interval = st.slider("Refresh Interval (seconds)", 10, 60, 30)
            
            if is_market_open():
                st.success(f"Auto-refresh enabled - Market is OPEN")
                st.info(f"Refreshing every {refresh_interval}s")
            else:
                st.info("Auto-refresh paused - Market is CLOSED")
                st.write("Market hours: 9:15 AM - 3:30 PM (Mon-Fri)")
    
    # Token management
    st.sidebar.subheader("Token Status")
    
    if st.session_state.upstox_api.access_token:
        st.sidebar.success("✓ Connected to Upstox")
        
        if st.sidebar.button("Test Connection"):
            profile_data, error = st.session_state.upstox_api.get_profile()
            if profile_data:
                st.sidebar.success("✓ Connection is valid")
            else:
                st.sidebar.error("⚠️ Connection error: " + str(error))
            st.rerun()
    else:
        st.sidebar.warning("No Access Token")
    
    # Authorization flow
    if api_key and not st.session_state.upstox_api.access_token:
        auth_url = st.session_state.upstox_api.get_auth_url(api_key, redirect_uri)
        
        st.sidebar.markdown("**Get Authorization Code:**")
        st.sidebar.markdown(f"[Click to Authorize]({auth_url})")
        
        with st.sidebar.expander("Instructions"):
            st.markdown(f"""
            **Step by Step:**
            1. Click the authorization link above
            2. Login to your Upstox account
            3. You'll be redirected to: `{redirect_uri}?code=XXXXX`
            4. Copy the entire redirect URL and paste below
            """)
        
        redirect_url_input = st.sidebar.text_area(
            "Paste Complete Redirect URL",
            placeholder=f"{redirect_uri}?code=ABC123XYZ",
            height=80
        )
        
        auth_code = None
        if redirect_url_input:
            try:
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(redirect_url_input.strip())
                query_params = parse_qs(parsed_url.query)
                
                if 'code' in query_params:
                    auth_code = query_params['code'][0]
                    st.sidebar.success(f"Code found: {auth_code[:15]}...")
                else:
                    st.sidebar.error("No 'code' parameter found")
            except Exception as e:
                st.sidebar.error(f"URL parsing error: {str(e)}")
        
        with st.sidebar.expander("Manual Code Entry"):
            manual_code = st.text_input("Authorization Code", placeholder="Paste code manually if URL parsing fails")
            if manual_code:
                auth_code = manual_code.strip()
        
        if auth_code and st.sidebar.button("Get Access Token"):
            with st.spinner("Getting access token..."):
                success, result = st.session_state.upstox_api.get_access_token(
                    auth_code, api_key, api_secret, redirect_uri
                )
                if success:
                    st.sidebar.success("Access token obtained!")
                    st.session_state.token_data = result
                else:
                    st.sidebar.error("Failed to get access token")
                    st.sidebar.json(result)
    
    # Refresh token section
    if st.session_state.upstox_api.refresh_token and api_key and api_secret:
        if st.sidebar.button("Refresh Access Token"):
            with st.spinner("Refreshing token..."):
                success, result = st.session_state.upstox_api.refresh_access_token(
                    api_key, api_secret, st.session_state.upstox_api.refresh_token
                )
                if success:
                    st.sidebar.success("Token refreshed!")
                    st.session_state.token_data = result
                else:
                    st.sidebar.error("Failed to refresh token")
                    st.sidebar.json(result)
    
    # Main content area - FIXED: Create a single persistent container for all content
    main_content_container = st.container()
    
    if st.session_state.upstox_api.access_token:
        with main_content_container:
            # Auto-refresh implementation - FIXED: Only trigger refresh, don't duplicate UI
            if (AUTORFR and 
                st.session_state.auto_refresh_enabled and 
                is_market_open() and 
                'selected_symbol' in st.session_state and 
                st.session_state.selected_expiry):
                
                count = st_autorefresh(interval=refresh_interval * 1000, limit=None, key="data_refresh")
                
                # Auto-fetch only triggers data refresh, UI remains the same
                if count > 0:
                    fo_instruments = get_fo_instruments()
                    instrument_key = fo_instruments.get(st.session_state.selected_symbol)
                    if instrument_key:
                        auto_fetch_option_chain(instrument_key, st.session_state.selected_symbol, 
                                              st.session_state.selected_expiry, itm_count, risk_free_rate)
            
            st.success("Ready to fetch options data!")
            
            # F&O Instrument Selection
            st.header("Select F&O Instrument")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Get F&O instruments
                fo_instruments = get_fo_instruments()
                
                # Find current index
                try:
                    current_index = list(fo_instruments.keys()).index(st.session_state.selected_symbol)
                except ValueError:
                    current_index = 0
                
                selected_symbol = st.selectbox(
                    "Select Symbol", 
                    list(fo_instruments.keys()), 
                    index=current_index,
                    key="symbol_selectbox"
                )
                if selected_symbol != st.session_state.selected_symbol:
                    st.session_state.selected_symbol = selected_symbol
                
                instrument_key = fo_instruments[selected_symbol]
            
            with col2:
                # Fetch available expiries
                contracts_data, error = st.session_state.upstox_api.get_option_contracts(instrument_key)
                expiry_dates = []
                if contracts_data and 'data' in contracts_data:
                    expiry_dates = sorted({c['expiry'] for c in contracts_data['data'] if 'expiry' in c})
                
                if expiry_dates:
                    # Find current expiry index
                    current_expiry_index = 0
                    if st.session_state.selected_expiry in expiry_dates:
                        current_expiry_index = expiry_dates.index(st.session_state.selected_expiry)
                    
                    selected_expiry = st.selectbox(
                        "Select Expiry", 
                        expiry_dates, 
                        index=current_expiry_index,
                        key="expiry_selectbox"
                    )
                    st.session_state.selected_expiry = selected_expiry
                else:
                    st.warning("No expiry dates found. Try another symbol.")
                    selected_expiry = None
            
            with col3:
                st.write("**Selected:**")
                st.write(f"Symbol: {selected_symbol}")
                st.write(f"Key: {instrument_key}")
                st.write(f"Expiry: {selected_expiry}")
                
            # Manual fetch button (always available)
            if selected_expiry:
                col_fetch1, col_fetch2 = st.columns([1, 1])
                
                with col_fetch1:
                    if st.button("Get Option Chain", type="primary", key="manual_fetch"):
                        fetch_and_display_option_chain(instrument_key, selected_symbol, selected_expiry, itm_count, risk_free_rate)
                
                with col_fetch2:
                    if st.session_state.last_data_update:
                        st.info(f"Last updated: {st.session_state.last_data_update} IST")
            
            # FIXED: Display data in the same container - no separate placeholder needed
            if st.session_state.option_chain_data:
                display_option_chain_dashboard(
                    st.session_state.option_chain_data,
                    selected_symbol,
                    selected_expiry,
                    itm_count,
                    risk_free_rate
                )
    
    else:
        with main_content_container:
            st.error("Developer access token not available or has expired.")
            st.info("""
            Please configure your developer credentials in .streamlit/secrets.toml with this format:
            ```toml
            [upstox]
            access_token = "your_permanent_access_token"
            api_key = "your_api_key"
            api_secret = "your_api_secret"
            redirect_uri = "your_redirect_uri"
            ```
            To get a permanent access token:
            1. Run the app locally
            2. Use the Upstox developer portal
            3. Generate a permanent access token
            4. Add it to your secrets.toml
            """)
            st.warning("Contact the developer if you need access to this application.")
            
            st.markdown("""
            ### Setup Guide:
            
            1. **Authorization Process:**
               - Click the authorization link in sidebar
               - Login to Upstox
               - Copy the complete redirect URL
               - Paste it to auto-extract the code
            
            2. **Token Management:**
               - Access tokens expire after some time
               - Use refresh token to get new access tokens
            
            3. **Enhanced Features:**
               - Real-time option chain with Greeks
               - Advanced sentiment analysis
               - Position tracking and PCR analysis
               - Auto-refresh during market hours
               - Volatility skew and gamma exposure analysis
            """)

def get_fo_instruments():
    """Get F&O instruments list"""
    try:
        import requests
        import gzip
        import json
        from io import BytesIO

        url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
        response = requests.get(url)
        
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
            data = json.load(gz)

        df = pd.DataFrame(data)
        fno_df = df[(df['segment'] == "NSE_FO") | (df['segment'] == "NSE_INDEX")]
        fno_stocks = [x for x in fno_df['name'].unique()]

        fo_instruments = {row['asset_symbol']: row['asset_key'] 
                        for _, row in fno_df.iterrows() 
                        if row['name'] in fno_stocks}

        # Add indices
        fo_instruments.update({
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
            "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
            "SENSEX": "BSE_INDEX|SENSEX"
        })
        
        return fo_instruments
        
    except Exception as e:
        st.error(f"Failed to load instruments: {str(e)}")
        return {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank"
        }

def auto_fetch_option_chain(instrument_key, symbol, expiry, itm_count, risk_free_rate):
    """Auto fetch option chain when conditions are met - FIXED: Only updates data, no UI duplication"""
    if st.session_state.upstox_api.access_token:
        # Fetch data silently without creating new UI elements
        option_data, error = st.session_state.upstox_api.get_pc_option_chain(instrument_key, expiry)
        
        if option_data and error is None and 'data' in option_data:
            st.session_state.option_chain_data = option_data['data']
            st.session_state.last_data_update = datetime.now().strftime('%H:%M:%S')
            # Note: UI will be updated automatically through the existing display logic

def fetch_and_display_option_chain(instrument_key, symbol, expiry, itm_count, risk_free_rate):
    """Fetch and display option chain data - FIXED: No separate placeholder needed"""
    with st.spinner(f"Fetching option chain for {symbol}..."):
        option_data, error = st.session_state.upstox_api.get_pc_option_chain(
            instrument_key, expiry
        )
        
        if option_data and error is None and 'data' in option_data:
            st.session_state.option_chain_data = option_data['data']
            current_time = get_ist_now()
            st.session_state.last_data_update = format_ist_time(current_time)
            st.success(f"Data fetched successfully at {st.session_state.last_data_update} IST")
            # UI will automatically update through the session state
        else:
            st.error(f"Failed to fetch option chain: {error}")

def display_option_chain_dashboard(data, symbol, expiry, itm_count, risk_free_rate):
    """Main dashboard display function - Fixed to prevent blank rows"""
    try:
        if not isinstance(data, list) or len(data) == 0:
            st.warning("No option chain data available")
            return
        
        # Extract spot price
        spot_price = data[0].get('underlying_spot_price', 0) if data else 0
        if spot_price == 0:
            st.error("Unable to get spot price from data")
            return
        
        # Calculate time to expiry
        time_to_expiry = get_time_to_expiry(expiry)
        
        # Display spot price banner
        st.markdown(
            f"""
        <div style="background-color:#0A71E2;padding:12px;border-radius:12px;text-align:center;margin-bottom:12px">
          <div style="color:#fff;font-size:24px;font-weight:700;">{symbol} Spot: ₹{spot_price}</div>
          <div style="color:#D8E9FF;font-size:16px;">Expiry: {expiry} | Time to Expiry: {time_to_expiry*365:.0f} days</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        
        # Process option chain data
        processed_data = process_option_chain_data(data, spot_price)
        
        if not processed_data:
            st.warning("No processed data available")
            return
        
        # Convert to DataFrame and clean data
        table = pd.DataFrame(processed_data)
        
        # Remove any rows with NaN or invalid Strike prices
        table = table.dropna(subset=['Strike'])
        table = table[table['Strike'] > 0]
        
        # Sort by strike and reset index
        table = table.sort_values('Strike').reset_index(drop=True)
        
        # Remove duplicate strikes (keep first occurrence)
        table = table.drop_duplicates(subset=["Strike"], keep='first')
        
        if len(table) == 0:
            st.warning("No valid option chain data after cleaning")
            return
        
        # Find ATM strike and filter data
        atm_strike = table.loc[table["Strike"].sub(spot_price).abs().idxmin(), "Strike"]
        
        below_atm = table[table["Strike"] < atm_strike].tail(itm_count)
        above_atm = table[table["Strike"] > atm_strike].head(itm_count)
        atm_row = table[table["Strike"] == atm_strike]
        
        # Combine tables ensuring no empty DataFrames
        filtered_parts = []
        if not below_atm.empty:
            filtered_parts.append(below_atm)
        if not atm_row.empty:
            filtered_parts.append(atm_row)
        if not above_atm.empty:
            filtered_parts.append(above_atm)
        
        if not filtered_parts:
            st.warning("No data available for selected ITM count")
            return
            
        filtered_table = pd.concat(filtered_parts, axis=0, ignore_index=True)
        filtered_table = filtered_table.sort_values('Strike').reset_index(drop=True)
        
        # Add PCR calculations
        filtered_table["PCR_Strike_OI"] = filtered_table.apply(lambda row: calculate_pcr(row["PE_OI"], row["CE_OI"]), axis=1)
        filtered_table["PCR_Volume"] = filtered_table.apply(lambda row: calculate_pcr(row["PE_Volume"], row["CE_Volume"]), axis=1)
        filtered_table["PCR_ChgOI"] = filtered_table.apply(lambda row: calculate_pcr(row["PE_ChgOI"], row["CE_ChgOI"]), axis=1)
        
        # Calculate GEX data once and store it
        gex_data = []
        for _, row in filtered_table.iterrows():
            ce_gex = row['CE_Gamma'] * row['CE_OI'] * 100 * (spot_price ** 2) * 0.01
            pe_gex = -row['PE_Gamma'] * row['PE_OI'] * 100 * (spot_price ** 2) * 0.01
            net_gex = ce_gex + pe_gex
            
            gex_data.append({
                'Strike': row['Strike'],
                'CE_GEX': ce_gex,
                'PE_GEX': pe_gex,
                'Net_GEX': net_gex,
                'Distance': abs(row['Strike'] - spot_price)
            })
            
        # Store in session state for reuse
        st.session_state.current_gex_data = gex_data
        
        # Create visualizations and analysis
        create_option_chain_visualization(filtered_table, spot_price, symbol)
        
        bucket_summary = calculate_bucket_summaries(filtered_table, atm_strike, spot_price)
        pcr_data = calculate_comprehensive_pcr(bucket_summary)
        
        display_bucket_summaries(bucket_summary, pcr_data)
        
        sentiment_analysis = calculate_comprehensive_sentiment_score(filtered_table, bucket_summary, pcr_data, spot_price)
        display_sentiment_analysis(sentiment_analysis, symbol)
        
        display_option_chain_table(filtered_table, atm_strike, spot_price)
        display_quick_stats(filtered_table, atm_strike)
        
        # Advanced analysis sections
        with st.expander("Volatility Skew Analysis", expanded=False):
            calculate_volatility_skew_analysis(filtered_table, spot_price)
        
        with st.expander("Gamma Exposure Analysis", expanded=False):
            if 'current_gex_data' in st.session_state:
                gex_df = pd.DataFrame(st.session_state.current_gex_data)
                calculate_gamma_exposure_analysis(filtered_table, spot_price, gex_df)
            else:
                st.warning("GEX data not available. Please refresh the page.")
        
        with st.expander("Custom Volatility Index", expanded=False):
            implement_vix_like_index(filtered_table, spot_price, time_to_expiry)
        
        with st.expander("Support & Resistance Levels", expanded=False):
            display_support_resistance_levels(filtered_table, spot_price)
        
        with st.expander("Put-Call Parity Analysis", expanded=False):
            calculate_put_call_parity_analysis(filtered_table, atm_strike)
        
    except Exception as e:
        st.error(f"Error processing option chain data: {str(e)}")
        with st.expander("Raw Data for Debugging"):
            st.json(data)

def process_option_chain_data(data, spot_price):
    """Process raw option chain data into structured format - Fixed to prevent blank rows"""
    processed_data = []
    
    for strike_data in data:
        if 'call_options' in strike_data and 'put_options' in strike_data:
            strike_price = strike_data.get('strike_price', 0)
            
            # Skip if strike price is 0 or invalid
            if strike_price <= 0:
                continue
            
            # Call options data
            call_data = strike_data['call_options']
            call_market = call_data.get('market_data', {})
            call_greeks = call_data.get('option_greeks', {})
            
            # Put options data  
            put_data = strike_data['put_options']
            put_market = put_data.get('market_data', {})
            put_greeks = put_data.get('option_greeks', {})
            
            # Extract market data with proper defaults
            ce_ltp = call_market.get('ltp', 0) or 0
            ce_volume = call_market.get('volume', 0) or 0
            ce_oi = call_market.get('oi', 0) or 0
            ce_prev_oi = call_market.get('prev_oi', 0) or 0
            ce_close_price = call_market.get('close_price', 0) or 0
            
            ce_chg_oi = ce_oi - ce_prev_oi
            ce_change = ce_ltp - ce_close_price if ce_close_price != 0 else 0
            
            pe_ltp = put_market.get('ltp', 0) or 0
            pe_volume = put_market.get('volume', 0) or 0
            pe_oi = put_market.get('oi', 0) or 0
            pe_prev_oi = put_market.get('prev_oi', 0) or 0
            pe_close_price = put_market.get('close_price', 0) or 0
            
            pe_chg_oi = pe_oi - pe_prev_oi
            pe_change = pe_ltp - pe_close_price if pe_close_price != 0 else 0
            
            # Extract Greeks with proper defaults
            ce_iv = call_greeks.get('iv', 0) if call_greeks.get('iv') is not None else 0
            pe_iv = put_greeks.get('iv', 0) if put_greeks.get('iv') is not None else 0

            ce_delta = call_greeks.get('delta', 0) or 0
            ce_gamma = call_greeks.get('gamma', 0) or 0
            ce_theta = call_greeks.get('theta', 0) or 0
            ce_vega = call_greeks.get('vega', 0) or 0
            
            pe_delta = put_greeks.get('delta', 0) or 0
            pe_gamma = put_greeks.get('gamma', 0) or 0
            pe_theta = put_greeks.get('theta', 0) or 0
            pe_vega = put_greeks.get('vega', 0) or 0
            
            # Calculate position signals
            ce_position = get_position_signal(ce_ltp, ce_change, ce_chg_oi)
            pe_position = get_position_signal(pe_ltp, pe_change, pe_chg_oi)
            
            # Only add row if we have valid strike price
            processed_data.append({
                "Strike": strike_price,
                "CE_OI": ce_oi,
                "CE_LTP": ce_ltp,
                "CE_Change": ce_change,
                "CE_Volume": ce_volume,
                "CE_ChgOI": ce_chg_oi,
                "CE_IV": ce_iv,
                "CE_Delta": ce_delta,
                "CE_Gamma": ce_gamma,
                "CE_Theta": ce_theta,
                "CE_Vega": ce_vega,
                "CE_Position": ce_position,
                "PE_OI": pe_oi,
                "PE_LTP": pe_ltp,
                "PE_Change": pe_change,
                "PE_Volume": pe_volume,
                "PE_ChgOI": pe_chg_oi,
                "PE_IV": pe_iv,
                "PE_Delta": pe_delta,
                "PE_Gamma": pe_gamma,
                "PE_Theta": pe_theta,
                "PE_Vega": pe_vega,
                "PE_Position": pe_position,
            })
    
    return processed_data

# All the visualization and analysis functions from the original code
def create_option_chain_visualization(table, spot_price, symbol):
    """Create visualization for option chain data"""
    st.header("OI, ChgOI & Volume Distribution")
    
    # Set better default style
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        'figure.figsize': [16, 8],  # Adjusted for better fit in wide layout
        'figure.dpi': 100,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.autolayout': True  # Enable automatic layout adjustments
    })
    
    # Create figure with better spacing
    fig, ax1 = plt.subplots()
    
    indices = np.arange(len(table))
    bar_width = 0.2
    
    # Plot OI and ChgOI
    ax1.bar(indices - 0.2, table["CE_OI"], bar_width, color="#1f77b4", label="CE OI", alpha=0.7)
    ax1.bar(indices, table["PE_OI"], bar_width, color="#2ca02c", label="PE OI", alpha=0.7)
    ax1.bar(indices + 0.2, table["CE_ChgOI"], bar_width, color="#aec7e8", label="CE ChgOI", alpha=0.7)
    ax1.bar(indices + 0.4, table["PE_ChgOI"], bar_width, color="#98df8a", label="PE ChgOI", alpha=0.7)
    
    ax1.set_ylabel("OI / ChgOI")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(table["Strike"], rotation=45)
    
    # Right axis for Volume
    ax2 = ax1.twinx()
    ax2.plot(indices, table["CE_Volume"], marker="o", color="#ff7f0e", label="CE Volume", linewidth=2)
    ax2.plot(indices, table["PE_Volume"], marker="o", color="#d62728", label="PE Volume", linewidth=2)
    ax2.set_ylabel("Volume")
    
    # Spot line
    spot_position = table.index[table["Strike"].sub(spot_price).abs().idxmin()]
    ax1.axvline(spot_position, color="red", linestyle="--", linewidth=2, label=f"Spot ₹{spot_price}")
    
    # Merge legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax1.set_title(f"{symbol} Option Chain Distribution")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Use full width container for the plot
    with st.container():
        # Add CSS to ensure full width
        st.markdown("""
            <style>
            .element-container {
                width: 100% !important;
                max-width: 100% !important;
            }
            .stPlotlyChart, .stplot {
                width: 100% !important;
                max-width: 100% !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display plot in full width
        st.pyplot(fig, use_container_width=True)
    
    plt.close()

def calculate_bucket_summaries(table, atm_strike, spot_price):
    """Calculate ITM/OTM bucket summaries"""
    
    # Separate ITM and OTM
    ce_itm = table[table["Strike"] < atm_strike]  # CE ITM (below spot)
    ce_otm = table[table["Strike"] > atm_strike]  # CE OTM (above spot)
    pe_itm = table[table["Strike"] > atm_strike]  # PE ITM (above spot)
    pe_otm = table[table["Strike"] < atm_strike]  # PE OTM (below spot)
    
    def aggregate_bucket(df, side):
        if df.empty:
            return {"OI": 0, "ChgOI": 0, "Volume": 0, "IV": 0, "Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
        
        total_oi = df[f"{side}_OI"].sum()
        if total_oi == 0:
            return {"OI": 0, "ChgOI": 0, "Volume": 0, "IV": 0, "Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0}
        
        # Weight Greeks by OI
        weighted_delta = (df[f"{side}_Delta"] * df[f"{side}_OI"]).sum() / total_oi
        weighted_gamma = (df[f"{side}_Gamma"] * df[f"{side}_OI"]).sum() / total_oi
        weighted_theta = (df[f"{side}_Theta"] * df[f"{side}_OI"]).sum() / total_oi
        weighted_vega = (df[f"{side}_Vega"] * df[f"{side}_OI"]).sum() / total_oi
        
        return {
            "OI": total_oi,
            "ChgOI": df[f"{side}_ChgOI"].sum(),
            "Volume": df[f"{side}_Volume"].sum(),
            "IV": df[f"{side}_IV"].mean(),
            "Delta": weighted_delta,
            "Gamma": weighted_gamma,
            "Theta": weighted_theta,
            "Vega": weighted_vega,
        }
    
    return {
        "CE_ITM": aggregate_bucket(ce_itm, "CE"),
        "CE_OTM": aggregate_bucket(ce_otm, "CE"),
        "PE_ITM": aggregate_bucket(pe_itm, "PE"),
        "PE_OTM": aggregate_bucket(pe_otm, "PE"),
    }

def calculate_comprehensive_pcr(bucket_summary):
    """Calculate comprehensive PCR data"""
    return {
        "ITM_PCR_OI": calculate_pcr(bucket_summary["PE_ITM"]["OI"], bucket_summary["CE_ITM"]["OI"]),
        "OTM_PCR_OI": calculate_pcr(bucket_summary["PE_OTM"]["OI"], bucket_summary["CE_OTM"]["OI"]),
        "OVERALL_PCR_OI": calculate_pcr(
            bucket_summary["PE_ITM"]["OI"] + bucket_summary["PE_OTM"]["OI"],
            bucket_summary["CE_ITM"]["OI"] + bucket_summary["CE_OTM"]["OI"]
        ),
        "ITM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_ITM"]["ChgOI"], bucket_summary["CE_ITM"]["ChgOI"]),
        "OTM_PCR_CHGOI": calculate_pcr(bucket_summary["PE_OTM"]["ChgOI"], bucket_summary["CE_OTM"]["ChgOI"]),
        "OVERALL_PCR_CHGOI": calculate_pcr(
            bucket_summary["PE_ITM"]["ChgOI"] + bucket_summary["PE_OTM"]["ChgOI"],
            bucket_summary["CE_ITM"]["ChgOI"] + bucket_summary["CE_OTM"]["ChgOI"]
        ),
        "ITM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_ITM"]["Volume"], bucket_summary["CE_ITM"]["Volume"]),
        "OTM_PCR_VOLUME": calculate_pcr(bucket_summary["PE_OTM"]["Volume"], bucket_summary["CE_OTM"]["Volume"]),
        "OVERALL_PCR_VOLUME": calculate_pcr(
            bucket_summary["PE_ITM"]["Volume"] + bucket_summary["PE_OTM"]["Volume"],
            bucket_summary["CE_ITM"]["Volume"] + bucket_summary["CE_OTM"]["Volume"]
        ),
    }

# Utility functions for color coding
def get_change_color(value):
    """Return color based on value change"""
    if value > 0:
        return "#4caf50"  # Green
    elif value < 0:
        return "#f44336"  # Red
    else:
        return "#757575"  # Gray

def get_pcr_color(pcr_value):
    """Return color based on PCR value"""
    if pcr_value > 1.2:
        return "#f44336"  # Red - Bearish
    elif pcr_value < 0.8:
        return "#4caf50"  # Green - Bullish
    else:
        return "#ff9800"  # Orange - Neutral

def display_bucket_summaries(bucket_summary, pcr_data):
    """Display bucket summaries with enhanced color coding"""
    st.subheader("Bucket Summaries with Greeks Analysis")
    
    left, middle, right = st.columns([1, 1, 1])
    
    def get_pcr_signal(pcr_value, type):
        if type == 'OI':
            if pcr_value > 1.2: return "Bearish"
            elif pcr_value < 0.8: return "Bullish"
            else: return "Neutral"
        return ""  # Default return for other types
    
    def display_bucket_stats(data, category, oi_color, chgoi_color):
        return f"""
        <div style="background: linear-gradient(90deg, {oi_color}15, transparent); padding: 10px; border-radius: 8px; border-left: 4px solid {oi_color}; margin-bottom: 10px;">
            <div style="color: {oi_color}; font-weight: bold;">OI: {format_number(data['OI'])}</div>
            <div style="color: {chgoi_color}; font-weight: bold;">ChgOI: {'+' if data['ChgOI'] >= 0 else ''}{format_number(data['ChgOI'])}</div>
            <div>Volume: {format_number(data['Volume'])}</div>
            <div style="font-size: 0.9em;">IV: {data['IV']:.2f}%</div>
            <div style="font-size: 0.9em;">Delta: {data['Delta']:.4f}</div>
            <div style="font-size: 0.8em; color: #666;">
                Gamma: {data['Gamma']:.4f} | Theta: {data['Theta']:.2f} | Vega: {data['Vega']:.4f}
            </div>
        </div>
        """

    def display_pcr_metric(label, value, signal=""):
        pcr_color = get_pcr_color(value)
        return f"""
        <div style="background: linear-gradient(90deg, {pcr_color}15, transparent); padding: 8px; border-radius: 6px; border-left: 3px solid {pcr_color}; margin-bottom: 8px;">
            <div style="font-weight: bold; margin-bottom: 4px;">{label}</div>
            <div style="color: {pcr_color}; font-weight: bold; font-size: 1.1em;">
                {value:.3f} {f'({signal})' if signal else ''}
            </div>
        </div>
        """
    
    with left:
        st.markdown("### Calls (CE)")
        
        st.markdown("**ITM (below spot)**")
        ce_itm_oi_color = get_change_color(bucket_summary['CE_ITM']['OI'])
        ce_itm_chgoi_color = get_change_color(bucket_summary['CE_ITM']['ChgOI'])
        st.markdown(display_bucket_stats(bucket_summary['CE_ITM'], "ITM", ce_itm_oi_color, ce_itm_chgoi_color), unsafe_allow_html=True)
        
        st.markdown("**OTM (above spot)**")
        ce_otm_oi_color = get_change_color(bucket_summary['CE_OTM']['OI'])
        ce_otm_chgoi_color = get_change_color(bucket_summary['CE_OTM']['ChgOI'])
        st.markdown(display_bucket_stats(bucket_summary['CE_OTM'], "OTM", ce_otm_oi_color, ce_otm_chgoi_color), unsafe_allow_html=True)
    
    with middle:
        st.markdown("### Puts (PE)")
        
        st.markdown("**ITM (above spot)**")
        pe_itm_oi_color = get_change_color(bucket_summary['PE_ITM']['OI'])
        pe_itm_chgoi_color = get_change_color(bucket_summary['PE_ITM']['ChgOI'])
        st.markdown(display_bucket_stats(bucket_summary['PE_ITM'], "ITM", pe_itm_oi_color, pe_itm_chgoi_color), unsafe_allow_html=True)
        
        st.markdown("**OTM (below spot)**")
        pe_otm_oi_color = get_change_color(bucket_summary['PE_OTM']['OI'])
        pe_otm_chgoi_color = get_change_color(bucket_summary['PE_OTM']['ChgOI'])
        st.markdown(display_bucket_stats(bucket_summary['PE_OTM'], "OTM", pe_otm_oi_color, pe_otm_chgoi_color), unsafe_allow_html=True)
    
    with right:
        st.markdown("### PCR Analysis")
        
        st.markdown("**Open Interest PCR**")
        st.markdown(display_pcr_metric(
            "Overall", 
            pcr_data['OVERALL_PCR_OI'],
            get_pcr_signal(pcr_data['OVERALL_PCR_OI'], 'OI')
        ), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(display_pcr_metric("ITM", pcr_data['ITM_PCR_OI']), unsafe_allow_html=True)
        with col2:
            st.markdown(display_pcr_metric("OTM", pcr_data['OTM_PCR_OI']), unsafe_allow_html=True)
        
        st.markdown("**Change in OI PCR**")
        st.markdown(display_pcr_metric("Overall", pcr_data['OVERALL_PCR_CHGOI']), unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(display_pcr_metric("ITM", pcr_data['ITM_PCR_CHGOI']), unsafe_allow_html=True)
        with col2:
            st.markdown(display_pcr_metric("OTM", pcr_data['OTM_PCR_CHGOI']), unsafe_allow_html=True)
        
        st.markdown("**Volume PCR**")
        st.markdown(display_pcr_metric("Overall", pcr_data['OVERALL_PCR_VOLUME']), unsafe_allow_html=True)

def display_sentiment_analysis(sentiment_analysis, symbol):
    """Display comprehensive sentiment analysis"""
    st.markdown("---")
    st.subheader("Market Sentiment Analysis")
    
    # Sentiment colors
    sentiment_colors = {
        "STRONG BULLISH": "#4caf50",
        "BULLISH": "#8bc34a", 
        "BULLISH BIAS": "#cddc39",
        "NEUTRAL": "#6b7280",
        "BEARISH BIAS": "#ff9800",
        "BEARISH": "#ff5722",
        "STRONG BEARISH": "#f44336"
    }
    
    sentiment_color = sentiment_colors.get(sentiment_analysis["sentiment"], "#6b7280")
    
    # Main sentiment display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {sentiment_color}15 0%, {sentiment_color}05 100%);
                border: 2px solid {sentiment_color};
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="color: {sentiment_color}; margin: 0;">
                    {sentiment_analysis["sentiment"]}
                </h2>
                <p style="margin: 5px 0; color: {sentiment_color}; font-weight: 600;">
                    Confidence: {sentiment_analysis["confidence"]}
                </p>
            </div>
            <div style="text-align: center; padding: 15px; border-radius: 10px; 
                        background: #f0f0f0; border: 2px solid {sentiment_color};">
                <h3 style="color: {sentiment_color}; margin: 0;">
                    {sentiment_analysis["final_score"]:+.1f}
                </h3>
                <small style="color: #666; font-weight: 600;">Sentiment Score</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Component breakdown
    st.markdown("### Score Breakdown")
    for component, score in sentiment_analysis["component_scores"].items():
        component_name = component.replace('_', ' ').title()
        
        bar_color = "#4caf50" if score > 0 else "#f44336"
        bar_width = abs(score)
        
        st.markdown(f"""
        <div style="margin: 10px 0; padding: 10px; border-radius: 8px; background-color: #f8f9fa;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <strong>{component_name}</strong>
                <span style="font-weight: 600;">{score:+.1f}</span>
            </div>
            <div style="background-color: #e0e0e0; height: 6px; border-radius: 3px;">
                <div style="background-color: {bar_color}; height: 6px; border-radius: 3px; 
                           width: {bar_width}%; float: {'right' if score < 0 else 'left'};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def format_option_chain_number(x):
    """Format numbers in option chain table with K/M suffix"""
    if not isinstance(x, (int, float)):
        return x
    
    abs_x = abs(x)
    if abs_x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs_x >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.2f}"

def highlight_option_position(val):
    """Return style for option position cells"""
    style_map = {
        'Long Build': ('c8e6c9', '1b5e20'),      # Light green
        'Short Buildup': ('ffcdd2', 'b71c1c'),   # Light red
        'Short Covering': ('bbdefb', '0d47a1'),   # Light blue
        'Long Unwinding': ('ffe0b2', 'e65100'),   # Light orange
    }
    
    if val in style_map:
        bg_color, text_color = style_map[val]
        return f'background-color: #{bg_color}; color: #{text_color}'
    return ''

def display_option_chain_table(table, atm_strike, spot_price):
    """Display option chain table with formatting - Fixed to prevent blank rows"""
    st.subheader("Option Chain Table")
    
    # Create a copy and ensure clean data
    display_df = table.copy()
    
    # Remove any remaining NaN or invalid rows
    display_df = display_df.dropna(subset=['Strike'])
    display_df = display_df[display_df['Strike'] > 0]
    
    if len(display_df) == 0:
        st.warning("No valid data to display in option chain table")
        return
    
    # Convert Strike to integer
    display_df['Strike'] = display_df['Strike'].astype(int)
    
    # Define available columns
    basic_columns = [
        # Call options (CE)
        'CE_OI', 'CE_LTP', 'CE_Change', 'CE_Volume', 'CE_ChgOI', 'CE_Position',
        # Strike price
        'Strike',
        # Put options (PE)
        'PE_Position', 'PE_ChgOI', 'PE_Volume', 'PE_Change', 'PE_LTP', 'PE_OI'
    ]
    
    greek_columns = [
        'CE_Delta', 'CE_Gamma', 'CE_Theta', 'CE_Vega',
        'PE_Delta', 'PE_Gamma', 'PE_Theta', 'PE_Vega'
    ]
    
    iv_columns = ['CE_IV', 'PE_IV']
    
    # Add column selection options in sidebar
    st.sidebar.markdown("### Column Selection")
    
    # IV Selection
    show_iv = st.sidebar.checkbox("Show IV", value=False)
    
    # Greeks Selection
    st.sidebar.markdown("#### Greek Columns")
    show_delta = st.sidebar.checkbox("Show Delta", value=False)
    show_gamma = st.sidebar.checkbox("Show Gamma", value=False)
    show_theta = st.sidebar.checkbox("Show Theta", value=False)
    show_vega = st.sidebar.checkbox("Show Vega", value=False)
    
    # Build column list based on selections
    columns = basic_columns.copy()
    ce_pos_idx = columns.index('CE_Position')
    pe_pos_idx = columns.index('PE_Position')
    
    # Add selected Greek columns
    greek_mapping = [
        (show_delta, ('CE_Delta', 'PE_Delta')),
        (show_gamma, ('CE_Gamma', 'PE_Gamma')),
        (show_theta, ('CE_Theta', 'PE_Theta')),
        (show_vega, ('CE_Vega', 'PE_Vega'))
    ]
    
    for show_greek, (ce_col, pe_col) in greek_mapping:
        if show_greek:
            columns.insert(ce_pos_idx + 1, ce_col)
            columns.insert(pe_pos_idx + 1, pe_col)
            ce_pos_idx += 1
            pe_pos_idx += 1
    
    if show_iv:
        # Insert IV columns after LTP columns
        ce_ltp_idx = columns.index('CE_LTP')
        pe_ltp_idx = columns.index('PE_LTP')
        columns.insert(ce_ltp_idx + 1, 'CE_IV')
        columns.insert(pe_ltp_idx + 1, 'PE_IV')
    
    # Select and reorder columns - ensure all columns exist
    available_columns = [col for col in columns if col in display_df.columns]
    display_df = display_df[available_columns]
    
    # Define numeric column groups
    oi_volume_cols = ['CE_OI', 'PE_OI', 'CE_Volume', 'PE_Volume']
    price_cols = ['CE_LTP', 'PE_LTP']
    change_cols = ['CE_Change', 'PE_Change']
    chgoi_cols = ['CE_ChgOI', 'PE_ChgOI']
    
    # Build Greek columns list based on selections
    greek_cols = []
    if show_delta:
        greek_cols.extend(['CE_Delta', 'PE_Delta'])
    if show_gamma:
        greek_cols.extend(['CE_Gamma', 'PE_Gamma'])
    if show_theta:
        greek_cols.extend(['CE_Theta', 'PE_Theta'])
    if show_vega:
        greek_cols.extend(['CE_Vega', 'PE_Vega'])
    
    iv_cols = ['CE_IV', 'PE_IV'] if show_iv else []
    
    def format_number(x):
        """Format regular numbers with K/M suffix"""
        if not isinstance(x, (int, float)) or pd.isna(x):
            return "0"
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:.2f}K"
        return f"{x:.2f}"
    
    def format_chgoi(x):
        """Format Change in OI numbers with K/M suffix"""
        if not isinstance(x, (int, float)) or pd.isna(x):
            return "0"
        formatted = format_number(abs(x))
        return f"+{formatted}" if x > 0 else f"-{formatted}"
    
    def format_greek_value(x):
        """Format Greek values with high precision"""
        if not isinstance(x, (int, float)) or pd.isna(x):
            return "0.0000"
        return f"{x:.4f}"
    
    def format_iv_value(x):
        """Format IV values as percentage"""
        if not isinstance(x, (int, float)) or pd.isna(x):
            return "0.00"
        return f"{x:.2f}"
    
    # Format numeric columns based on type
    for col in display_df.columns:
        if col in oi_volume_cols or col in price_cols:
            display_df[col] = display_df[col].apply(format_number)
        elif col in change_cols:
            display_df[col] = display_df[col].apply(lambda x: f"+{x:.2f}" if pd.notna(x) and x > 0 else f"{x:.2f}" if pd.notna(x) else "0.00")
        elif col in chgoi_cols:
            display_df[col] = display_df[col].apply(format_chgoi)
        elif col in greek_cols:
            display_df[col] = display_df[col].apply(format_greek_value)
        elif col in iv_cols:
            display_df[col] = display_df[col].apply(format_iv_value)
    
    # Define table styles
    table_styles = [{
        'selector': 'th',
        'props': [
            ('font-size', '12px'),
            ('text-align', 'center'),
            ('background-color', '#f5f5f5'),
            ('color', '#333'),
            ('font-weight', 'bold'),
            ('padding', '5px'),
            ('border', '1px solid #e0e0e0')
        ]
    }, {
        'selector': 'td',
        'props': [('border', '1px solid #e0e0e0')]
    }]

    # Cell properties
    cell_props = {
        'font-size': '12px',
        'text-align': 'right',
        'padding': '5px',
        'border': '1px solid #e0e0e0'
    }
    
    def highlight_atm(row):
        """Highlight ATM row"""
        if row['Strike'] == atm_strike:
            return ['background-color: #fff9c4'] * len(row)  # Light yellow
        return [''] * len(row)
    
    # Style the dataframe
    styled_df = (display_df.style
        # Color code changes
        .applymap(
            lambda x: ('color: #4caf50' if isinstance(x, str) and x.startswith('+') else 
                      'color: #f44336' if isinstance(x, str) and x.startswith('-') else ''),
            subset=[col for col in change_cols if col in display_df.columns]
        )
        # Highlight positions
        .applymap(highlight_option_position, subset=[col for col in ['CE_Position', 'PE_Position'] if col in display_df.columns])
        # Highlight ATM strike
        .apply(highlight_atm, axis=1)
        # Alternate row colors for non-ATM rows
        .apply(
            lambda x: [
                'background-color: #e3f2fd' if x.name % 2 == 0 and x['Strike'] != atm_strike 
                else '' for i in x
            ], 
            axis=1
        )
        # Apply cell properties
        .set_properties(**cell_props)
        # Apply table styles
        .set_table_styles(table_styles)
    )
    
    # Get the clean DataFrame and create fresh styling
    display_df = display_df[display_df['Strike'].notna()]  # Remove rows with NaN Strike
    display_df = display_df.loc[~(display_df == '').all(axis=1)]  # Remove completely empty rows
    
    # Create fresh styling for the cleaned DataFrame
    styled_df = (display_df.style
        # Color code changes
        .applymap(
            lambda x: ('color: #4caf50' if isinstance(x, str) and x.startswith('+') else 
                      'color: #f44336' if isinstance(x, str) and x.startswith('-') else ''),
            subset=[col for col in change_cols if col in display_df.columns]
        )
        # Highlight positions
        .applymap(highlight_option_position, subset=[col for col in ['CE_Position', 'PE_Position'] if col in display_df.columns])
        # Highlight ATM strike
        .apply(highlight_atm, axis=1)
        # Alternate row colors for non-ATM rows
        .apply(
            lambda x: [
                'background-color: #e3f2fd' if x.name % 2 == 0 and x['Strike'] != atm_strike 
                else '' for i in x
            ], 
            axis=1
        )
        # Apply cell properties
        .set_properties(**cell_props)
        # Apply table styles
        .set_table_styles(table_styles)
    )
    
    # Display the table with fixed height to avoid empty space
    actual_height = min(600, len(display_df) * 35 + 50)  # 35px per row + 50px buffer
    st.dataframe(styled_df, use_container_width=True, height=actual_height)

def display_quick_stats(table, atm_strike):
    """Display quick statistics"""
    st.markdown("---")
    st.subheader("Quick Stats")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total CE OI", format_number(table["CE_OI"].sum()))
        
    with col2:
        st.metric("Total PE OI", format_number(table["PE_OI"].sum()))
        
    with col3:
        st.metric("Total CE Volume", format_number(table["CE_Volume"].sum()))
        
    with col4:
        st.metric("Total PE Volume", format_number(table["PE_Volume"].sum()))
        
    with col5:
        st.metric("ATM Strike", f"₹{atm_strike:,.0f}")

def calculate_volatility_skew_analysis(table, spot_price):
    """Fixed Volatility Skew Analysis"""
    st.subheader("Volatility Skew Analysis")
    
    # Find ATM strike
    atm_strike = table.loc[table["Strike"].sub(spot_price).abs().idxmin(), "Strike"]
    atm_iv = table.loc[table["Strike"] == atm_strike, ["CE_IV", "PE_IV"]].mean().mean()
    
    # Calculate skew metrics
    skew_data = []
    
    for _, row in table.iterrows():
        strike = row['Strike']
        ce_iv = row['CE_IV']
        pe_iv = row['PE_IV']
        
        # Calculate moneyness
        moneyness = spot_price / strike
        
        # Calculate skew relative to ATM
        ce_skew = ce_iv - atm_iv
        pe_skew = pe_iv - atm_iv
        
        # Classify by moneyness
        if moneyness > 1.05:
            category = "Deep ITM"
        elif moneyness > 1.02:
            category = "ITM"
        elif 0.98 <= moneyness <= 1.02:
            category = "ATM"
        elif moneyness > 0.95:
            category = "OTM"
        else:
            category = "Deep OTM"
        
        skew_data.append({
            'Strike': strike,
            'Moneyness': moneyness,
            'CE_IV': ce_iv,
            'PE_IV': pe_iv,
            'CE_Skew': ce_skew,
            'PE_Skew': pe_skew,
            'Category': category
        })
    
    skew_df = pd.DataFrame(skew_data)
    
    # Calculate skew metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Risk Reversal
        otm_put_iv = skew_df[skew_df['Moneyness'] < 1]['PE_IV'].mean()
        otm_call_iv = skew_df[skew_df['Moneyness'] > 1]['CE_IV'].mean()
        risk_reversal = otm_put_iv - otm_call_iv
        st.metric("Risk Reversal", f"{risk_reversal:+.2f}%")
    
    with col2:
        # Butterfly
        otm_avg_iv = skew_df[skew_df['Category'] == 'OTM'][['CE_IV', 'PE_IV']].mean().mean()
        butterfly = atm_iv - otm_avg_iv
        st.metric("Butterfly", f"{butterfly:+.2f}%")
    
    with col3:
        # CE Skew slope
        if len(skew_df) > 1:
            ce_skew_slope = np.corrcoef(skew_df['Moneyness'], skew_df['CE_IV'])[0, 1]
            st.metric("CE Skew Slope", f"{ce_skew_slope:.3f}")
        else:
            st.metric("CE Skew Slope", "N/A")
    
    with col4:
        # PE Skew slope
        if len(skew_df) > 1:
            pe_skew_slope = np.corrcoef(skew_df['Moneyness'], skew_df['PE_IV'])[0, 1]
            st.metric("PE Skew Slope", f"{pe_skew_slope:.3f}")
        else:
            st.metric("PE Skew Slope", "N/A")
    
    # Visualization
    if len(skew_df) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IV Smile/Skew
        ax1.plot(skew_df['Strike'], skew_df['CE_IV'], 'g-o', label='Call IV', linewidth=2, markersize=4)
        ax1.plot(skew_df['Strike'], skew_df['PE_IV'], 'r-o', label='Put IV', linewidth=2, markersize=4)
        ax1.axvline(spot_price, color='blue', linestyle='--', alpha=0.7, label=f'Spot: ₹{spot_price}')
        ax1.axvline(atm_strike, color='orange', linestyle='--', alpha=0.7, label=f'ATM: ₹{atm_strike}')
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title('Volatility Smile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Moneyness vs IV
        ax2.scatter(skew_df['Moneyness'], skew_df['CE_IV'], alpha=0.7, color='green', label='Call IV', s=60)
        ax2.scatter(skew_df['Moneyness'], skew_df['PE_IV'], alpha=0.7, color='red', label='Put IV', s=60)
        ax2.axvline(x=1.0, color='blue', linestyle='--', alpha=0.7, label='ATM')
        ax2.set_xlabel('Moneyness (S/K)')
        ax2.set_ylabel('Implied Volatility (%)')
        ax2.set_title('IV vs Moneyness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Skew interpretation
        st.markdown("### Skew Analysis")
        
        if risk_reversal > 2:
            st.info("Strong Put Skew: Market expects downside volatility")
        elif risk_reversal < -2:
            st.info("Call Skew: Market expects upside volatility")
        else:
            st.info("Balanced Skew: No strong directional bias")

def display_gamma_blast_analysis(table, spot_price, gex_df):
    """Display gamma blast analysis with enhanced visualization"""
    try:
        # Calculate market context
        market_context = {
            'regime': calculate_market_regime(None, gex_df, table)
        }
        
        # Call the dynamic function with market context
        blast_signal, blast_direction, reasons, entry_signal, is_post_1_30_pm = detect_gamma_blast(
            table, spot_price, gex_df, None, market_context
        )
    except Exception as e:
        st.error(f"Error in gamma blast detection: {str(e)}")
        return "No Blast", None, [], False, False
    
    # Rest of your existing display code remains the same...
    st.subheader("🎯 Gamma Blast Detection System")
    
    # Your existing visualization code continues here...
    signal_colors = {
        "No Blast": "#6b7280",
        "Gamma Blast Watch": "#f59e0b", 
        "Gamma Blast Setup - Upside": "#10b981",
        "Gamma Blast Setup - Downside": "#ef4444",
        "Gamma Blast Setup - Bidirectional": "#8b5cf6",
        "Gamma Compression": "#06b6d4",
        "Gamma Blast ENTRY SIGNAL - Upside": "#059669",
        "Gamma Blast ENTRY SIGNAL - Downside": "#dc2626",
        "Gamma Blast ENTRY SIGNAL - Bidirectional": "#7c3aed"
    }
    
    signal_color = signal_colors.get(blast_signal, "#6b7280")
    
    # Display main signal
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {signal_color}15 0%, {signal_color}05 100%);
                border: 2px solid {signal_color};
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 15px;">
        <h3 style="color: {signal_color}; margin: 0; text-align: center;">
            {blast_signal}
        </h3>
        {f'<p style="text-align: center; margin: 5px 0; font-weight: 600;">Direction: {blast_direction}</p>' if blast_direction else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Display reasons
    if reasons:
        st.markdown("**Analysis Factors:**")
        for reason in reasons:
            st.markdown(f"• {reason}")
    
    return blast_signal, blast_direction, reasons, entry_signal, is_post_1_30_pm


# Update the main gamma exposure analysis function
def calculate_gamma_exposure_analysis(table, spot_price, gex_df=None):
    """Calculate or display Gamma Exposure and GEX levels with enhanced blast detection"""
    st.subheader("Gamma Exposure Analysis")
    
    # Calculate GEX if not provided
    if gex_df is None:
        gex_data = []
        for _, row in table.iterrows():
            strike = row['Strike']
            ce_gex = row['CE_Gamma'] * row['CE_OI'] * 100 * (spot_price ** 2) * 0.01
            pe_gex = -row['PE_Gamma'] * row['PE_OI'] * 100 * (spot_price ** 2) * 0.01  # Negative for puts
            net_gex = ce_gex + pe_gex
            gex_data.append({
                'Strike': strike,
                'CE_GEX': ce_gex,
                'PE_GEX': pe_gex,
                'Net_GEX': net_gex,
                'Distance': abs(strike - spot_price)
            })
        gex_df = pd.DataFrame(gex_data)
    
    # Calculate total positive and negative GEX
    total_positive_gex = gex_df[gex_df['Net_GEX'] > 0]['Net_GEX'].sum()
    total_negative_gex = abs(gex_df[gex_df['Net_GEX'] < 0]['Net_GEX'].sum())
    
    # Find zero gamma level
    zero_gamma_strike = None
    if len(gex_df) > 0:
        min_gex_idx = gex_df['Net_GEX'].abs().idxmin()
        zero_gamma_strike = gex_df.loc[min_gex_idx, 'Strike']
    
    # Display basic metrics in a 3-column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Positive GEX", format_number(total_positive_gex))
    
    with col2:
        st.metric("Total Negative GEX", format_number(total_negative_gex))
    
    with col3:
        net_market_gex = total_positive_gex - total_negative_gex
        st.metric("Net Market GEX", format_number(net_market_gex))
        
    # Display zero gamma level
    if zero_gamma_strike:
        st.metric("Zero Gamma Level", f"₹{zero_gamma_strike:,.2f}")
    
    # GEX Chart
    if len(gex_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['red' if gex < 0 else 'green' for gex in gex_df['Net_GEX']]
        
        ax.bar(gex_df['Strike'], gex_df['Net_GEX'], color=colors, alpha=0.7, width=25)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=spot_price, color='blue', linestyle='--', linewidth=2, label=f'Spot: ₹{spot_price}')
        
        if zero_gamma_strike:
            ax.axvline(x=zero_gamma_strike, color='orange', linestyle='--', linewidth=2, 
                      label=f'Zero Gamma: ₹{zero_gamma_strike}')
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Gamma Exposure (₹ Millions)')
        ax.set_title('Gamma Exposure by Strike')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Market implications
        st.markdown("### Gamma Exposure Implications")
        
        if net_market_gex > 0:
            st.success("Positive Net GEX: Market makers will sell into rallies and buy dips (stabilizing)")
        else:
            st.warning("Negative Net GEX: Market makers will buy rallies and sell dips (destabilizing)")
    
    try:
        # Enhanced Gamma Blast Detection with error handling
        blast_signal, blast_direction, reasons, entry_signal, is_post_1_30_pm = display_gamma_blast_analysis(table, spot_price, gex_df)
    except Exception as e:
        st.error(f"Could not complete gamma blast analysis: {str(e)}")
    
    return gex_df

# 2. ADD this helper function if not already present:
def calculate_market_regime(historical_data=None, gex_df=None, table=None):
    """Determine current market volatility regime"""
    import numpy as np
    
    if historical_data and 'vix' in historical_data:
        vix = historical_data['vix']
        if vix > 25:
            return 'high_vol'
        elif vix < 15:
            return 'low_vol'
        else:
            return 'normal'
    
    if table is not None:
        iv_values = []
        for col in ['CE_IV', 'PE_IV']:
            vals = table[col].dropna()
            vals = vals[vals > 0]
            iv_values.extend(vals.tolist())
        
        if iv_values:
            avg_iv = np.mean(iv_values)
            if avg_iv > 25:
                return 'high_vol'
            elif avg_iv < 15:
                return 'low_vol'
            else:
                return 'normal'
    
    return 'normal'

def detect_gamma_blast(table, spot_price, gex_df, historical_data=None, market_context=None):
    """
    Dynamic Gamma Blast Detection with adaptive thresholds
    """
    signal = "No Blast"
    direction = None
    reasons = []
    entry_signal = False

    # Get current IST time
    from datetime import datetime, time as dt_time
    import pytz
    import numpy as np
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist)
    
    # Create time objects for comparison
    current_time = dt_time(current_time_ist.hour, current_time_ist.minute)
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    entry_threshold = dt_time(13, 30)  # 1:30 PM
    
    # Convert times to minutes since midnight for calculations
    current_minutes = current_time.hour * 60 + current_time.minute
    open_minutes = market_open.hour * 60 + market_open.minute
    close_minutes = market_close.hour * 60 + market_close.minute
    
    # Calculate time factor
    time_factor = min((current_minutes - open_minutes) / (close_minutes - open_minutes), 1.0)
    is_entry_time = current_time >= entry_threshold
    
    # 1. Find ATM strike with maximum OI (dynamic ATM definition)
    table_copy = table.copy()
    table_copy['Total_OI'] = table_copy['CE_OI'] + table_copy['PE_OI']
    table_copy['distance_to_spot'] = abs(table_copy['Strike'] - spot_price)
    
    # Dynamic ATM range based on current volatility
    if historical_data and 'vix' in historical_data:
        current_vix = historical_data['vix']
        atm_range_pct = max(0.005, min(0.02, current_vix / 100 * 0.1))
    else:
        iv_values = table_copy[['CE_IV', 'PE_IV']].values.flatten()
        iv_values = iv_values[iv_values > 0]
        avg_iv = np.mean(iv_values) if len(iv_values) > 0 else 20
        atm_range_pct = max(0.005, min(0.02, avg_iv / 100 * 0.05))
    
    atm_candidates = table_copy[table_copy['distance_to_spot'] <= spot_price * atm_range_pct]
    
    if atm_candidates.empty:
        return signal, direction, reasons, entry_signal, is_entry_time

    max_oi_idx = atm_candidates['Total_OI'].idxmax()
    atm_strike = atm_candidates.loc[max_oi_idx, 'Strike']
    atm_row = atm_candidates.loc[max_oi_idx]
    
    spot_atm_distance_pct = abs(spot_price - atm_strike) / spot_price * 100
    proximity_threshold = atm_range_pct * 50
    is_near_max_oi_atm = spot_atm_distance_pct < proximity_threshold
    
    if not is_near_max_oi_atm:
        return signal, direction, reasons, entry_signal, is_entry_time

    reasons.append(f"Spot ({spot_price}) near max OI ATM ({atm_strike}) - Distance: {spot_atm_distance_pct:.2f}%")

    # 2. Extract OI metrics
    ce_oi = atm_row['CE_OI']
    pe_oi = atm_row['PE_OI']
    total_atm_oi = atm_row['Total_OI']
    ce_chg_oi = atm_row['CE_ChgOI'] 
    pe_chg_oi = atm_row['PE_ChgOI']
    
    # 3. Dynamic GEX analysis
    gex_at_atm = gex_df[gex_df['Strike'] == atm_strike]
    if gex_at_atm.empty:
        return signal, direction, reasons, entry_signal, is_entry_time
    
    net_gex_atm = gex_at_atm['Net_GEX'].iloc[0]
    
    # Statistical GEX thresholds (dynamic percentiles)
    gex_values = gex_df['Net_GEX'].dropna()
    gex_25th = gex_values.quantile(0.25)
    gex_10th = gex_values.quantile(0.10) 
    gex_5th = gex_values.quantile(0.05)
    
    # Market regime adjustment
    if market_context and 'regime' in market_context:
        regime = market_context['regime']
        if regime == 'high_vol':
            gex_negative_threshold = gex_25th
            gex_sharp_threshold = gex_10th
        elif regime == 'low_vol':
            gex_negative_threshold = gex_10th
            gex_sharp_threshold = gex_5th
        else:  # normal regime
            gex_negative_threshold = (gex_25th + gex_10th) / 2
            gex_sharp_threshold = (gex_10th + gex_5th) / 2
    else:
        gex_negative_threshold = gex_25th
        gex_sharp_threshold = gex_10th
    
    # Time-based sensitivity adjustment
    late_day_multiplier = 1 + (time_factor * 0.3)
    gex_negative_threshold *= late_day_multiplier
    gex_sharp_threshold *= late_day_multiplier
    
    # GEX conditions
    gex_negative = net_gex_atm < gex_negative_threshold
    gex_sharply_negative = net_gex_atm < gex_sharp_threshold
    
    if gex_negative:
        reasons.append(f"GEX negative: {net_gex_atm/1000000:.1f}M (threshold: {gex_negative_threshold/1000000:.1f}M)")
    if gex_sharply_negative:
        reasons.append(f"GEX sharply negative: {net_gex_atm/1000000:.1f}M (threshold: {gex_sharp_threshold/1000000:.1f}M)")
    
    # 4. Dynamic IV analysis
    ce_iv = atm_row['CE_IV']
    pe_iv = atm_row['PE_IV']
    atm_iv = (ce_iv + pe_iv) / 2
    
    # Calculate IV percentiles across all strikes
    all_iv_values = []
    for col in ['CE_IV', 'PE_IV']:
        iv_vals = table_copy[col].dropna()
        iv_vals = iv_vals[iv_vals > 0]
        all_iv_values.extend(iv_vals.tolist())
    
    if all_iv_values:
        iv_25th = np.percentile(all_iv_values, 25)
        iv_50th = np.percentile(all_iv_values, 50)
        iv_75th = np.percentile(all_iv_values, 75)
        
        iv_low_threshold = iv_25th
        
        nearby_strikes = table_copy[
            (table_copy['Strike'] != atm_strike) &
            (table_copy['distance_to_spot'] <= spot_price * (atm_range_pct * 2))
        ]
        
        if not nearby_strikes.empty:
            nearby_iv_values = []
            for col in ['CE_IV', 'PE_IV']:
                vals = nearby_strikes[col].dropna()
                vals = vals[vals > 0]
                nearby_iv_values.extend(vals.tolist())
            
            if nearby_iv_values:
                avg_nearby_iv = np.mean(nearby_iv_values)
                iv_skew = atm_iv - avg_nearby_iv
                skew_threshold = -(iv_75th - iv_50th) / 2
            else:
                iv_skew = 0
                skew_threshold = -1.0
        else:
            iv_skew = 0
            skew_threshold = -1.0
    else:
        iv_low_threshold = 18
        iv_skew = 0
        skew_threshold = -1.0
    
    # IV conditions
    iv_collapsing = iv_skew < skew_threshold
    atm_iv_low = atm_iv < iv_low_threshold
    
    if iv_collapsing:
        reasons.append(f"ATM IV collapsing: {iv_skew:.1f}% (threshold: {skew_threshold:.1f}%)")
    if atm_iv_low:
        reasons.append(f"ATM IV low: {atm_iv:.1f}% (threshold: {iv_low_threshold:.1f}%)")
    
    # 5. Dynamic OI unwinding detection
    ce_oi_changes = table_copy['CE_ChgOI'].dropna()
    pe_oi_changes = table_copy['PE_ChgOI'].dropna()
    
    if len(ce_oi_changes) > 0:
        ce_unwind_threshold = ce_oi_changes.quantile(0.25)
    else:
        ce_unwind_threshold = -ce_oi * 0.03
        
    if len(pe_oi_changes) > 0:
        pe_unwind_threshold = pe_oi_changes.quantile(0.25)
    else:
        pe_unwind_threshold = -pe_oi * 0.03
    
    total_unwind_pct = max(0.03, min(0.10, time_factor * 0.07))
    
    ce_unwinding = ce_chg_oi < ce_unwind_threshold
    pe_unwinding = pe_chg_oi < pe_unwind_threshold
    total_unwinding = (abs(ce_chg_oi) + abs(pe_chg_oi)) > total_atm_oi * total_unwind_pct
    
    unwinding_detected = ce_unwinding or pe_unwinding or total_unwinding
    
    if ce_unwinding:
        reasons.append(f"CE OI unwinding: {ce_chg_oi:,.0f} (threshold: {ce_unwind_threshold:.0f})")
    if pe_unwinding:
        reasons.append(f"PE OI unwinding: {pe_chg_oi:,.0f} (threshold: {pe_unwind_threshold:.0f})")
    
    # 6. DYNAMIC BLAST SIGNAL DETERMINATION
    gex_score = 0
    if gex_negative: gex_score += 1
    if gex_sharply_negative: gex_score += 2
    
    iv_score = 0
    if iv_collapsing: iv_score += 2
    if atm_iv_low: iv_score += 1
    
    oi_score = 0
    if unwinding_detected: oi_score += 2
    if ce_unwinding and pe_unwinding: oi_score += 1
    
    base_threshold = 3
    if market_context and market_context.get('regime') == 'high_vol':
        base_threshold = 2
    elif time_factor > 0.7:
        base_threshold = 2
    
    total_score = gex_score + iv_score + oi_score
    
    if total_score >= base_threshold + 2:
        ce_unwinding_strength = abs(ce_chg_oi) / max(ce_oi, 1) if ce_oi > 0 else 0
        pe_unwinding_strength = abs(pe_chg_oi) / max(pe_oi, 1) if pe_oi > 0 else 0
        
        directional_threshold = 1.3 + (time_factor * 0.4)
        
        if ce_unwinding_strength > pe_unwinding_strength * directional_threshold:
            direction = "UPSIDE_BREAKOUT"
            signal = "Gamma Blast Setup - Upside"
        elif pe_unwinding_strength > ce_unwinding_strength * directional_threshold:
            direction = "DOWNSIDE_BREAKOUT"  
            signal = "Gamma Blast Setup - Downside"
        else:
            direction = "BIDIRECTIONAL"
            signal = "Gamma Blast Setup - Bidirectional"
        
        if is_entry_time:
            entry_signal = True
            signal = signal.replace("Setup", "ENTRY SIGNAL")
            # Format time in 12-hour format with AM/PM
            entry_time_str = entry_threshold.strftime("%I:%M %p")
            reasons.append(f"POST {entry_time_str} - ENTRY CONDITIONS MET")
        else:
            # Format time in 12-hour format with AM/PM
            entry_time_str = entry_threshold.strftime("%I:%M %p")
            reasons.append(f"POST {entry_time_str} - ENTRY CONDITIONS MET")
    elif total_score >= base_threshold:
        signal = "Gamma Blast Watch"
        direction = "MONITOR"
        reasons.append(f"Partial conditions met (Score: {total_score}/{base_threshold+2}) - monitor closely")
    
    elif total_score >= base_threshold - 1:
        signal = "Gamma Compression"
        direction = "SETUP"
        reasons.append(f"Early setup conditions (Score: {total_score}/{base_threshold+2}) - watch for strengthening")

    return signal, direction, reasons, entry_signal, is_entry_time


def implement_vix_like_index(table, spot_price, time_to_expiry):
    """Calculate VIX-like implied volatility index"""
    st.subheader("Custom Volatility Index (VIX-like)")
    
    iv_weights = []
    total_weight = 0
    
    for _, row in table.iterrows():
        strike = row['Strike']
        ce_iv = row['CE_IV']
        pe_iv = row['PE_IV']
        ce_oi = row['CE_OI']
        pe_oi = row['PE_OI']
        
        distance = abs(strike - spot_price) / spot_price
        if distance <= 0.2:
            if ce_oi + pe_oi > 0:
                weighted_iv = (ce_iv * ce_oi + pe_iv * pe_oi) / (ce_oi + pe_oi)
                weight = ce_oi + pe_oi
                
                iv_weights.append({
                    'strike': strike,
                    'iv': weighted_iv,
                    'weight': weight,
                    'distance': distance
                })
                total_weight += weight
    
    if total_weight > 0:
        vix_like_value = sum(item['iv'] * item['weight'] for item in iv_weights) / total_weight
        
        if time_to_expiry < 1:
            vix_like_value = vix_like_value * np.sqrt(365 * time_to_expiry)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Custom VIX", f"{vix_like_value:.2f}")
        
        with col2:
            vix_percentile = "High" if vix_like_value > 25 else "Normal" if vix_like_value > 15 else "Low"
            st.metric("Volatility Regime", vix_percentile)
        
        with col3:
            fear_level = "Extreme Fear" if vix_like_value > 35 else "Fear" if vix_like_value > 25 else "Normal" if vix_like_value > 15 else "Complacency"
            st.metric("Market Sentiment", fear_level)
        
        return vix_like_value
    else:
        st.warning("Insufficient data for VIX calculation")
    
    return None

def display_support_resistance_levels(table, spot_price):
    """Identify support and resistance levels using OI, Volume and GEX data"""
    st.subheader("Support & Resistance Levels (OI + Volume + GEX Weighted)")
    
    # Get GEX data from session state if available
    table_copy = table.copy()
    gex_df = None
    if 'current_gex_data' in st.session_state:
        gex_df = pd.DataFrame(st.session_state.current_gex_data)
        # Merge GEX data with table
        table_copy = table_copy.merge(gex_df[['Strike', 'CE_GEX', 'PE_GEX', 'Net_GEX']], 
                                    on='Strike', how='left')
    
    # Compute base strength using OI and Volume
    table_copy['CE_Strength'] = table_copy['CE_OI'] * np.log1p(table_copy['CE_Volume'])
    table_copy['PE_Strength'] = table_copy['PE_OI'] * np.log1p(table_copy['PE_Volume'])
    
    # Add GEX component if available
    if gex_df is not None:
        # Scale GEX to be comparable with OI*Volume strength
        gex_scale = (table_copy['CE_Strength'].mean() + table_copy['PE_Strength'].mean()) / (abs(gex_df['Net_GEX']).mean() + 1e-10)
        
        # Add scaled GEX to strength - positive GEX adds to CE_Strength (resistance)
        # negative GEX adds to PE_Strength (support)
        table_copy['CE_Strength'] += np.where(table_copy['Net_GEX'] > 0, 
                                            abs(table_copy['Net_GEX']) * gex_scale, 0)
        table_copy['PE_Strength'] += np.where(table_copy['Net_GEX'] < 0, 
                                            abs(table_copy['Net_GEX']) * gex_scale, 0)
    
    # Calculate total strengths for normalization
    total_ce_strength = table_copy['CE_Strength'].sum()
    total_pe_strength = table_copy['PE_Strength'].sum()
    
    # Identify significant levels
    significant_levels = []
    
    # Calculate relative strength at each strike
    table_copy['CE_Relative_Strength'] = table_copy['CE_Strength'] / total_ce_strength if total_ce_strength > 0 else 0
    table_copy['PE_Relative_Strength'] = table_copy['PE_Strength'] / total_pe_strength if total_pe_strength > 0 else 0
    
    # Add distance from spot for weighting
    table_copy['Distance'] = abs(table_copy['Strike'] - spot_price)
    max_distance = table_copy['Distance'].max()
    table_copy['Distance_Weight'] = 1 - (table_copy['Distance'] / max_distance)
    
    # Calculate strength thresholds
    ce_threshold = table_copy['CE_Relative_Strength'].mean() + table_copy['CE_Relative_Strength'].std()
    pe_threshold = table_copy['PE_Relative_Strength'].mean() + table_copy['PE_Relative_Strength'].std()
    
    for _, row in table_copy.iterrows():
        strike = row['Strike']
        distance_weight = row['Distance_Weight']
        
        # Calculate weighted relative strengths
        ce_weighted_strength = row['CE_Relative_Strength'] * distance_weight
        pe_weighted_strength = row['PE_Relative_Strength'] * distance_weight
        
        # Add GEX influence if available
        if gex_df is not None and 'Net_GEX' in row:
            gex_influence = abs(row['Net_GEX']) * gex_scale * distance_weight
            if row['Net_GEX'] > 0:
                ce_weighted_strength += gex_influence
            else:
                pe_weighted_strength += gex_influence
        
        # Determine level type and strength
        # Check for resistance levels (strong call activity above spot)
        if strike > spot_price and ce_weighted_strength > ce_threshold:
            level_type = "Resistance"
            strength_pct = ce_weighted_strength * 100
            strength = "Strong" if strength_pct > 1.5 * ce_threshold * 100 else "Moderate"
            # Add resistance level
            significant_levels.append({
                'Level': strike,
                'Type': level_type,
                'Strength': strength,
                'Distance%': abs(strike - spot_price) / spot_price * 100,
                'OI_Volume_Weight': row['CE_Strength'],
                'GEX_Impact': row.get('Net_GEX', 0) if gex_df is not None else 0,
                'Total_Weight': ce_weighted_strength
            })
            
        # Check for support levels (strong put activity below spot)
        if strike < spot_price and pe_weighted_strength > pe_threshold:
            level_type = "Support"
            strength_pct = pe_weighted_strength * 100
            strength = "Strong" if strength_pct > 1.5 * pe_threshold * 100 else "Moderate"
            # Add support level
            significant_levels.append({
                'Level': strike,
                'Type': level_type,
                'Strength': strength,
                'Distance%': abs(strike - spot_price) / spot_price * 100,
                'OI_Volume_Weight': row['PE_Strength'],
                'GEX_Impact': row.get('Net_GEX', 0) if gex_df is not None else 0,
                'Total_Weight': pe_weighted_strength
            })
    
    if significant_levels:
        levels_df = pd.DataFrame(significant_levels)
        # Remove duplicate levels and keep the one with the highest weight
        levels_df = levels_df.sort_values('Total_Weight', ascending=False)
        levels_df = levels_df.drop_duplicates(subset=['Level', 'Type'], keep='first')
        levels_df = levels_df.sort_values(['Type', 'Total_Weight'], ascending=[True, False])
        
        col1, col2 = st.columns(2)
        
        # Display support levels
        with col1:
            support_levels = levels_df[levels_df['Type'] == 'Support'].head(3)
            st.markdown("**Key Support Levels**")
            if not support_levels.empty:
                for _, level in support_levels.iterrows():
                    gex_info = f" | GEX: {format_number(level['GEX_Impact'])}" if gex_df is not None else ""
                    st.write(f"• ₹{level['Level']:,.0f} ({level['Strength']}) - {level['Distance%']:.1f}% below spot{gex_info}")
            else:
                st.write("No significant support levels found")
        
        # Display resistance levels
        with col2:
            resistance_levels = levels_df[levels_df['Type'] == 'Resistance'].head(3)
            st.markdown("**Key Resistance Levels**")
            if not resistance_levels.empty:
                for _, level in resistance_levels.iterrows():
                    gex_info = f" | GEX: {format_number(level['GEX_Impact'])}" if gex_df is not None else ""
                    st.write(f"• ₹{level['Level']:,.0f} ({level['Strength']}) - {level['Distance%']:.1f}% above spot{gex_info}")
            else:
                st.write("No significant resistance levels found")
    else:
        st.info("No significant support/resistance levels identified")

def calculate_put_call_parity_analysis(table, atm_strike):
    """Put-Call Parity Analysis for OTM equidistant pairs"""
    st.subheader("Put-Call Parity Analysis (OTM Equidistant Pairs)")
    
    parity_pairs = []
    otm_calls = table[table["Strike"] > atm_strike].copy()
    otm_puts = table[table["Strike"] < atm_strike].copy()

    for _, call_row in otm_calls.iterrows():
        call_strike = call_row['Strike']
        call_distance = call_strike - atm_strike
        target_put_strike = atm_strike - call_distance

        put_row = otm_puts[otm_puts['Strike'] == target_put_strike]
        if not put_row.empty:
            put_row = put_row.iloc[0]

            ce_ltp = call_row['CE_LTP']
            pe_ltp = put_row['PE_LTP']
            ce_iv = call_row['CE_IV']
            pe_iv = put_row['PE_IV']

            actual_diff = ce_ltp - pe_ltp
            deviation_pct = (actual_diff / pe_ltp) * 100 if pe_ltp > 0 else None

            mispricing = "Overvalued" if actual_diff > 0 else "Undervalued" if actual_diff < 0 else "Fair"

            parity_pairs.append({
                'Distance': int(call_distance),
                'Call_Strike': int(call_strike),
                'Put_Strike': int(target_put_strike),
                'Call_Price': f"{ce_ltp:,.2f}",
                'Put_Price': f"{pe_ltp:,.2f}",
                'Call_IV': f"{ce_iv:.1f}%",
                'Put_IV': f"{pe_iv:.1f}%",
                'Deviation': f"{deviation_pct:.2f}%" if deviation_pct else "N/A",
                'Mispricing': mispricing
            })

    if parity_pairs:
        parity_df = pd.DataFrame(parity_pairs)
        
        def highlight_mispricing(val):
            if val == "Overvalued":
                return 'background-color: #ffcdd2; color: #b71c1c'
            elif val == "Undervalued":
                return 'background-color: #c8e6c9; color: #1b5e20'
            else:
                return 'background-color: #fff9c4; color: #f57f17'

        styled = parity_df.style.applymap(highlight_mispricing, subset=['Mispricing'])
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("No equidistant OTM pairs found for parity analysis.")

# Move the bucket summary display code inside the main function
def display_bucket_summary(bucket_summary, left_col, middle_col):
    def format_bucket_stats(bucket_data, oi_color, chgoi_color):
        return f"""
        <div style="background: linear-gradient(90deg, {oi_color}15, transparent); padding: 10px; border-radius: 8px; border-left: 4px solid {oi_color};">
            <div style="color: {oi_color}; font-weight: bold;">OI: {format_number(bucket_data['OI'])}</div>
            <div style="color: {chgoi_color}; font-weight: bold;">ChgOI: {'+' if bucket_data['ChgOI'] >= 0 else ''}{format_number(bucket_data['ChgOI'])}</div>
            <div>Volume: {format_number(bucket_data['Volume'])}</div>
            <div style="font-size: 0.9em;">IV: {bucket_data['IV']:.2f}%</div>
            <div style="font-size: 0.9em;">Delta: {bucket_data['Delta']:.4f}</div>
        </div>
        """

    with left_col:
        st.markdown("### Calls (CE)")
        
        # ITM Calls
        st.markdown("**ITM (below spot)**")
        ce_itm_oi_color = get_change_color(bucket_summary['CE_ITM']['OI'])
        ce_itm_chgoi_color = get_change_color(bucket_summary['CE_ITM']['ChgOI'])
        st.markdown(format_bucket_stats(bucket_summary['CE_ITM'], ce_itm_oi_color, ce_itm_chgoi_color), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # OTM Calls
        st.markdown("**OTM (above spot)**")
        ce_otm_oi_color = get_change_color(bucket_summary['CE_OTM']['OI'])
        ce_otm_chgoi_color = get_change_color(bucket_summary['CE_OTM']['ChgOI'])
        st.markdown(format_bucket_stats(bucket_summary['CE_OTM'], ce_otm_oi_color, ce_otm_chgoi_color), unsafe_allow_html=True)
    
    with middle_col:
        st.markdown("### Puts (PE)")
        
        # ITM Puts
        st.markdown("**ITM (above spot)**")
        pe_itm_oi_color = get_change_color(bucket_summary['PE_ITM']['OI'])
        pe_itm_chgoi_color = get_change_color(bucket_summary['PE_ITM']['ChgOI'])
        st.markdown(format_bucket_stats(bucket_summary['PE_ITM'], pe_itm_oi_color, pe_itm_chgoi_color), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # OTM Puts
        st.markdown("**OTM (below spot)**")
        pe_otm_oi_color = get_change_color(bucket_summary['PE_OTM']['OI'])
        pe_otm_chgoi_color = get_change_color(bucket_summary['PE_OTM']['ChgOI'])
        st.markdown(format_bucket_stats(bucket_summary['PE_OTM'], pe_otm_oi_color, pe_otm_chgoi_color), unsafe_allow_html=True)

# Format and display bucket summary for option chain data
def get_bucket_stats_html(data, oi_color, chgoi_color):
    return f"""<div style='background: linear-gradient(90deg, {oi_color}15, transparent); padding: 10px; border-radius: 8px; border-left: 4px solid {oi_color};'>
        <div style='color: {oi_color}; font-weight: bold;'>OI: {format_number(data["OI"])}</div>
        <div style='color: {chgoi_color}; font-weight: bold;'>ChgOI: {'+' if data["ChgOI"] >= 0 else ''}{format_number(data["ChgOI"])}</div>
        <div>Volume: {format_number(data["Volume"])}</div>
        <div style='font-size: 0.9em;'>IV: {data["IV"]:.2f}%</div>
        <div style='font-size: 0.9em;'>Delta: {data["Delta"]:.4f}</div>
    </div>"""

def display_bucket_summary(bucket_summary, left_col, middle_col):
    with left_col:
        st.markdown("### Calls (CE)")
        
        st.markdown("**ITM (below spot)**")
        ce_itm_oi_color = get_change_color(bucket_summary['CE_ITM']['OI'])
        ce_itm_chgoi_color = get_change_color(bucket_summary['CE_ITM']['ChgOI'])
        st.markdown(get_bucket_stats_html(bucket_summary['CE_ITM'], ce_itm_oi_color, ce_itm_chgoi_color), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**OTM (above spot)**")
        ce_otm_oi_color = get_change_color(bucket_summary['CE_OTM']['OI'])
        ce_otm_chgoi_color = get_change_color(bucket_summary['CE_OTM']['ChgOI'])
        st.markdown(get_bucket_stats_html(bucket_summary['CE_OTM'], ce_otm_oi_color, ce_otm_chgoi_color), unsafe_allow_html=True)
    
    with middle_col:
        st.markdown("### Puts (PE)")
        
        st.markdown("**ITM (above spot)**")
        pe_itm_oi_color = get_change_color(bucket_summary['PE_ITM']['OI'])
        pe_itm_chgoi_color = get_change_color(bucket_summary['PE_ITM']['ChgOI'])
        st.markdown(get_bucket_stats_html(bucket_summary['PE_ITM'], pe_itm_oi_color, pe_itm_chgoi_color), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**OTM (below spot)**")
        pe_otm_oi_color = get_change_color(bucket_summary['PE_OTM']['OI'])
        pe_otm_chgoi_color = get_change_color(bucket_summary['PE_OTM']['ChgOI'])
        st.markdown(get_bucket_stats_html(bucket_summary['PE_OTM'], pe_otm_oi_color, pe_otm_chgoi_color), unsafe_allow_html=True)

# Only run the app when this file is run directly
if __name__ == "__main__":
    main()
       