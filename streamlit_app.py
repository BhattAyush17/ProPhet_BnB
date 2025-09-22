import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.scraper import scrape_catalog
from src.downloader import download_dataset
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_price_model, cluster_hosts
from src.recommendation import build_recommendation_scores, filter_by_preferences
from src.visualizations import parallel_recommendations, radar_for_listing
from src.ui_theme import inject_base_css
from src.data_sources.direct_csv_url_source import DirectCSVURLSource
from src.data_sources.external_site_source import ExternalSiteSource
from src.metrics import compute_metrics

st.set_page_config(page_title="ProPhet-BnB", layout="wide")
inject_base_css()

st.markdown("""
    <style>
    body, .stApp {
        background: url('https://tse1.mm.bing.net/th/id/OIP.z7pMD9WSe9JepI42_5EwrAHaE8?rs=1&pid=ImgDetMain&o=7&rm=3') no-repeat center center fixed !important;
        background-size: cover !important;
    }
    .main-card {
        background: rgba(255,255,255,0.82);
        border-radius: 18px;
        margin: 48px auto 0 auto;
        padding: 32px 40px 28px 40px;
        box-shadow: 0 4px 24px #1113;
        max-width: 730px;
        text-align: center;
    }
    h1 {
        font-size: 2.6em;
        color: #222;
        font-weight: 900;
        margin-bottom: 0.15em;
    }
    .subtitle {
        color: #3466a3;
        font-size: 1.17em;
        margin-bottom: 0.8em;
        font-weight: 600;
    }
    .simple-steps {
        font-size: 1.13em;
        color: #444;
        font-weight: 500;
        margin-bottom: 1.2em;
    }
    .data-link-info {
        background: rgba(240,245,255,0.95);
        border-radius: 14px;
        padding: 22px 16px;
        font-size: 1.13em;
        color: #3466a3;
        font-weight: 600;
        margin-top: 18px;
        box-shadow: 0 2px 14px #1113;
    }
    .stButton>button, .stDownloadButton>button {
        background: #3466a3 !important;
        color: #fff !important;
        border-radius: 6px !important;
        font-size: 1.07em !important;
        font-weight: 600 !important;
        padding: 0.45em 1.5em !important;
        margin: 10px 0 !important;
        border: none !important;
        box-shadow: 0 1px 2px #3466a380;
    }
    .stButton>button:disabled {
        background-color: #cccccc !important;
        color: #888 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #f6f8fa;
        border-radius: 8px;
        padding: 0.5em;
        color: #222 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #3466a3 !important;
        font-weight: bold !important;
        background: #e3eaf5 !important;
    }
    label {
        color: #3466a3 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>ProPhet-BnB</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Simple Airbnb Analytics & Price Predictions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='simple-steps'>"
    "<b>1.</b> Choose data source<br>"
    "<b>2.</b> Adjust filters<br>"
    "<b>3.</b> Click <b>Analyze</b> for insights"
    "</div>", unsafe_allow_html=True
)

if st.session_state.get("df_base") is None:
    st.markdown(
        "<div class='data-link-info'>"
        "Paste a data link, pick a city, or upload a CSV, then hit <b>Analyze Listings</b>.<br>"
        "Or use Demo Mode for a walkthrough."
        "</div>", unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

st.caption("Supports InsideAirbnb, CSVs, direct links, and custom scraping. Use InsideAirbnb or a clean CSV for best results.")

# ---- Sidebar and main logic (unchanged from previous versions) ----
with st.sidebar:
    st.header("Data Source")
    source_mode = st.radio(
        "Choose Source Type",
        [
            "InsideAirbnb Snapshot",
            "Local CSV Upload",
            "Direct CSV URL",
            "Website (Custom Scraper)"
        ]
    )
    if st.button("Load Example Data"):
        st.session_state["df_base"] = pd.DataFrame({
            "id": range(1, 11),
            "name": [f"Demo Home {i}" for i in range(1, 11)],
            "neighbourhood": ["Downtown", "Uptown", "Central", "Beach", "Suburb", "Park", "Museum", "Old Town", "Lake", "Market"],
            "room_type": ["Entire home", "Private room", "Shared room", "Entire home", "Private room", "Entire home", "Private room", "Shared room", "Entire home", "Private room"],
            "price": [120, 80, 45, 200, 90, 130, 60, 30, 170, 100],
            "review_scores_rating": [4.8, 4.5, 4.2, 4.9, 4.0, 4.7, 4.6, 3.9, 4.8, 4.4],
            "image_url": ["https://picsum.photos/200/150?random=%d" % i for i in range(1, 11)],
            "num_reviews": [25, 40, 12, 60, 10, 15, 30, 8, 50, 20],
            "availability_365": [320, 180, 90, 360, 200, 300, 150, 60, 330, 210],
            "amenities_count": [12, 8, 6, 15, 7, 10, 5, 4, 13, 9],
            "total_score": [9.7, 8.2, 7.3, 9.8, 7.8, 8.9, 8.0, 6.7, 9.5, 8.3],
            "recommendation_reason": ["Great reviews"]*10
        })
        st.session_state["source_label"] = "Demo Example Data"
        st.session_state["demo_mode"] = True
    # ... (add your data source logic here, as in previous versions)

# (Rest of your app logic: filters, analysis, tabs, etc. can go below, reusing code logic from previous versions)
