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
    .stApp {
        background: url('https://media.timeout.com/images/105900700/image.jpg') no-repeat center center fixed !important;
        background-size: cover !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(20, 29, 44, 0.98) !important;
        color: #fff !important;
        min-width: 350px !important;
        max-width: 400px !important;
        padding-top: 28px !important;
        padding-bottom: 28px !important;
    }
    .sidebar-section-header {
        font-size: 1.4em;
        font-weight: 800;
        color: #fff !important;
        margin-bottom: 18px;
        margin-top: 8px;
        letter-spacing: 0.03em;
    }
    .sidebar-step {
        color: #90caf9;
        font-size: 1.12em;
        font-weight: 700;
        margin: 18px 0 7px 0;
        letter-spacing: 0.03em;
    }
    .sidebar-help {
        background: rgba(36, 44, 66, 0.98);
        border-radius: 14px;
        margin-bottom: 18px;
        padding: 20px 18px;
        color: #dde3ec;
        font-size: 1.09em;
        box-shadow: 0 2px 8px #1a223355;
        border: 1px solid #29426d;
    }
    .sidebar-btn {
        background: #1976d2 !important;
        color: #fff !important;
        border-radius: 7px !important;
        font-size: 1.11em !important;
        font-weight: 700 !important;
        padding: 0.6em 1.8em !important;
        margin: 10px 0 !important;
        border: none !important;
        box-shadow: 0 1px 2px #22304e;
        transition: background 0.2s;
    }
    .sidebar-btn:hover {
        background: #1565c0 !important;
    }
    .sidebar-radio label, .sidebar-radio div {
        color: #dde3ec !important;
        font-size: 1.15em !important;
        font-weight: 600;
    }
    .sidebar-caption {
        color: #b4c3d8 !important;
        font-size: 1.01em;
        margin-bottom: 6px;
    }
    label, .stSelectbox label, .stSlider label, .stTextInput label {
        color: #90caf9 !important;
        font-weight: 700 !important;
    }
    /* Main block */
    .main-card {
        background: rgba(20, 29, 44, 0.91);
        border-radius: 20px;
        margin: 60px auto 0 auto;
        padding: 38px 52px;
        box-shadow: 0 8px 40px #22304e99;
        max-width: 860px;
        text-align: center;
    }
    h1.main-title {
        font-size: 3.2em;
        color: #eaf6ff !important;
        font-weight: 900;
        margin-bottom: 0.13em;
        letter-spacing: 0.05em;
        text-shadow: 0 3px 24px #1a223399;
    }
    .subtitle {
        color: #90caf9;
        font-size: 1.23em;
        font-weight: 600;
        margin-bottom: 1.15em;
        margin-top: 0.3em;
        letter-spacing: 0.02em;
    }
    .steps-list {
        font-size: 1.14em;
        color: #eaf6ff;
        font-weight: 500;
        margin-bottom: 1.2em;
        text-align: left;
        margin: 0 auto 1.2em auto;
        max-width: 60%;
        line-height: 1.7em;
    }
    .steps-list .step-num {
        font-size: 1.05em;
        font-weight: bold;
        color: #90caf9;
        margin-right: 7px;
    }
    .steps-list .step-action {
        color: #90caf9;
        font-weight: 700;
    }
    .data-link-info {
        background: rgba(36,44,66,0.98);
        border-radius: 16px;
        padding: 26px 22px;
        font-size: 1.19em;
        color: #90caf9;
        font-weight: 600;
        margin-top: 22px;
        box-shadow: 0 2px 18px #22304e33;
        text-shadow: 0 1px 6px #15223844;
    }
    .data-link-info b, .data-link-info span {
        color: #fff;
        font-weight: bold;
    }
    .stCaption {
        color: #bfc8db !important;
        font-size: 1.07em;
        text-align: center;
        text-shadow: 0 1px 6px #15223844;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #151c28;
        border-radius: 8px;
        padding: 0.5em;
        color: #dde3ec !important;
    }
    .stTabs [aria-selected="true"] {
        color: #90caf9 !important;
        font-weight: bold !important;
        background: #222e44 !important;
    }
    /* Responsive adjustments for small screens */
    @media (max-width: 900px) {
        .main-card { padding: 20px 8px; }
        h1.main-title { font-size: 2.1em; }
        .sidebar-section-header { font-size: 1.15em; }
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='sidebar-section-header'>1. Data Source</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-caption'>Choose how you'd like to provide listing data.</div>", unsafe_allow_html=True)
    source_mode = st.radio(
        "",
        [
            "InsideAirbnb Snapshot",
            "Local CSV Upload",
            "Direct CSV URL",
            "Website (Custom Scraper)"
        ],
        key="sidebar_radio"
    )
    if st.button("Load Example Data", key="demo_btn"):
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

    if source_mode == "InsideAirbnb Snapshot":
        st.markdown("<div class='sidebar-step'>1.1 InsideAirbnb City/Date Picker</div>", unsafe_allow_html=True)
        st.caption("Pick country, region, city, and date.")
        @st.cache_data(show_spinner=False)
        def get_catalog():
            return scrape_catalog()
        try:
            catalog = get_catalog()
        except Exception as e:
            st.error(f"Could not load city catalog: {e}")
            st.stop()
        countries = sorted(catalog.keys())
        country = st.selectbox("Country", countries)
        region = st.selectbox("Region", sorted(catalog[country].keys()))
        city = st.selectbox("City", sorted(catalog[country][region].keys()))
        city_entry = catalog[country][region][city]
        dates = sorted(city_entry.versions.keys(), reverse=True)
        date = st.selectbox("Snapshot Date", dates, index=0)
        st.caption(f"Latest available date: {city_entry.latest_date}")
        force_download = st.checkbox("Force Fresh Download", value=False)
        custom_url = st.text_input("Custom Listings URL (override)", "", placeholder="https://insideairbnb.com/data/.../listings.csv.gz")
        version = city_entry.versions[date]
    elif source_mode == "Local CSV Upload":
        st.markdown("<div class='sidebar-step'>1.2 Upload Your Listings</div>", unsafe_allow_html=True)
        uploaded_listings = st.file_uploader("Listings CSV", type=["csv"])
        uploaded_reviews = st.file_uploader("Reviews CSV (optional)", type=["csv"])
    elif source_mode == "Direct CSV URL":
        st.markdown("<div class='sidebar-step'>1.3 Paste Direct CSV URL</div>", unsafe_allow_html=True)
        csv_url = st.text_input("Paste Direct CSV URL", "", placeholder="https://.../listings.csv")
    elif source_mode == "Website (Custom Scraper)":
        st.markdown("<div class='sidebar-step'>1.4 Custom Site Scraper</div>", unsafe_allow_html=True)
        site_url = st.text_input("Paste Listing Website Link", "", placeholder="https://www.example.com/listings")
        with st.expander("Advanced Scraper Settings"):
            listing_selector = st.text_input("Listing CSS Selector", value=".listing-card")
            price_selector = st.text_input("Price CSS Selector", value=".price")
            name_selector = st.text_input("Name CSS Selector", value=".name")
            image_selector = st.text_input("Image CSS Selector", value="img")

    st.markdown("<div class='sidebar-section-header'>2. Adjust Filters</div>", unsafe_allow_html=True)
    st.caption("Filter listings by price, reviews, ratings, and more.")
    default_filters = {
        "price_mode": "Budget",
        "custom_price_range": (0.0, 10000.0),
        "reviews_range": (0, 1000),
        "stars_range": (1.0, 5.0),
        "availability_range": (0, 365),
        "occupancy_group": "Any",
        "suggestions": 6,
        "map_sample": 2000
    }
    uf = st.session_state.get("user_filters", default_filters.copy())

    uf["suggestions"] = st.slider("Suggestions to Show", 3, 10, uf.get("suggestions", 6))
    uf["price_mode"] = st.radio("Price Band", ["Budget", "Comfort", "Premium", "Custom Range"], index=["Budget","Comfort","Premium","Custom Range"].index(uf.get("price_mode", "Budget")))
    if uf["price_mode"] == "Custom Range":
        uf["custom_price_range"] = st.slider("Custom Price Range [$]", 0.0, 10000.0, uf.get("custom_price_range", (0.0, 10000.0)))
    uf["reviews_range"] = st.slider("Reviews Count", 0, 1000, uf.get("reviews_range", (0, 1000)))
    uf["stars_range"] = st.slider("Rating (Stars)", 1.0, 5.0, uf.get("stars_range", (1.0, 5.0)), 0.5)
    uf["availability_range"] = st.slider("Availability Days", 0, 365, uf.get("availability_range", (0, 365)))
    uf["occupancy_group"] = st.selectbox("Guest Group", ["Any", "Solo (1)", "Duo (2)", "Small group (3-4)", "Family (5-6)", "Large (7+)"], index=["Any","Solo (1)","Duo (2)","Small group (3-4)","Family (5-6)","Large (7+)"].index(uf.get("occupancy_group", "Any")))
    st.session_state["user_filters"] = uf

    run_clicked = st.button("Analyze Listings", type="primary")

# --- MAIN BLOCK ---
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>ProPhet-BnB</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Simple Airbnb Analytics & Price Predictions</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='steps-list'>"
    "<span class='step-num'>1.</span> <span class='step-action'>Choose data source</span><br>"
    "<span class='step-num'>2.</span> <span class='step-action'>Adjust filters</span><br>"
    "<span class='step-num'>3.</span> <span class='step-action'>Click Analyze</span> for insights"
    "</div>", unsafe_allow_html=True
)

def get_demo_df():
    return pd.DataFrame({
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

if st.session_state.get("df_base") is None:
    st.markdown(
        "<div class='data-link-info'>"
        "Paste a data link, pick a city, or upload a CSV, then hit <b>Analyze Listings</b>.<br>"
        "Or use <span>Demo Mode</span> for a walkthrough."
        "</div>", unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)
st.caption("Supports InsideAirbnb, CSVs, direct links, and custom scraping. Use InsideAirbnb or a clean CSV for best results.")

# MAIN LOGIC GOES HERE (see previous completions for analysis, tabs, etc.)
