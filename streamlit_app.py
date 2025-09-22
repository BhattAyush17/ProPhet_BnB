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

# ---- Custom CSS for dark UI, sidebar, and hero block ----
st.markdown("""
    <style>
    .stApp {
        background: url('https://media.timeout.com/images/105900700/image.jpg') no-repeat center center fixed !important;
        background-size: cover !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(20, 29, 44, 0.99) !important;
        color: #fff !important;
        min-width: 350px !important;
        max-width: 400px !important;
        padding-top: 30px !important;
        padding-bottom: 32px !important;
    }
    .sidebar-instruct-box {
        background: rgba(12, 17, 27, 0.98);
        border-radius: 18px;
        padding: 24px 22px 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px #22304e88;
        color: #eaf6ff;
        font-size: 1.18em;
        font-weight: 600;
        text-align: left;
        letter-spacing: 0.02em;
    }
    .sidebar-instruct-box .instruct-title {
        font-size: 1.23em;
        font-weight: 900;
        color: #90caf9;
        margin-bottom: 8px;
        letter-spacing: 0.04em;
    }
    .sidebar-instruct-box ul {
        padding-left: 1.2em;
        margin: 0.5em 0 0 0;
    }
    .sidebar-instruct-box li {
        margin-bottom: 7px;
        font-size: 1.06em;
        color: #eaf6ff;
    }
    .sidebar-instruct-box .spec-title {
        color: #7bdfff;
        font-size: 1.08em;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 2px;
    }
    .sidebar-instruct-box .spec-list {
        font-size: 0.97em;
        color: #b4c3d8;
        margin-left: 1.1em;
    }
    .sidebar-section-header {
        font-size: 1.4em;
        font-weight: 800;
        color: #fff !important;
        margin-bottom: 18px;
        margin-top: 8px;
        letter-spacing: 0.03em;
    }
    .sidebar-caption {
        color: #b4c3d8 !important;
        font-size: 1.01em;
        margin-bottom: 6px;
    }
    .sidebar-step {
        color: #90caf9;
        font-size: 1.14em;
        font-weight: 700;
        margin: 18px 0 7px 0;
        letter-spacing: 0.03em;
    }
    .sidebar-radio label, .sidebar-radio div {
        color: #dde3ec !important;
        font-size: 1.15em !important;
        font-weight: 600;
    }
    .stButton>button, .stDownloadButton>button {
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
    .stButton>button:disabled {
        background-color: #555 !important;
        color: #aaa !important;
    }
    label, .stSelectbox label, .stSlider label, .stTextInput label {
        color: #90caf9 !important;
        font-weight: 700 !important;
    }
    /* Hero block */
    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        min-height: 65vh;
        width: 100%;
        margin-top: 56px;
    }
    .hero-title {
        font-size: 3.2em;
        color: #eaf6ff;
        font-weight: 900;
        letter-spacing: 0.07em;
        margin-bottom: 12px;
        line-height: 1.05em;
        text-align: center;
        text-shadow: 0 4px 28px #101428b0;
    }
    .hero-desc {
        color: #7bdfff;
        font-size: 1.28em;
        font-weight: 700;
        margin-bottom: 36px;
        letter-spacing: 0.01em;
        text-align: center;
        text-shadow: 0 2px 16px #22304e88;
        max-width: 700px;
    }
    .hero-steps-card {
        background: rgba(16, 22, 34, 0.98);
        border-radius: 22px;
        padding: 34px 36px 28px 36px;
        box-shadow: 0 8px 40px #22304e99;
        max-width: 640px;
        margin: 0 auto;
        margin-bottom: 22px;
        text-align: left;
        display: block;
    }
    .hero-steps-list {
        font-size: 1.18em;
        font-weight: 500;
        line-height: 2.1em;
        color: #eaf6ff;
        margin: 0;
        padding: 0;
    }
    .hero-steps-list li {
        margin-bottom: 10px;
        list-style: none;
    }
    .step-num {
        font-size: 1.12em;
        font-weight: 800;
        color: #60baff;
        margin-right: 7px;
    }
    .step-action {
        color: #eaf6ff;
        font-weight: 700;
        font-size: 1.07em;
        margin-right: 7px;
    }
    .step-desc {
        color: #b4c3d8;
        font-size: 1em;
        font-weight: 500;
    }
    .info-caption {
        margin-top: 15px;
        color: #ffcf7b !important;
        font-size: 1.13em;
        font-weight: 500;
        text-shadow: 0 1px 8px #10142888;
        text-align: center;
        width: 100%;
    }
    .main-card {
        background: rgba(20, 29, 44, 0.91);
        border-radius: 20px;
        margin: 60px auto 0 auto;
        padding: 38px 52px;
        box-shadow: 0 8px 40px #22304e99;
        max-width: 860px;
        text-align: center;
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
    @media (max-width: 900px) {
        .hero-section { margin-top: 20px; }
        .hero-title { font-size: 2em; }
        .hero-desc { font-size: 1em;}
        .hero-steps-card { font-size: 1em; padding: 16px 9px;}
    }
    </style>
""", unsafe_allow_html=True)

# --- HERO (Landing) Section ---
st.markdown(
    """
    <div class='hero-section'>
        <div class='hero-title'>ProPhet-BnB</div>
        <div class='hero-desc'>
            Welcome! Discover, analyze, and compare Airbnb listings and prices.<br>
            Upload your data or select a city, tweak your search, and get instant smart recommendations.<br>
            It's easy, visual, and designed for both hosts and travelers!
        </div>
        <div class='hero-steps-card'>
            <ul class='hero-steps-list'>
                <li>
                    <span class='step-num'>1.</span>
                    <span class='step-action'>Choose data source</span>
                    <span class='step-desc'>Select Airbnb or upload your own CSV file.</span>
                </li>
                <li>
                    <span class='step-num'>2.</span>
                    <span class='step-action'>Adjust filters</span>
                    <span class='step-desc'>Set your price, guest, date, and review preferences.</span>
                </li>
                <li>
                    <span class='step-num'>3.</span>
                    <span class='step-action'>Click Analyze</span>
                    <span class='step-desc'>Get instant recommendations and insights.</span>
                </li>
            </ul>
        </div>
        <div class='info-caption'>
            Supports InsideAirbnb, CSVs, direct links, and custom scraping.<br>
            For best results, use InsideAirbnb or a clean CSV!
        </div>
    </div>
    """, unsafe_allow_html=True
)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class='sidebar-instruct-box'>
        <div class='instruct-title'>How to Begin</div>
        <ul>
            <li>Choose your data source from Airbnb, CSV, or web.</li>
            <li>Set your filters for price, guests, reviews, etc.</li>
            <li>Click <span style='color:#7bdfff;font-weight:700;'>Analyze Listings</span> for insights.</li>
        </ul>
        <div class='spec-title'>Data & Format Specs:</div>
        <div class='spec-list'>
            • Supported: InsideAirbnb, CSV (with at least: id, name, price)<br>
            • For best experience: Use clean, recent datasets<br>
            • Need help? Try <span style='color:#7bdfff;font-weight:700;'>Load Example Data</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

# ---- MAIN LOGIC (analysis, tabs, etc. go below) ----
# ... (rest of your app logic unchanged, see previous code blocks) ...
