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
    uploaded_listings = uploaded_reviews = None
    site_url = csv_url = ""
    city_entry = None
    version = None
    custom_url = ""
    catalog = None

    if "demo_mode" not in st.session_state:
        st.session_state["demo_mode"] = False

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

# ---- MAIN LOGIC (analysis, tabs, etc.) ----
def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    for col in df.columns:
        for name in names:
            if name.lower() in col.lower():
                return col
    return None

def get_numeric_cols(df):
    return [c for c in df.select_dtypes(include='number').columns if df[c].nunique() > 1]

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

df, source_label = None, ""
max_rows = 10000

def load_dataset():
    if source_mode == "InsideAirbnb Snapshot":
        files = download_dataset(
            version,
            city=city,
            date=date,
            force=force_download,
            override_listings_url=custom_url or None
        )
        df_local = load_data(files["listings"], files["reviews"], files.get("neighbourhoods"))
        df_local = clean_data(df_local)
        meta = {
            "source_label": f"{city} {date}",
            "files": files,
            "mode": "InsideAirbnb"
        }
        return df_local, meta
    if source_mode == "Local CSV Upload":
        if not uploaded_listings:
            st.error("Please upload a listings CSV file.")
            st.stop()
        try:
            df_local = pd.read_csv(uploaded_listings)
        except Exception as e:
            st.error(f"Could not read listings file: {e}")
            st.stop()
        if uploaded_reviews:
            try:
                rev_df = pd.read_csv(uploaded_reviews)
                if "id" in df_local.columns and "listing_id" in rev_df.columns:
                    summary = rev_df.groupby("listing_id").size().rename("num_reviews")
                    df_local = df_local.merge(summary, left_on="id", right_index=True, how="left")
            except Exception as e:
                st.warning(f"Could not read reviews file: {e}")
        df_local = clean_data(df_local)
        return df_local, {"source_label": "Manual Upload", "mode": "LocalCSV"}
    if source_mode == "Direct CSV URL":
        if not csv_url.strip():
            st.error("Please provide a valid CSV URL.")
            st.stop()
        src = DirectCSVURLSource(url=csv_url)
        result = src.load()
        return result.df, {"source_label": "Direct CSV URL", "url": csv_url, "mode": "DirectURL"}
    if source_mode == "Website (Custom Scraper)":
        if not site_url.strip():
            st.error("Please provide a valid listing website link.")
            st.stop()
        src = ExternalSiteSource(
            url=site_url,
            listing_selector=listing_selector,
            field_map={
                "name": {"selector": name_selector, "attr": "text"},
                "price": {"selector": price_selector, "attr": "text"},
                "image_url": {"selector": image_selector, "attr": "src"},
            }
        )
        result = src.load()
        df_local = getattr(result, "df", None)
        if df_local is None or df_local.empty:
            st.error("No listings found. Check your selectors or try a different site.")
            st.stop()
        return df_local, {"source_label": f"Scraped from {site_url}", "mode": "CustomScraper"}
    raise RuntimeError("Unsupported source mode.")

if run_clicked and not st.session_state.get("demo_mode", False):
    try:
        df, meta = load_dataset()
        source_label = meta.get("source_label", "")
        if df is None or df.empty:
            st.error("No data extracted. Please check your upload/site/link or selectors.")
            st.stop()
        if len(df) > max_rows:
            df = df.sample(max_rows)
            st.warning(f"Sampled {max_rows} rows for performance.")
        try:
            _, df = train_price_model(df)
        except Exception:
            pass
        try:
            _, df = cluster_hosts(df)
        except Exception:
            pass
        df = build_recommendation_scores(df)
        st.session_state["df_base"] = df
        st.session_state["source_label"] = source_label
        st.success(f"Loaded {len(df)} listings.")
    except Exception as e:
        st.error(f"Could not read or process data: {e}")
        st.stop()
if st.session_state.get("demo_mode", False):
    df = st.session_state.get("df_base")
    source_label = st.session_state.get("source_label", "")
else:
    df = st.session_state.get("df_base")
    source_label = st.session_state.get("source_label", "")

# ---- MAIN ANALYSIS UI ----
if df is not None:
    st.markdown(f"<div class='main-card'><h2 style='color:#90caf9;'>Source: {source_label}</h2></div>", unsafe_allow_html=True)

    metrics, price_col = compute_metrics(df)
    def fmt(v): return f"{v:,.1f}" if v is not None and pd.notnull(v) else "—"

    img_col = find_col(df, ["image_url", "Image", "img", "photo", "picture"])
    table_cols = ["id", "name", "neighbourhood", "room_type"]
    for col in [price_col, 'review_scores_rating', img_col]:
        if col and col in df.columns: table_cols.append(col)

    tab_overview, tab_recommend, tab_compare, tab_scatter3d = st.tabs(
        [
            "Overview",
            "Recommendations",
            "Comparison",
            "3D Scatter Plot"
        ]
    )

    with tab_overview:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#90caf9;'>Overview & Sample</h3>", unsafe_allow_html=True)
        st.caption("Quickly explore your first 25 listings and summary metrics.")
        st.dataframe(df.head(25)[table_cols], height=350)
        kcols = st.columns(6)
        metrics_display = [
            ("Avg Price", metrics['avg_price']),
            ("Avg Reviews", metrics['avg_reviews']),
            ("Avg Rating", metrics['avg_rating']),
            ("Avg Availability", metrics['avg_availability']),
            ("Avg Amenities", metrics['avg_amenities']),
            ("Listings", metrics['listings'])
        ]
        for (label, val), col in zip(metrics_display, kcols):
            col.metric(label, fmt(val))
        st.write(f"**Active Price Range:** {fmt(metrics['avg_price'])}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_recommend:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Top Suggested Listings")
        st.caption("Ranked by your selected preferences.")
        recomm_df = df.sort_values("total_score", ascending=False).head(uf["suggestions"])
        rec_cols = [c for c in ["id", "name", "neighbourhood", "room_type", price_col, "review_scores_rating", img_col] if c in recomm_df.columns]
        st.dataframe(recomm_df[rec_cols], height=400)
        st.download_button(
            "Download Suggestions CSV",
            recomm_df[rec_cols].to_csv(index=False),
            file_name="suggestions.csv",
            mime="text/csv"
        )
        st.markdown("#### Most Accurate & Optimized Option")
        best_row = recomm_df.iloc[0]
        info = f"**{best_row.get('name', 'Listing')}**"
        if 'neighbourhood' in best_row and pd.notnull(best_row['neighbourhood']):
            info += f" in *{best_row['neighbourhood']}*"
        if price_col in best_row and pd.notnull(best_row[price_col]):
            info += f" (${best_row[price_col]}/night)"
        st.markdown(info)
        if img_col and pd.notnull(best_row[img_col]):
            st.image(best_row[img_col], width=220)
        radar_fig = radar_for_listing(best_row, metrics)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("#### Why is this the best for you?")
        st.info(
            f"This listing was chosen because it matches your selected price range, guest group, "
            f"and offers strong ratings and amenities. Reason: {best_row.get('recommendation_reason','N/A')}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_compare:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Compare Top Picks (Visual & Images)")
        st.caption("Visualize and compare your top recommendations side-by-side.")
        pfig = parallel_recommendations(recomm_df, max_recs=uf["suggestions"])
        if pfig:
            st.plotly_chart(pfig, use_container_width=True)
        else:
            st.warning("Not enough scoring columns to show parallel recommendations for this dataset.")
        top_n = min(3, len(recomm_df))
        img_cols = st.columns(top_n)
        for idx in range(top_n):
            row = recomm_df.iloc[idx]
            with img_cols[idx]:
                st.markdown(f"**{row.get('name', 'Listing')}**")
                if img_col and pd.notnull(row[img_col]):
                    st.image(row[img_col], width=170)
                price = row.get(price_col, "N/A")
                rating = row.get("review_scores_rating", "N/A")
                location = row.get('neighbourhood', 'N/A')
                amenities = row.get('amenities_count', 'N/A')
                st.caption(f"Price: ${price}, Rating: {rating}, Area: {location}, Amenities: {amenities}")
        def format_listing(x):
            row = recomm_df[recomm_df["id"] == x]
            if not row.empty and "name" in row.columns:
                return f"{row.iloc[0]['name']} ({row.iloc[0]['neighbourhood']})"
            return str(x)
        chosen_id = st.selectbox("Select Listing for Radar", recomm_df["id"], format_func=format_listing)
        rrow = recomm_df[recomm_df["id"] == chosen_id]
        if not rrow.empty:
            rrow = rrow.iloc[0]
            listing_info = f"**{rrow.get('name', 'Listing')}**"
            if 'neighbourhood' in rrow and pd.notnull(rrow['neighbourhood']):
                listing_info += f" in *{rrow['neighbourhood']}*"
            if price_col in rrow and pd.notnull(rrow[price_col]):
                listing_info += f" (${rrow[price_col]}/night)"
            st.markdown(listing_info)
            if img_col and pd.notnull(rrow[img_col]):
                st.image(rrow[img_col], width=180)
            rfig = radar_for_listing(rrow, metrics)
            if rfig:
                st.plotly_chart(rfig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_scatter3d:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("3D Scatter Plot")
        st.caption("Explore listings across three dimensions.")
        numeric_cols = get_numeric_cols(df)
        if len(numeric_cols) < 3:
            st.info("Not enough numeric columns for 3D scatter plot.")
        else:
            x_col = st.selectbox("X axis", numeric_cols, index=0, key="3d_x")
            y_col = st.selectbox("Y axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="3d_y")
            z_col = st.selectbox("Z axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="3d_z")
            color_col = st.selectbox(
                "Color by",
                [c for c in df.columns if df[c].nunique() < 50 and df[c].dtype == object],
                index=0,
                key="3d_color"
            ) if any(df[c].nunique() < 50 and df[c].dtype == object for c in df.columns) else None
            fig3d = px.scatter_3d(
                df,
                x=x_col,
                y=y_col,
                z=z_col,
                color=color_col,
                hover_name="name" if "name" in df.columns else None,
                hover_data=table_cols,
                title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
                height=700
            )
            st.plotly_chart(fig3d, use_container_width=True)
            st.markdown("### Top Listings Visual Comparison (by 3D scatter plot values)")
            top_points = df.sort_values([z_col, y_col, x_col], ascending=False).head(3)
            img_cols = st.columns(3)
            for idx in range(len(top_points)):
                row = top_points.iloc[idx]
                with img_cols[idx]:
                    st.markdown(f"**{row.get('name', 'Listing')}**")
                    if img_col and pd.notnull(row[img_col]):
                        st.image(row[img_col], width=170)
                    price = row.get(price_col, "N/A")
                    rating = row.get("review_scores_rating", "N/A")
                    location = row.get('neighbourhood', 'N/A')
                    amenities = row.get('amenities_count', 'N/A')
                    st.caption(f"Price: ${price}, Rating: {rating}, Area: {location}, Amenities: {amenities}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='main-card'>"
        "<div class='info-caption'>Paste a data link, pick a city, or upload a CSV, then hit <b>Analyze Listings</b>.<br>"
        "Or use <span style='color:#7bdfff;font-weight:700;'>Demo Mode</span> for a walkthrough."
        "</div></div>", unsafe_allow_html=True
    )
st.caption("Supports InsideAirbnb, CSVs, direct links, and custom scraping. Use InsideAirbnb or a clean CSV for best results.")
