import streamlit as st
import pandas as pd
import os
import sys
import joblib
from PIL import Image
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from datetime import datetime, timezone

# Tell Python where to find our custom modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.risk_module.weather_features import WeatherFetcher
from src.image_module.vit_model import FireImageDetector
from src.risk_module.spread_model import calculate_spread_cone
from src.hotspot_module.modis_loader import ModisLoader

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PyroWatch — AI Wildfire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 2. Inject Professional Custom CSS
# ─────────────────────────────────────────────────────────────
from src.workspace_ui import render_workspace_ui

render_workspace_ui()


# ─────────────────────────────────────────────────────────────
# 3. Load Models and Data (Cached for speed)
# ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "latest_hotspots.csv")
RF_MODEL_PATH = os.path.join(ROOT_DIR, "models", "rf_risk_model.pkl")

@st.cache_data
def load_data(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource
def load_rf_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_vit_model():
    return FireImageDetector()

# Initialize classes and data
df = load_data(DATA_PATH)
rf_model = load_rf_model(RF_MODEL_PATH)
vit_model = load_vit_model()
weather_fetcher = WeatherFetcher()


# ─────────────────────────────────────────────────────────────
# 4. Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="pyro-header">
    <h1>🔥 PyroWatch</h1>
    <p>Real-time AI-powered wildfire detection, risk assessment, and spread prediction engine</p>
    <div class="pyro-status">
        <span class="pyro-status-dot"></span>
        All Systems Online
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 5. Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🔥 PyroWatch</div>
        <div class="sidebar-brand-sub">Wildfire Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📍 Location")
    
    # Initialize search coordinates if not set
    if 'search_lat' not in st.session_state:
        st.session_state['search_lat'] = 34.05
    if 'search_lon' not in st.session_state:
        st.session_state['search_lon'] = -118.24
    
    # Place name search
    place_search = st.text_input(
        "🔍 Search by place name",
        placeholder="e.g., Los Angeles, California Wildfires, Yosemite",
        help="Search for a location by name. Coordinates will be auto-filled."
    )
    
    if place_search and st.button("🌍 Search Location", use_container_width=True):
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="pyrowatch_app")
            location = geolocator.geocode(place_search)
            if location:
                st.session_state['search_lat'] = location.latitude
                st.session_state['search_lon'] = location.longitude
                st.session_state['last_search_place'] = location.address
                st.success(f"✅ Found: {location.address[:60]}")
                st.rerun()
            else:
                st.error("❌ Location not found. Try a different search term.")
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
    
    # Use session state directly with key parameter for two-way binding
    sim_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=st.session_state['search_lat'], format="%.4f", key='search_lat')
    sim_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=st.session_state['search_lon'], format="%.4f", key='search_lon')

    st.markdown("---")

    # Initialize session state for weather variables
    if 'temp' not in st.session_state: st.session_state['temp'] = 35.0
    if 'humidity' not in st.session_state: st.session_state['humidity'] = 15.0
    if 'wind_speed' not in st.session_state: st.session_state['wind_speed'] = 5.0
    if 'wind_deg' not in st.session_state: st.session_state['wind_deg'] = 180.0
    if 'risk_level' not in st.session_state: st.session_state['risk_level'] = -1
    if 'weather_fetched' not in st.session_state: st.session_state['weather_fetched'] = False
    if 'alert_history' not in st.session_state: st.session_state['alert_history'] = []

    st.markdown("## 🌤️ Weather Data")

    coordinates = (sim_lat, sim_lon)
    location_changed = st.session_state.get("weather_location") != coordinates
    refresh = st.button("📡  Fetch Live Weather", use_container_width=True)
    if location_changed or refresh:
        st.session_state.weather_location = coordinates
        with st.spinner("Fetching current weather..."):
            reading = weather_fetcher.get_current_reading(*coordinates)
        st.session_state.weather_reading = reading
        if reading and reading["values"][3] is not None:
            for key, value in zip(("temp", "humidity", "wind_speed", "wind_deg"), reading["values"]):
                st.session_state[key] = float(value)
        elif location_changed:
            st.session_state.update(temp=35.0, humidity=15.0, wind_speed=5.0, wind_deg=180.0)

    sim_temp = st.slider("🌡️ Temperature (°C)", min_value=min(-10.0, st.session_state.temp), max_value=max(50.0, st.session_state.temp), value=st.session_state['temp'], key='temp')
    sim_humidity = st.slider("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=st.session_state['humidity'], key='humidity')
    sim_wind_speed = st.slider("💨 Wind Speed (m/s)", min_value=0.0, max_value=max(50.0, st.session_state.wind_speed), value=st.session_state['wind_speed'], key='wind_speed')
    sim_wind_deg = st.slider("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=st.session_state['wind_deg'], key='wind_deg')

    st.markdown("---")
    st.markdown("## 🎯 Risk Prediction")

    inputs = (sim_lat, sim_lon, sim_temp, sim_humidity, sim_wind_speed, sim_wind_deg)
    if st.session_state.get("assessed_inputs") != inputs or rf_model is None:
        st.session_state.risk_level = -1
        st.session_state.pop("assessed_inputs", None)
    st.caption("Experimental Random Forest trained with simulated weather.")

    if rf_model is not None:
        if st.button("⚡  Predict Ignition Risk", use_container_width=True):
            input_features = pd.DataFrame(
                [[sim_lat, sim_lon, sim_temp, sim_humidity, sim_wind_speed, sim_wind_deg]],
                columns=['latitude', 'longitude', 'temperature_c', 'humidity_percent', 'wind_speed', 'wind_deg']
            )
            st.session_state.risk_level = -1
            try:
                st.session_state.risk_level = int(rf_model.predict(input_features)[0])
                st.session_state.assessed_inputs = inputs
            except ValueError:
                st.error("Prediction failed for these inputs.")
    else:
        st.warning("⚠️ Risk model not loaded")

    # Show risk result in sidebar
    if st.session_state['risk_level'] == 1:
        st.markdown("""
        <div class="risk-badge risk-high">
            <span class="risk-icon">🚨</span>
            <div>
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">Experimental risk</div>
                <div>HIGH</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state['risk_level'] == 0:
        st.markdown("""
        <div class="risk-badge risk-low">
            <span class="risk-icon">✅</span>
            <div>
                <div style="font-size:0.7rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px;">Experimental risk</div>
                <div>LOW</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🛰️ Live Data Feed")
    
    # Add days selector for historical data
    days_to_fetch = st.selectbox(
        "Data Range",
        options=[1, 2, 3],
        index=0,
        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}",
        help="NASA FIRMS NRT data is most reliable for the last 1-3 days"
    )

    if st.button("🔄  Refresh Hotspot Data", use_container_width=True):
        with st.status("Fetching latest MODIS fire data from NASA FIRMS...", expanded=True) as status:
            try:
                loader = ModisLoader()
                raw_data = loader.fetch_latest_data(days=days_to_fetch)
                if raw_data is not None and not raw_data.empty:
                    save_path = os.path.join(ROOT_DIR, "data", "processed", "latest_hotspots.csv")
                    loader.preprocess_and_save(raw_data, output_path=save_path)
                    # Clear the cached data so it reloads on next run
                    load_data.clear()
                    status.update(label=f"✅ Retrieved {len(raw_data):,} records from last {days_to_fetch} day(s)! Reloading...", state="complete")
                    st.rerun()
                else:
                    status.update(label=f"⚠️ No fire data available for the last {days_to_fetch} day(s). This could mean no fires were detected globally during this period, or the API has temporary issues.", state="error")
            except ValueError as ve:
                status.update(label=f"❌ Configuration error: {str(ve)[:80]}", state="error")
            except Exception as e:
                status.update(label=f"❌ Error: {str(e)[:80]}", state="error")

    st.markdown(
        f'<p style="color:#6B7280;font-size:0.7rem;text-align:center;">📊 Hotspots loaded: {len(df):,}<br>'
        f'📅 {datetime.now().strftime("%b %d, %Y • %H:%M")}</p>',
        unsafe_allow_html=True
    )
    
    # Fire Alert History Log
    if st.session_state['alert_history']:
        st.markdown("---")
        st.markdown("## 📋 Alert History")
        
        with st.expander(f"🔍 View {len(st.session_state['alert_history'])} Recent Analyses", expanded=False):
            for entry in st.session_state['alert_history'][:10]:  # Show last 10
                icon = "🔥" if entry['is_fire'] else "✅"
                color = "#EF4444" if entry['is_fire'] else "#10B981"
                st.markdown(f"""
                <div style="background:rgba(17,24,39,0.5);border-left:3px solid {color};
                            padding:0.75rem;margin-bottom:0.5rem;border-radius:4px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:1.2rem;">{icon}</span>
                        <span style="color:#6B7280;font-size:0.7rem;">{entry['timestamp']}</span>
                    </div>
                    <div style="color:#F9FAFB;font-size:0.8rem;font-weight:600;margin:0.3rem 0;">
                        {entry['image_name'][:30]}{'...' if len(entry['image_name']) > 30 else ''}
                    </div>
                    <div style="color:#9CA3AF;font-size:0.75rem;">
                        {entry['classification']} • {entry['confidence']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state['alert_history'] = []
                st.rerun()


# ─────────────────────────────────────────────────────────────
# 6. Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🛰️  Command Center", "🔬  AI Vision Lab", "📊  Analytics"])


# ══════════════════════════════════════════════════════════════
# TAB 1: Command Center — Map & Risk Engine
# ══════════════════════════════════════════════════════════════
with tab1:

    # ── Metric Cards ──
    temp_color = "#EF4444" if sim_temp > 35 else ("#F59E0B" if sim_temp > 25 else "#3B82F6")
    humidity_color = "#10B981" if sim_humidity > 60 else ("#F59E0B" if sim_humidity > 30 else "#EF4444")
    wind_color = "#EF4444" if sim_wind_speed > 20 else ("#F59E0B" if sim_wind_speed > 10 else "#10B981")

    reading = st.session_state.get("weather_reading")
    if reading and reading["values"] == inputs[2:]:
        st.caption(f"OpenWeather current weather · {reading['place']} "
                   f"({sim_lat:.4f}, {sim_lon:.4f}) · Data time {reading['time']:%Y-%m-%d %H:%M} UTC")
    elif reading and reading["values"][3] is not None:
        st.caption("Manual weather inputs · Edited after the last OpenWeather reading.")
    else:
        st.warning("Current weather unavailable or incomplete. The cards show scenario inputs.")

    # Wind direction label
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wind_dir_label = directions[int(((sim_wind_deg + 22.5) % 360) / 45)]

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <span class="metric-icon">🌡️</span>
            <div class="metric-label">Temperature</div>
            <div class="metric-value" style="color:{temp_color};">{sim_temp:.1f}<span class="metric-unit">°C</span></div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">💧</span>
            <div class="metric-label">Humidity</div>
            <div class="metric-value" style="color:{humidity_color};">{sim_humidity:.1f}<span class="metric-unit">%</span></div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">💨</span>
            <div class="metric-label">Wind Speed</div>
            <div class="metric-value" style="color:{wind_color};">{sim_wind_speed:.1f}<span class="metric-unit">m/s</span></div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">🧭</span>
            <div class="metric-label">Wind Direction</div>
            <div class="metric-value">{'Calm' if sim_wind_speed == 0 else f'{sim_wind_deg:.0f}° {wind_dir_label}'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Alert Banner (main area) ──
    risk_level = st.session_state.get('risk_level', -1)
    if risk_level == 1:
        st.markdown("""
        <div class="risk-badge risk-high" style="justify-content:center;">
            <span class="risk-icon">🚨</span>
            <span>Experimental risk classification — HIGH</span>
        </div>
        """, unsafe_allow_html=True)
    elif risk_level == 0:
        st.markdown("""
        <div class="risk-badge risk-low" style="justify-content:center;">
            <span class="risk-icon">✅</span>
            <span>Experimental risk classification — LOW</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Experimental risk: Not assessed")

    # ── Section Header + Hotspot Toggle ──
    sec_col1, sec_col2 = st.columns([3, 1])
    with sec_col1:
        st.markdown("""
        <div class="section-header">
            <h3>🗺️ Wind scenario map</h3>
            <div class="section-divider"></div>
        </div>
        """, unsafe_allow_html=True)
    with sec_col2:
        show_hotspots = st.toggle("🔴 Show Global Hotspots", value=False)

    # ── Folium Map ──
    map_zoom = 3 if show_hotspots else 11
    m = folium.Map(
        location=[sim_lat, sim_lon],
        zoom_start=map_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Ignition Point Marker
    marker_color = "red" if risk_level == 1 else ("green" if risk_level == 0 else "blue")
    popup_html = f"""
    <div style="font-family:Inter,sans-serif;min-width:180px;padding:4px;">
        <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:#1F2937;">
            📍 Ignition Point
        </div>
        <div style="font-size:12px;color:#6B7280;">
            <b>Lat:</b> {sim_lat:.4f} &nbsp; <b>Lon:</b> {sim_lon:.4f}<br>
            <b>Temp:</b> {sim_temp:.1f}°C &nbsp; <b>Humidity:</b> {sim_humidity:.0f}%<br>
            <b>Wind:</b> {sim_wind_speed:.1f} m/s @ {sim_wind_deg:.0f}°<br>
            <b>Experimental risk:</b> <span style="color:{'#EF4444' if risk_level == 1 else '#10B981'};font-weight:700;">
            {'HIGH' if risk_level == 1 else 'LOW' if risk_level == 0 else 'Not assessed'}</span>
        </div>
    </div>
    """
    folium.Marker(
        [sim_lat, sim_lon],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color=marker_color, icon="fire", prefix="fa"),
    ).add_to(m)

    # Danger Cone with Timeline Animation
    st.markdown("### ⏱️ Wind scenario timeline")
    
    timeline_hours = st.select_slider(
        "Select time horizon",
        options=[1, 3, 6, 12, 24],
        value=6,
        format_func=lambda x: f"{x}h",
        help="Hours from the first forecast time"
    )
    
    @st.cache_data(ttl=900, show_spinner=False)
    def forecast_for_location(lat, lon):
        return weather_fetcher.get_forecast(lat, lon), datetime.now(timezone.utc)

    forecast, retrieved = forecast_for_location(sim_lat, sim_lon)
    if forecast:
        start = datetime.fromtimestamp(forecast[0]["dt"], timezone.utc)
        st.caption(f"Scenario starts {start:%Y-%m-%d %H:%M} UTC. OpenWeather forecast retrieved "
                   f"{retrieved:%Y-%m-%d %H:%M} UTC. Issue time unavailable.")
        st.caption("Each 3-hour wind estimate moves the scenario downwind at 1% of wind speed. "
                   "Cone width and travel rate are illustrative.")
        origin = (sim_lat, sim_lon)
        elapsed = 0
        for interval in forecast:
            wind = interval.get("wind", {})
            direction, speed = wind.get("deg"), wind.get("speed")
            if interval["dt"] != forecast[0]["dt"] + elapsed * 3600 or direction is None or speed is None:
                st.info(f"Scenario stops at {elapsed}h because forecast wind is incomplete.")
                break
            hours = min(3, timeline_hours - elapsed)
            if speed > 0:
                polygon = calculate_spread_cone(*origin, speed, direction, hours)
                folium.Polygon(polygon, color="#FF6B35", fill=True,
                               tooltip=f"Wind scenario {elapsed}–{elapsed + hours}h").add_to(m)
                origin = polygon[2]
            elapsed += hours
            if elapsed == timeline_hours:
                break
        else:
            st.info(f"Forecast coverage ends at {elapsed}h.")
    else:
        st.info("Wind scenario unavailable. Check forecast access and the OpenWeather key.")

    # ── Hotspot Layer (toggled) ──
    if show_hotspots and not df.empty:
        cluster = MarkerCluster(name="🔥 MODIS Hotspots").add_to(m)
        # Sample if too many points for performance
        display_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df
        for _, row in display_df.iterrows():
            conf = row.get('confidence', 0)
            frp = row.get('frp', 0)
            # Color by confidence: red ≥90, orange ≥70, yellow <70
            if conf >= 90:
                dot_color = "#EF4444"
            elif conf >= 70:
                dot_color = "#F59E0B"
            else:
                dot_color = "#FBBF24"
            popup_text = (
                f"<div style='font-family:Inter,sans-serif;font-size:12px;min-width:150px;'>"
                f"<b style='color:#1F2937;'>🔥 Fire Hotspot</b><br>"
                f"<span style='color:#6B7280;'>Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}<br>"
                f"Confidence: <b>{conf}%</b><br>"
                f"FRP: <b>{frp:.1f}</b> MW<br>"
                f"Date: {row.get('acq_date', 'N/A')}</span></div>"
            )
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=dot_color,
                fill=True,
                fill_color=dot_color,
                fill_opacity=0.7,
                weight=1,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"Confidence: {conf}%",
            ).add_to(cluster)
        folium.LayerControl().add_to(m)

    # Render map in a styled container
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st_folium(m, width=None, height=520, use_container_width=True, returned_objects=[])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Map Legend ──
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1:
        st.markdown(
            '<p style="color:#9CA3AF;font-size:0.78rem;">🔴 <b>Red Marker</b> — High experimental risk</p>',
            unsafe_allow_html=True
        )
    with col_l2:
        st.markdown(
            '<p style="color:#9CA3AF;font-size:0.78rem;">🟢 <b>Green Marker</b> — Low experimental risk</p>',
            unsafe_allow_html=True
        )
    with col_l3:
        st.markdown(
            '<p style="color:#9CA3AF;font-size:0.78rem;">🟠 <b>Orange Cone</b> — Wind scenario</p>',
            unsafe_allow_html=True
        )
    with col_l4:
        st.markdown(
            '<p style="color:#9CA3AF;font-size:0.78rem;">🔵 <b>Clusters</b> — MODIS Hotspots</p>',
            unsafe_allow_html=True
        )
    
    # ── Export Report Button ──
    st.markdown("---")
    if st.button("📄 Export Risk Assessment Report (PDF)", use_container_width=True):
        try:
            from src.utils.pdf_report import generate_fire_report
            
            # Prepare data for PDF
            location_data = {
                'lat': sim_lat,
                'lon': sim_lon,
                'place_name': st.session_state.get('last_search_place', '')
            }
            
            weather_data = {
                'temp': sim_temp,
                'humidity': sim_humidity,
                'wind_speed': sim_wind_speed,
                'wind_deg': sim_wind_deg
            }
            
            risk_data = {
                'risk_level': st.session_state.get('risk_level', -1)
            }
            
            # Generate hotspot stats if data available
            hotspot_stats = None
            if not df.empty and 'acq_date' in df.columns:
                df_analysis = df.copy()
                df_analysis['acq_date'] = pd.to_datetime(df_analysis['acq_date'])
                daily_counts = df_analysis.groupby('acq_date').size()
                date_range = (df_analysis['acq_date'].max() - df_analysis['acq_date'].min()).days
                
                hotspot_stats = {
                    'total': len(df_analysis),
                    'avg_per_day': daily_counts.mean(),
                    'peak_day': daily_counts.max(),
                    'date_range': date_range
                }
            
            # Generate PDF
            pdf_buffer = generate_fire_report(
                location_data=location_data,
                weather_data=weather_data,
                risk_data=risk_data,
                hotspot_stats=hotspot_stats
            )
            
            # Download button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name=f"pyrowatch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.success("✅ Report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate report: {str(e)}")


# ══════════════════════════════════════════════════════════════
# TAB 2: AI Vision Lab — Satellite Image Analysis
# ══════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="section-header">
        <h3>🛰️ Satellite & Drone Imagery Analysis</h3>
        <div class="section-divider"></div>
    </div>
    <p style="color:#9CA3AF;font-size:0.9rem;margin-bottom:1.5rem;">
        Upload aerial, satellite, or drone imagery to analyze fire presence using our custom-trained 
        Vision Transformer (ViT) deep learning model.
    </p>
    """, unsafe_allow_html=True)

    # Upload area with visual cue
    st.markdown("""
    <div class="upload-zone">
        <span class="upload-icon">📸</span>
        <div class="upload-title">Drag & drop imagery or browse files</div>
        <div class="upload-subtitle">Supports JPG, JPEG, and PNG • Max 200MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload satellite or drone imagery",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Side-by-side layout: image + results
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("""
            <div class="section-header">
                <h3>📷 Uploaded Image</h3>
                <div class="section-divider"></div>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)

            # File info
            file_size = uploaded_file.size / 1024
            st.markdown(
                f'<p style="color:#6B7280;font-size:0.75rem;margin-top:0.5rem;">'
                f'📁 {uploaded_file.name} • {file_size:.1f} KB • {image.size[0]}×{image.size[1]}px</p>',
                unsafe_allow_html=True,
            )

        with col_result:
            st.markdown("""
            <div class="section-header">
                <h3>🧠 Analysis Results</h3>
                <div class="section-divider"></div>
            </div>
            """, unsafe_allow_html=True)

            # Save temp image for ViT
            temp_dir = os.path.join(ROOT_DIR, "data", "raw", "images")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "temp_upload.jpg")
            image.convert("RGB").save(temp_path)

            if st.button("🔍  Run Deep Learning Analysis", use_container_width=True):
                with st.spinner("Analyzing spatial features with Vision Transformer..."):
                    analysis_result = vit_model.analyze_image(temp_path)

                    if analysis_result:
                        # Handle new return structure (dict with 'results' and 'attention_heatmap')
                        if isinstance(analysis_result, dict):
                            results = analysis_result.get('results', [])
                            attention_heatmap = analysis_result.get('attention_heatmap')
                            error_msg = analysis_result.get('error')
                        else:
                            # Backward compatibility: if old format (list), use it directly
                            results = analysis_result
                            attention_heatmap = None
                            error_msg = None
                        
                        if results:
                            top_result = results[0]
                            is_fire = "Fire" in top_result["label"] and "No" not in top_result["label"]
                            confidence = top_result["confidence"]
                            
                            # Log to alert history
                            from datetime import datetime
                            alert_entry = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'image_name': uploaded_file.name,
                                'classification': top_result["label"],
                                'confidence': confidence,
                                'is_fire': is_fire
                            }
                            st.session_state['alert_history'].insert(0, alert_entry)  # Add to beginning
                            # Keep only last 50 entries
                            if len(st.session_state['alert_history']) > 50:
                                st.session_state['alert_history'] = st.session_state['alert_history'][:50]

                        # Primary result card
                        card_class = "result-fire" if is_fire else "result-safe"
                        result_icon = "🔥" if is_fire else "🌲"
                        result_text = "FIRE DETECTED" if is_fire else "NO FIRE DETECTED"
                        result_color = "#EF4444" if is_fire else "#10B981"

                        st.markdown(f"""
                        <div class="analysis-card {card_class}">
                            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
                                <span style="font-size:2.2rem;">{result_icon}</span>
                                <div>
                                    <div style="font-size:0.7rem;color:#6B7280;text-transform:uppercase;
                                                letter-spacing:1px;font-weight:600;">Classification</div>
                                    <div style="font-size:1.3rem;font-weight:800;color:{result_color};">
                                        {result_text}
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Confidence bars for all results
                        bars_html = ""
                        for res in results:
                            conf = res["confidence"]
                            label = res["label"]
                            is_fire_label = "Fire" in label and "No" not in label
                            bar_class = "conf-bar-fire" if is_fire_label else "conf-bar-safe"

                            bars_html += f"""
                            <div class="conf-bar-container">
                                <div class="conf-bar-label">
                                    <span>{label}</span>
                                    <span>{conf:.1f}%</span>
                                </div>
                                <div class="conf-bar-track">
                                    <div class="conf-bar-fill {bar_class}" style="width:{conf}%;"></div>
                                </div>
                            </div>
                            """

                        st.markdown(bars_html + "</div>", unsafe_allow_html=True)

                        # Alert banner
                        if is_fire:
                            st.markdown("""
                            <div class="risk-badge risk-high" style="margin-top:1rem;justify-content:center;">
                                <span class="risk-icon">⚠️</span>
                                <span>Active fire signature detected in imagery. Recommend immediate verification.</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="risk-badge risk-low" style="margin-top:1rem;justify-content:center;">
                                <span class="risk-icon">✅</span>
                                <span>No fire signatures detected. Area appears clear.</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display attention heatmap if available
                        if attention_heatmap is not None:
                            st.markdown("""
                            <div class="section-header" style="margin-top:2rem;">
                                <h3>🔍 Attention Heatmap</h3>
                                <div class="section-divider"></div>
                            </div>
                            <p style="color:#9CA3AF;font-size:0.85rem;margin-bottom:1rem;">
                                Visualization showing which image regions the AI model focused on during analysis.
                                <span style="color:#F59E0B;">Warm colors</span> indicate high attention, 
                                <span style="color:#3B82F6;">cool colors</span> indicate low attention.
                            </p>
                            """, unsafe_allow_html=True)
                            
                            col_orig, col_heat = st.columns(2)
                            with col_orig:
                                st.markdown("**Original Image**")
                                st.image(image, use_container_width=True)
                            with col_heat:
                                st.markdown("**Attention Overlay**")
                                st.image(attention_heatmap, use_container_width=True)
                            
                            # Download button for heatmap
                            from PIL import Image as PILImage
                            import io
                            
                            heatmap_pil = PILImage.fromarray(attention_heatmap)
                            buf = io.BytesIO()
                            heatmap_pil.save(buf, format='PNG')
                            buf.seek(0)
                            
                            st.download_button(
                                label="📥 Download Attention Heatmap",
                                data=buf,
                                file_name=f"attention_heatmap_{uploaded_file.name}",
                                mime="image/png",
                                use_container_width=True
                            )
                        elif error_msg:
                            st.markdown(f"""
                            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
                                        border-radius:8px;padding:1rem;margin-top:1rem;">
                                <span style="color:#FCA5A5;font-size:0.85rem;">
                                    ⚠️ Attention visualization unavailable: {error_msg}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("Analysis failed. Please check the image and try again.")
            else:
                st.markdown("""
                <div class="analysis-card" style="text-align:center;padding:3rem;">
                    <span style="font-size:3rem;display:block;margin-bottom:0.75rem;">🧠</span>
                    <div style="color:#9CA3AF;font-size:0.9rem;font-weight:500;">
                        Click <b>Run Deep Learning Analysis</b> to classify this image
                    </div>
                    <div style="color:#6B7280;font-size:0.78rem;margin-top:0.4rem;">
                        Powered by a custom Vision Transformer (ViT) fine-tuned on wildfire imagery
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: Analytics — Historical Trends
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <h3>📊 Historical Hotspot Trends</h3>
        <div class="section-divider"></div>
    </div>
    <p style="color:#9CA3AF;font-size:0.9rem;margin-bottom:1.5rem;">
        Analyze fire hotspot patterns over time from MODIS satellite data
    </p>
    """, unsafe_allow_html=True)
    
    if not df.empty and 'acq_date' in df.columns:
        # Convert date column to datetime
        df_analysis = df.copy()
        df_analysis['acq_date'] = pd.to_datetime(df_analysis['acq_date'])
        
        # Daily hotspot counts
        daily_counts = df_analysis.groupby('acq_date').size().reset_index(name='count')
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 🔥 Daily Hotspot Detections")
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_counts['acq_date'],
                y=daily_counts['count'],
                mode='lines+markers',
                name='Hotspots',
                line=dict(color='#FF6B35', width=2),
                marker=dict(size=6, color='#E63946'),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 53, 0.1)'
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9CA3AF'),
                xaxis=dict(
                    title='Date',
                    gridcolor='rgba(255,255,255,0.1)',
                    showgrid=True
                ),
                yaxis=dict(
                    title='Number of Hotspots',
                    gridcolor='rgba(255,255,255,0.1)',
                    showgrid=True
                ),
                hovermode='x unified',
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            st.markdown("### 🌡️ Confidence Distribution")
            
            if 'confidence' in df_analysis.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=df_analysis['confidence'],
                    nbinsx=20,
                    marker=dict(
                        color='#FF6B35',
                        line=dict(color='#E63946', width=1)
                    ),
                    name='Confidence'
                ))
                
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#9CA3AF'),
                    xaxis=dict(
                        title='Confidence Level (%)',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title='Frequency',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    height=350
                )
                
                st.plotly_chart(fig2, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            total_hotspots = len(df_analysis)
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">🔥</span>
                <div class="metric-label">Total Hotspots</div>
                <div class="metric-value">{total_hotspots:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            avg_per_day = daily_counts['count'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">📅</span>
                <div class="metric-label">Avg Per Day</div>
                <div class="metric-value">{avg_per_day:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat3:
            max_day = daily_counts.loc[daily_counts['count'].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <span class="metric-icon">📊</span>
                <div class="metric-label">Peak Day</div>
                <div class="metric-value">{max_day['count']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat4:
            date_range = (df_analysis['acq_date'].max() - df_analysis['acq_date'].min()).days
            
            # Show more meaningful info for single-day data
            if date_range == 0:
                # Single day data - show the date instead
                single_date = df_analysis['acq_date'].max().strftime("%b %d, %Y")
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-icon">📅</span>
                    <div class="metric-label">Data Date</div>
                    <div class="metric-value" style="font-size:1.1rem;">{single_date}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-icon">🗓️</span>
                    <div class="metric-label">Date Range</div>
                    <div class="metric-value">{date_range}<span class="metric-unit">days</span></div>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.info("📊 No hotspot data available. Click 'Refresh Hotspot Data' in the sidebar to load data.")


# ─────────────────────────────────────────────────────────────
# 7. Footer
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pyro-footer">
    <p style="font-weight:600;color:#9CA3AF;">PyroWatch v2.0 — AI Wildfire Intelligence Platform</p>
    <p>Random Forest Risk Engine • Vision Transformer (ViT) Image Classifier • Wind-Driven Spread Simulator</p>
    <p>Data: <a href="https://firms.modaps.eosdis.nasa.gov/" target="_blank">NASA FIRMS</a> • 
       Weather: <a href="https://openweathermap.org/" target="_blank">OpenWeatherMap</a> •
       Built {datetime.now().year}</p>
</div>
""", unsafe_allow_html=True)

