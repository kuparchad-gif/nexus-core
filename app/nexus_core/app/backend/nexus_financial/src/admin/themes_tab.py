```python
import streamlit as st
import yaml
from pathlib import Path
from ..utils.logging_config import logger

def apply_theme_config(config: dict):
    try:
        with open("config/theme.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        logger.info("Theme configuration saved to config/theme.yaml")
        st.success("Theme applied; restart dashboard to see changes")
    except Exception as e:
        logger.error(f"Failed to apply theme: {e}")
        st.error(f"Failed to apply theme: {e}")

def themes_tab():
    st.header("🎨 Visual Theme Customizer")
    
    theme = st.selectbox("Theme Preset", [
        "Cyberpunk", "Bloomberg Terminal", "Dark Matrix", 
        "Light Professional", "High Contrast", "Custom"
    ])
    
    primary, background, secondary, accent = "#00FF88", "#0A0A0A", "#0088FF", "#FF0088"
    if theme == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            primary = st.color_picker("Primary Color", primary)
            background = st.color_picker("Background", background)
        with col2:
            secondary = st.color_picker("Secondary", secondary) 
            accent = st.color_picker("Accent", accent)
        
        st.subheader("Live Preview")
        css = f"""
        <style>
        .preview-box {{
            background: {background};
            color: {primary};
            padding: 20px;
            border: 2px solid {accent};
            border-radius: 10px;
        }}
        </style>
        <div class="preview-box">
        🚀 Your Custom Theme Preview
        </div>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    st.subheader("Chart Styles")
    chart_style = st.selectbox("Chart Engine", ["Plotly", "LightweightCharts", "TradingView"])
    candle_style = st.selectbox("Candlestick Style", ["Classic", "Hollow", "Heikin Ashi", "Renko"])
    
    if st.button("Apply Theme"):
        apply_theme_config({
            "colors": {"primary": primary, "background": background, "secondary": secondary, "accent": accent},
            "charts": {"engine": chart_style, "candle_style": candle_style}
        })
```