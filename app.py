#!/usr/bin/env python3
"""
UK Research Position Interactive Explorer

A Streamlit app for exploring UK research positioning across 4,516 topics.

Deployed version for Streamlit Community Cloud.

Views:
    1. Topic Browser - Filterable table with search
    2. Topic Detail - Deep dive with trajectory chart + country comparison
    3. Bloc Comparison - Three blocs analysis
    4. Country Analysis - 46 country comparison
    5. UK Strengths Dashboard - Momentum summary
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from pathlib import Path

# Configuration - use relative path for deployment
OUTPUT_DIR = Path(__file__).parent / "data"

# Color palette (matching static visualizations)
COLORS = {
    'uk_advantage': '#2E7D32',
    'uk_neutral': '#757575',
    'uk_concern': '#F57C00',
    'uk_erosion': '#C62828',
    'uk_accent': '#1976D2',
    'peer_average': '#9E9E9E',
}

POSITION_COLORS = {
    'STRONG_ADVANTAGE': '#2E7D32',
    'EMERGING_ADVANTAGE': '#4CAF50',
    'OUTPACING_PEERS': '#66BB6A',
    'RELATIVE_ADVANTAGE': '#81C784',
    'NEUTRAL': '#9E9E9E',
    'UK_SPECIFIC_DECLINE': '#EF5350',
    'DECLINING_SLOWER': '#FFCA28',
    'STRATEGIC_CONCERN': '#C62828',
    'RIDING_WAVE': '#2E7D32',
    'FALLING_BEHIND_RELATIVELY': '#F57C00',
    'GAINING_GROUND': '#4CAF50',
    'HOLDING_POSITION': '#9E9E9E',
    'LOSING_GROUND': '#EF5350',
    'DOUBLING_DOWN': '#66BB6A',
    'DECLINING_WITH_FIELD': '#BDBDBD',
    'APPROPRIATE_EXIT': '#9E9E9E',
}

# Trajectory pattern icons for consistent display
PATTERN_ICONS = {
    'ACCELERATING': '🚀',
    'RECOVERING': '📈',
    'STEADY': '➡️',
    'DECELERATING': '📉',
    'RAPID_RETREAT': '🚨',
}


@st.cache_data
def load_data():
    """Load all required datasets."""
    data = {}

    # Phase 2 data
    data['trajectories'] = pl.read_parquet(OUTPUT_DIR / "phase2_topic_trajectories.parquet")
    data['windows'] = pl.read_parquet(OUTPUT_DIR / "phase2_window_aggregates.parquet")

    # Phase 2b data - country-level trajectories and window aggregates
    data['country_trajectories'] = pl.read_parquet(OUTPUT_DIR / "phase2b_country_trajectories.parquet")
    data['country_windows'] = pl.read_parquet(OUTPUT_DIR / "phase2b_country_window_aggregates.parquet")

    # Phase 2c data - pre-computed bloc trends
    data['bloc_trends'] = pl.read_parquet(OUTPUT_DIR / "phase2c_bloc_trends.parquet")

    # Phase 3b data
    data['uk_advantage'] = pl.read_parquet(OUTPUT_DIR / "phase3b_uk_comparative_advantage.parquet")
    data['country_profiles'] = pl.read_parquet(OUTPUT_DIR / "phase3b_country_profiles.parquet")
    data['country_dynamics'] = pl.read_parquet(OUTPUT_DIR / "phase3b_country_dynamics.parquet")

    # Phase 3 peer comparison
    data['peer_comparison'] = pl.read_parquet(OUTPUT_DIR / "phase3_peer_comparison.parquet")

    return data


def format_number(n):
    """Format number with commas."""
    if n is None:
        return "N/A"
    return f"{n:,.0f}"


def format_percent(p, include_sign=True):
    """Format percentage."""
    if p is None:
        return "N/A"
    if include_sign and p > 0:
        return f"+{p:.1f}%"
    return f"{p:.1f}%"


def format_pp(p, include_sign=True):
    """Format percentage points."""
    if p is None:
        return "N/A"
    if include_sign and p > 0:
        return f"+{p:.2f}pp"
    return f"{p:.2f}pp"


def format_enum(value):
    """Format enum values for display (RIDING_WAVE -> Riding Wave)."""
    if value is None:
        return "N/A"
    return value.replace('_', ' ').title()


# Three blocs for strategic comparison (Paul Nurse framing)
BLOC_EUROPE = [
    # EU27
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
    'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK',
    'SI', 'ES', 'SE',
    # EFTA + associates
    'CH', 'NO', 'IS',
    # Close collaborators
    'IL',
]
BLOC_USA = ['US']
BLOC_CHINA = ['CN']

# Country code to name mapping
COUNTRY_NAMES = {
    'US': 'United States', 'CN': 'China', 'GB': 'United Kingdom', 'DE': 'Germany',
    'FR': 'France', 'JP': 'Japan', 'IN': 'India', 'IT': 'Italy', 'CA': 'Canada',
    'AU': 'Australia', 'ES': 'Spain', 'BR': 'Brazil', 'KR': 'South Korea',
    'NL': 'Netherlands', 'RU': 'Russia', 'TR': 'Turkey', 'PL': 'Poland',
    'SE': 'Sweden', 'CH': 'Switzerland', 'BE': 'Belgium', 'AT': 'Austria',
    'IL': 'Israel', 'DK': 'Denmark', 'NO': 'Norway', 'FI': 'Finland',
    'PT': 'Portugal', 'IE': 'Ireland', 'NZ': 'New Zealand', 'SG': 'Singapore',
    'SA': 'Saudi Arabia', 'ZA': 'South Africa', 'MX': 'Mexico', 'TW': 'Taiwan',
    'CZ': 'Czech Republic', 'GR': 'Greece', 'HU': 'Hungary', 'CL': 'Chile',
    'CO': 'Colombia', 'MY': 'Malaysia', 'TH': 'Thailand', 'EG': 'Egypt',
    'PK': 'Pakistan', 'IR': 'Iran', 'AR': 'Argentina', 'UA': 'Ukraine',
}


def get_country_name(code):
    """Get country name from code."""
    return COUNTRY_NAMES.get(code, code)


# =============================================================================
# VIEW 1: Topic Browser
# =============================================================================

def topic_browser(data):
    """Filterable table of all topics."""
    st.header("Topic Browser")
    st.markdown("Explore UK research positioning across 4,516 topics.")

    df = data['trajectories']
    bloc_trends = data['bloc_trends']

    # Join bloc trends to trajectories
    df = df.join(bloc_trends.select([
        "topic_id", "europe_trend", "usa_trend", "china_trend", "global_trend", "interpretation"
    ]), on="topic_id", how="left")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fields = ["All Fields"] + sorted([f for f in df["field_name"].unique().to_list() if f is not None])
        selected_field = st.selectbox("Field", fields)

    with col2:
        # Interpretation filter (bloc-based context)
        interpretations = ["All Contexts"] + sorted([i for i in df["interpretation"].unique().to_list() if i is not None])
        selected_interp = st.selectbox("Context", interpretations,
                                       help="Bloc-based interpretation of UK position")

    with col3:
        patterns = ["All Patterns"] + sorted([p for p in df["trajectory_pattern"].unique().to_list() if p is not None])
        selected_pattern = st.selectbox("UK Momentum", patterns)

    with col4:
        search = st.text_input("Search topics", "")

    # Apply filters
    filtered = df

    if selected_field != "All Fields":
        filtered = filtered.filter(pl.col("field_name") == selected_field)

    if selected_interp != "All Contexts":
        filtered = filtered.filter(pl.col("interpretation") == selected_interp)

    if selected_pattern != "All Patterns":
        filtered = filtered.filter(pl.col("trajectory_pattern") == selected_pattern)

    if search:
        filtered = filtered.filter(
            pl.col("topic_name").str.to_lowercase().str.contains(search.lower())
        )

    # Display count and legend
    st.markdown(f"**{len(filtered):,} topics** matching filters")
    st.caption("Bloc trends (↑ growing, → stable, ↓ declining) based on recent 5-year share changes")

    # Prepare display dataframe
    display_df = filtered.select([
        "topic_name",
        "field_name",
        "current_uk_papers",
        "current_uk_share",
        "trajectory_pattern",
        "europe_trend",
        "usa_trend",
        "china_trend",
        "global_trend",
    ]).sort("current_uk_papers", descending=True)

    # Convert to pandas and format
    pandas_df = display_df.to_pandas()

    # Format trajectory pattern with icon
    def format_pattern(p):
        if p is None:
            return "N/A"
        icons = {'ACCELERATING': '🚀', 'RECOVERING': '📈', 'STEADY': '➡️',
                 'DECELERATING': '📉', 'RAPID_RETREAT': '🚨'}
        return f"{icons.get(p, '?')} {p.replace('_', ' ').title()}"

    pandas_df["trajectory_pattern"] = pandas_df["trajectory_pattern"].apply(format_pattern)

    # Format bloc trends as arrow indicators
    def format_trend(t):
        if t is None:
            return "?"
        icons = {'GROWING': '↑', 'STABLE': '→', 'DECLINING': '↓'}
        return icons.get(t, '?')

    pandas_df["🇪🇺+"] = pandas_df["europe_trend"].apply(format_trend)
    pandas_df["🇺🇸"] = pandas_df["usa_trend"].apply(format_trend)
    pandas_df["🇨🇳"] = pandas_df["china_trend"].apply(format_trend)
    pandas_df["🌍"] = pandas_df["global_trend"].apply(format_trend)

    # Drop original trend columns and rename
    pandas_df = pandas_df.drop(columns=["europe_trend", "usa_trend", "china_trend", "global_trend"])
    pandas_df = pandas_df.rename(columns={
        "topic_name": "Topic",
        "field_name": "Field",
        "current_uk_papers": "UK Papers",
        "current_uk_share": "UK Share %",
        "trajectory_pattern": "🇬🇧 Momentum",
    })

    # Reorder columns
    pandas_df = pandas_df[["Topic", "Field", "UK Papers", "UK Share %", "🇬🇧 Momentum", "🇪🇺+", "🇺🇸", "🇨🇳", "🌍"]]

    # Clickable table
    event = st.dataframe(
        pandas_df,
        height=400,
        hide_index=True,
        column_config={
            "Topic": st.column_config.TextColumn(width=280),
            "Field": st.column_config.TextColumn(width=150),
            "UK Papers": st.column_config.NumberColumn(format="%d", width=80),
            "UK Share %": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "🇬🇧 Momentum": st.column_config.TextColumn(width=140),
            "🇪🇺+": st.column_config.TextColumn(width=40),
            "🇺🇸": st.column_config.TextColumn(width=40),
            "🇨🇳": st.column_config.TextColumn(width=40),
            "🌍": st.column_config.TextColumn(width=40),
        },
        selection_mode="single-row",
        on_select="rerun",
        key="topic_table"
    )

    # Handle row selection
    selected_topic = None
    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_topic = pandas_df.iloc[selected_idx]["Topic"]

    # Navigation section
    st.markdown("---")

    if selected_topic:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**Selected:** {selected_topic}")
        with col2:
            if st.button("View Details →", type="primary"):
                st.session_state['selected_topic'] = selected_topic
                st.session_state['navigate_to'] = "Topic Detail"
                # Clear radio widget state so it picks up the new index
                if 'view_selector' in st.session_state:
                    del st.session_state['view_selector']
                st.rerun()
    else:
        st.caption("Click a row above to select a topic, then navigate to details.")


# =============================================================================
# VIEW 2: Topic Detail
# =============================================================================

# Pattern interpretations for policy guidance
PATTERN_INTERPRETATIONS = {
    'RAPID_RETREAT': {
        'icon': '🚨',
        'summary': 'Rapid Retreat',
        'description': 'UK share was stable or growing historically but is now actively declining.',
        'guidance': 'Investigate urgently — this was a UK strength that is now eroding.',
        'color': '#C62828',
    },
    'DECELERATING': {
        'icon': '⚠️',
        'summary': 'Decelerating',
        'description': 'UK share growth is slowing — momentum is fading.',
        'guidance': 'Watch closely — early warning sign that may require attention.',
        'color': '#F57C00',
    },
    'STEADY': {
        'icon': 'ℹ️',
        'summary': 'Steady',
        'description': 'UK share trajectory is consistent over time.',
        'guidance': 'Evaluate strategically — predictable trajectory.',
        'color': '#9E9E9E',
    },
    'RECOVERING': {
        'icon': '📈',
        'summary': 'Recovering',
        'description': 'UK share was declining but is now growing again.',
        'guidance': 'Monitor positively — turnaround in progress.',
        'color': '#4CAF50',
    },
    'ACCELERATING': {
        'icon': '✅',
        'summary': 'Accelerating',
        'description': 'UK share is rising faster recently than historically.',
        'guidance': 'Success in progress — UK is gaining momentum.',
        'color': '#2E7D32',
    },
}


def topic_detail(data):
    """Deep dive on a single topic."""
    st.header("Topic Detail")

    df = data['trajectories']
    windows = data['windows']
    country_dyn = data['country_dynamics']
    uk_adv = data['uk_advantage']

    # Topic selector
    topic_names = sorted([t for t in df["topic_name"].unique().to_list() if t is not None])

    # Use session state if available
    default_idx = 0
    if 'selected_topic' in st.session_state:
        try:
            default_idx = topic_names.index(st.session_state['selected_topic'])
        except ValueError:
            pass

    selected_topic = st.selectbox("Select Topic", topic_names, index=default_idx)

    # Get topic data
    topic_row = df.filter(pl.col("topic_name") == selected_topic).row(0, named=True)
    topic_id = topic_row['topic_id']

    # Header info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(selected_topic)
        st.markdown(f"**Field:** {topic_row['field_name']}")
        st.markdown(f"**Subfield:** {topic_row['subfield_name']}")
        st.markdown(f"**Domain:** {topic_row['domain_name']}")

    with col2:
        # TRAJECTORY PATTERN FIRST (most policy-relevant)
        pattern = topic_row.get('trajectory_pattern', 'STEADY')
        pattern_info = PATTERN_INTERPRETATIONS.get(pattern, PATTERN_INTERPRETATIONS['STEADY'])

        st.markdown(f"""
        <div style="background-color: {pattern_info['color']}; color: white; padding: 12px; border-radius: 5px; text-align: center;">
            <span style="font-size: 1.2em;">{pattern_info['icon']}</span>
            <strong style="font-size: 1.1em;">{pattern_info['summary']}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Get pre-computed bloc trends (Europe, USA, China, Global)
        bloc_trends_row = data['bloc_trends'].filter(pl.col("topic_id") == topic_id)
        if len(bloc_trends_row) > 0:
            bt = bloc_trends_row.row(0, named=True)
            bloc_trends = {
                'europe': {'trend': bt.get('europe_trend', 'STABLE'), 'slope': bt.get('europe_slope', 0)},
                'usa': {'trend': bt.get('usa_trend', 'STABLE'), 'slope': bt.get('usa_slope', 0)},
                'china': {'trend': bt.get('china_trend', 'STABLE'), 'slope': bt.get('china_slope', 0)},
                'uk': {'trend': bt.get('uk_trend', 'STABLE'), 'slope': bt.get('uk_slope', 0)},
                'global': {'trend': bt.get('global_trend', 'STABLE'), 'slope': bt.get('global_slope', 0)},
            }
        else:
            bloc_trends = {b: {'trend': 'STABLE', 'slope': 0} for b in ['europe', 'usa', 'china', 'uk', 'global']}

        # Bloc trends display
        trend_icons = {'GROWING': '↑', 'STABLE': '→', 'DECLINING': '↓'}
        trend_colors = {'GROWING': '#4CAF50', 'STABLE': '#9E9E9E', 'DECLINING': '#FF9800'}

        st.markdown("""
        <div style="background-color: #f5f5f5; padding: 8px; border-radius: 5px; margin-top: 8px;">
            <div style="font-size: 0.75em; color: #666; margin-bottom: 4px;">Recent trends (2016-24):</div>
        """, unsafe_allow_html=True)

        bloc_html = []
        for bloc_name, bloc_label in [('europe', '🇪🇺 EU+'), ('usa', '🇺🇸 USA'), ('china', '🇨🇳 China'), ('global', '🌍 Global')]:
            trend = bloc_trends.get(bloc_name, {}).get('trend', 'STABLE')
            icon = trend_icons[trend]
            color = trend_colors[trend]
            bloc_html.append(f'<span style="color: {color}; font-weight: bold;">{bloc_label} {icon}</span>')

        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                {''.join(f'<span>{b}</span>' for b in bloc_html)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interpretation box using THREE BLOCS framing (Paul Nurse)
    st.markdown("---")

    # Extract bloc trends
    europe_trend = bloc_trends.get('europe', {}).get('trend', 'STABLE')
    usa_trend = bloc_trends.get('usa', {}).get('trend', 'STABLE')
    china_trend = bloc_trends.get('china', {}).get('trend', 'STABLE')
    global_trend = bloc_trends.get('global', {}).get('trend', 'STABLE')

    # Count declining blocs
    declining_blocs = sum(1 for t in [europe_trend, usa_trend, china_trend] if t == 'DECLINING')
    growing_blocs = sum(1 for t in [europe_trend, usa_trend, china_trend] if t == 'GROWING')

    # Build bloc summary for messages
    def bloc_summary():
        parts = []
        if europe_trend == 'DECLINING':
            parts.append("Europe ↓")
        elif europe_trend == 'GROWING':
            parts.append("Europe ↑")
        if usa_trend == 'DECLINING':
            parts.append("USA ↓")
        elif usa_trend == 'GROWING':
            parts.append("USA ↑")
        if china_trend == 'DECLINING':
            parts.append("China ↓")
        elif china_trend == 'GROWING':
            parts.append("China ↑")
        return ", ".join(parts) if parts else "all stable"

    # Interpretation based on UK pattern + bloc trends
    is_concern = topic_row.get('strategic_concern', False)

    if pattern == 'RAPID_RETREAT':
        if declining_blocs >= 2:
            # All major blocs declining - structural shift
            st.info(f"**🌍 Structural shift.** UK retreating, but so are major blocs ({bloc_summary()}). "
                    f"This appears to be a global rebalancing, not a UK-specific problem.")
        elif europe_trend == 'DECLINING' and usa_trend != 'GROWING':
            # Europe declining, UK following European pattern
            st.info(f"**🇪🇺 European pattern.** UK retreating alongside Europe ({bloc_summary()}). "
                    f"This reflects broader European dynamics.")
        elif china_trend == 'GROWING' and declining_blocs >= 1:
            # China consolidating while West retreats
            st.warning(f"**🇨🇳 China consolidating.** UK retreating while China grows ({bloc_summary()}). "
                       f"Consider strategic implications of ceding ground.")
        elif europe_trend != 'DECLINING' and usa_trend != 'DECLINING':
            # UK retreating alone while peers stable
            st.error(f"**🚨 UK-specific retreat.** UK share declining while Europe and USA remain stable. "
                     f"This warrants investigation.")
        else:
            st.warning(f"**{pattern_info['icon']} {pattern_info['description']}** Bloc trends: {bloc_summary()}.")

    elif pattern == 'DECELERATING':
        if declining_blocs >= 2:
            st.info(f"**ℹ️ Slowing with the field.** UK momentum fading, but major blocs also declining ({bloc_summary()}).")
        else:
            st.warning(f"**{pattern_info['icon']} {pattern_info['description']}** {pattern_info['guidance']}")

    elif pattern == 'ACCELERATING':
        if china_trend == 'GROWING' and europe_trend != 'GROWING':
            st.success(f"**🚀 Competing with China.** UK accelerating in a field where China is also growing. "
                       f"Positive momentum in a competitive space.")
        elif declining_blocs >= 2:
            st.warning(f"**⚠️ Counter-trend growth.** UK accelerating while major blocs retreat ({bloc_summary()}). "
                       f"Evaluate whether this positioning is strategic.")
        elif is_concern:
            st.success(f"**{pattern_info['icon']} Catching up.** UK share rising faster than ever — success in progress.")
        else:
            st.success(f"**{pattern_info['icon']} {pattern_info['description']}** {pattern_info['guidance']}")

    elif pattern == 'RECOVERING':
        if declining_blocs >= 2:
            st.info(f"**📈 Recovery against the tide.** UK rebounding while major blocs decline ({bloc_summary()}). "
                    f"Consider whether this recovery is strategically valuable.")
        else:
            st.info(f"**{pattern_info['icon']} {pattern_info['description']}** {pattern_info['guidance']}")

    else:  # STEADY
        st.info(f"**{pattern_info['icon']} {pattern_info['description']}** {pattern_info['guidance']}")

    # Key metrics
    st.markdown("### Key Metrics (2020-2024)")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("UK Papers", format_number(topic_row['current_uk_papers']))
    with m2:
        st.metric("Global Papers", format_number(topic_row['current_global_papers']))
    with m3:
        st.metric("UK Share", format_percent(topic_row['current_uk_share'] * 100 if topic_row['current_uk_share'] else None, include_sign=False))
    with m4:
        global_trend = bloc_trends.get('global', {}).get('trend', 'STABLE')
        st.metric("Global Trend (Recent)", global_trend.replace('_', ' ').title())

    # Momentum metrics
    st.markdown("### UK Momentum")
    m5, m6, m7, m8 = st.columns(4)
    with m5:
        early_slope = topic_row.get('early_uk_share_slope', 0)
        st.metric("Long-term Trend (2010-17)", f"{early_slope*100:+.3f}pp/window" if early_slope else "N/A")
    with m6:
        recent_slope = topic_row.get('recent_uk_share_slope', 0)
        st.metric("Short-term Trend (2016-24)", f"{recent_slope*100:+.3f}pp/window" if recent_slope else "N/A")
    with m7:
        peak_share = topic_row.get('peak_uk_share', 0)
        st.metric("Peak UK Share", format_percent(peak_share * 100 if peak_share else None, include_sign=False))
    with m8:
        peak_window = topic_row.get('peak_uk_share_window', 'N/A')
        st.metric("Peak Window", peak_window)

    # Trajectory chart with early/recent shading
    st.markdown("---")
    st.markdown("### UK Share Trajectory (2010-2024)")

    topic_windows = windows.filter(pl.col("topic_id") == topic_id).sort("window_start")

    if len(topic_windows) > 0:
        window_labels = topic_windows["time_window"].to_list()
        uk_shares = [s * 100 for s in topic_windows["uk_share_of_topic"].to_list()]

        fig = go.Figure()

        # Add shaded regions for early vs recent periods
        y_min = min(uk_shares) * 0.9 if uk_shares else 0
        y_max = max(uk_shares) * 1.1 if uk_shares else 10

        # Early period shading (windows 0-5)
        fig.add_vrect(
            x0=-0.5, x1=5.5,
            fillcolor="rgba(33, 150, 243, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="Long-term",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="rgba(33, 150, 243, 0.7)"
        )

        # Recent period shading (windows 6-10)
        fig.add_vrect(
            x0=5.5, x1=10.5,
            fillcolor="rgba(76, 175, 80, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="Short-term",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="rgba(76, 175, 80, 0.7)"
        )

        # UK share line
        fig.add_trace(go.Scatter(
            x=list(range(len(window_labels))),
            y=uk_shares,
            mode='lines+markers',
            name='UK Share',
            line=dict(color=COLORS['uk_accent'], width=3),
            marker=dict(size=8),
            text=window_labels,
            hovertemplate='%{text}<br>UK Share: %{y:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(window_labels))),
                ticktext=[w[:4] for w in window_labels],
                title="5-Year Window"
            ),
            yaxis_title="UK Share (%)",
            height=350,
            margin=dict(l=50, r=50, t=30, b=50),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, key="trajectory_chart")
    else:
        st.info("No trajectory data available for this topic.")

    # Country comparison with trajectory patterns
    st.markdown("---")
    st.markdown("### Country Comparison")
    st.caption("Momentum patterns based on early (2010-17) vs recent (2016-24) trajectory slopes")

    # Use the new country trajectory data
    country_traj = data['country_trajectories']
    topic_countries = country_traj.filter(pl.col("topic_id") == topic_id)

    if len(topic_countries) > 0:
        # Get top countries by current papers
        top_countries = topic_countries.sort("current_papers", descending=True).head(15)

        # Check if UK is in the list
        uk_in_list = 'GB' in top_countries['country_code'].to_list()

        country_df = top_countries.select([
            "country_code",
            "current_papers",
            "current_share",
            "early_slope",
            "recent_slope",
            "trajectory_pattern"
        ]).to_pandas()

        # Format trajectory pattern with icons
        country_df["Pattern"] = country_df["trajectory_pattern"].apply(
            lambda p: f"{PATTERN_ICONS.get(p, '?')} {p.replace('_', ' ').title()}"
        )

        # Convert country codes to names and mark UK
        country_df["is_uk"] = country_df["country_code"] == "GB"
        country_df["Country"] = country_df["country_code"].apply(get_country_name)
        country_df["Country"] = country_df.apply(
            lambda r: f"🇬🇧 **{r['Country']}**" if r["is_uk"] else r["Country"],
            axis=1
        )

        # Format columns
        country_df["Share %"] = country_df["current_share"] * 100
        country_df["Long-term"] = country_df["early_slope"] * 100  # pp per window
        country_df["Short-term"] = country_df["recent_slope"] * 100

        # Select and order columns
        display_df = country_df[[
            "Country", "current_papers", "Share %", "Long-term", "Short-term", "Pattern"
        ]].copy()
        display_df.columns = ["Country", "Papers", "Share %", "Long-term (pp)", "Short-term (pp)", "Momentum"]

        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Country": st.column_config.TextColumn(width="medium"),
                "Papers": st.column_config.NumberColumn(format="%d"),
                "Share %": st.column_config.NumberColumn(format="%.2f"),
                "Long-term (pp)": st.column_config.NumberColumn(format="%+.3f"),
                "Short-term (pp)": st.column_config.NumberColumn(format="%+.3f"),
                "Momentum": st.column_config.TextColumn(width="medium"),
            }
        )

        if not uk_in_list:
            # Show UK separately if not in top 15
            uk_data = topic_countries.filter(pl.col("country_code") == "GB")
            if len(uk_data) > 0:
                uk_row = uk_data.row(0, named=True)
                pattern = uk_row['trajectory_pattern']
                icon = PATTERN_ICONS.get(pattern, '?')
                st.markdown(f"""
                ---
                **🇬🇧 United Kingdom** (not in top 15 by volume):
                Papers: {uk_row['current_papers']:,} | Share: {uk_row['current_share']*100:.2f}% |
                Long-term: {uk_row['early_slope']*100:+.3f}pp | Short-term: {uk_row['recent_slope']*100:+.3f}pp |
                **{icon} {pattern.replace('_', ' ').title()}**
                """)
    else:
        st.info("No country comparison data available for this topic.")

    # UK vs European peers momentum comparison
    st.markdown("---")
    st.markdown("### UK vs European Peers (Momentum)")
    st.caption("Comparing recent trend slopes with DE, FR, NL, BE, CH, AT")

    # Get European peers' trajectory data for this topic
    european_peers = ['DE', 'FR', 'NL', 'BE', 'CH', 'AT']
    peer_data = topic_countries.filter(pl.col("country_code").is_in(european_peers))
    uk_traj_data = topic_countries.filter(pl.col("country_code") == "GB")

    if len(uk_traj_data) > 0:
        uk_traj = uk_traj_data.row(0, named=True)
        uk_recent_slope = uk_traj['recent_slope']
        uk_pattern = uk_traj['trajectory_pattern']

        # Calculate peer average recent slope
        if len(peer_data) > 0:
            peer_avg_recent = peer_data['recent_slope'].mean()
            peer_patterns = peer_data['trajectory_pattern'].to_list()
        else:
            peer_avg_recent = 0
            peer_patterns = []

        col1, col2, col3 = st.columns(3)

        with col1:
            delta = (uk_recent_slope - peer_avg_recent) * 100 if peer_avg_recent else None
            st.metric(
                "UK Recent Trend",
                f"{uk_recent_slope*100:+.3f}pp/window",
                delta=f"{delta:+.3f}pp vs peers" if delta is not None else None,
                delta_color="normal"
            )

        with col2:
            if peer_avg_recent:
                st.metric(
                    "Peer Avg Recent Trend",
                    f"{peer_avg_recent*100:+.3f}pp/window"
                )

        with col3:
            # Show UK momentum pattern
            icon = PATTERN_ICONS.get(uk_pattern, '?')
            st.metric("UK Momentum", f"{icon} {uk_pattern.replace('_', ' ').title()}")

        # Show peer patterns if available
        if peer_patterns:
            peer_pattern_summary = {}
            for p in peer_patterns:
                peer_pattern_summary[p] = peer_pattern_summary.get(p, 0) + 1
            pattern_text = ", ".join([
                f"{PATTERN_ICONS.get(p, '?')} {p.replace('_', ' ').title()} ({c})"
                for p, c in sorted(peer_pattern_summary.items(), key=lambda x: -x[1])
            ])
            st.caption(f"Peer momentum patterns: {pattern_text}")


# =============================================================================
# VIEW 3: Bloc Comparison (Three Blocs framing)
# =============================================================================

def country_comparison(data):
    """Compare UK against the three major science blocs."""
    st.header("Bloc Comparison")
    st.markdown("Compare UK research dynamics against Europe, USA, and China — the three major science blocs.")

    bloc_trends = data['bloc_trends']

    # Bloc trend summary
    st.markdown("### Bloc Trend Overview")
    st.caption("How many topics is each bloc growing, stable, or declining in? (Recent 5-year trend)")

    # Calculate summary for each bloc
    bloc_summary = []
    for bloc_name, bloc_label, bloc_icon in [
        ('europe', 'Europe (EU+)', '🇪🇺'),
        ('usa', 'USA', '🇺🇸'),
        ('china', 'China', '🇨🇳'),
        ('uk', 'United Kingdom', '🇬🇧'),
    ]:
        col = f"{bloc_name}_trend"
        counts = bloc_trends.group_by(col).len()
        count_dict = {row[col]: row['len'] for row in counts.iter_rows(named=True)}

        growing = count_dict.get('GROWING', 0)
        stable = count_dict.get('STABLE', 0)
        declining = count_dict.get('DECLINING', 0)
        total = growing + stable + declining

        bloc_summary.append({
            'Bloc': f"{bloc_icon} {bloc_label}",
            'Growing ↑': growing,
            'Stable →': stable,
            'Declining ↓': declining,
            'Net': growing - declining,
            'Growing %': 100 * growing / total if total > 0 else 0,
        })

    summary_df = pl.DataFrame(bloc_summary).to_pandas()

    st.dataframe(
        summary_df,
        hide_index=True,
        column_config={
            "Bloc": st.column_config.TextColumn(width=180),
            "Growing ↑": st.column_config.NumberColumn(format="%d"),
            "Stable →": st.column_config.NumberColumn(format="%d"),
            "Declining ↓": st.column_config.NumberColumn(format="%d"),
            "Net": st.column_config.NumberColumn(format="%+d"),
            "Growing %": st.column_config.NumberColumn(format="%.1f%%"),
        }
    )

    # Visualization
    st.markdown("### Bloc Dynamics Comparison")

    fig = go.Figure()

    blocs = ['🇬🇧 UK', '🇪🇺 Europe', '🇺🇸 USA', '🇨🇳 China']
    growing = [summary_df.iloc[3]['Growing ↑'], summary_df.iloc[0]['Growing ↑'],
               summary_df.iloc[1]['Growing ↑'], summary_df.iloc[2]['Growing ↑']]
    declining = [summary_df.iloc[3]['Declining ↓'], summary_df.iloc[0]['Declining ↓'],
                 summary_df.iloc[1]['Declining ↓'], summary_df.iloc[2]['Declining ↓']]

    fig.add_trace(go.Bar(
        name='Growing ↑',
        y=blocs,
        x=growing,
        orientation='h',
        marker_color='#4CAF50',
    ))

    fig.add_trace(go.Bar(
        name='Declining ↓',
        y=blocs,
        x=[-d for d in declining],  # Negative for left side
        orientation='h',
        marker_color='#F44336',
    ))

    fig.update_layout(
        barmode='relative',
        xaxis_title="Topics (← Declining | Growing →)",
        yaxis_title="",
        height=300,
        margin=dict(l=100, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # UK vs Bloc comparison
    st.markdown("---")
    st.markdown("### UK Position Relative to Blocs")
    st.caption("In how many topics is UK outperforming or underperforming each bloc?")

    # Calculate UK vs each bloc
    comparisons = []
    for bloc_name, bloc_label in [('europe', '🇪🇺 Europe'), ('usa', '🇺🇸 USA'), ('china', '🇨🇳 China')]:
        uk_col = 'uk_trend'
        bloc_col = f'{bloc_name}_trend'

        # UK growing while bloc not growing
        uk_ahead = len(bloc_trends.filter(
            (pl.col(uk_col) == 'GROWING') & (pl.col(bloc_col) != 'GROWING')
        ))

        # Bloc growing while UK not growing
        bloc_ahead = len(bloc_trends.filter(
            (pl.col(bloc_col) == 'GROWING') & (pl.col(uk_col) != 'GROWING')
        ))

        # Both growing
        both_growing = len(bloc_trends.filter(
            (pl.col(uk_col) == 'GROWING') & (pl.col(bloc_col) == 'GROWING')
        ))

        # UK declining while bloc not declining
        uk_losing = len(bloc_trends.filter(
            (pl.col(uk_col) == 'DECLINING') & (pl.col(bloc_col) != 'DECLINING')
        ))

        comparisons.append({
            'vs': bloc_label,
            'UK ahead': uk_ahead,
            'Bloc ahead': bloc_ahead,
            'Both growing': both_growing,
            'UK losing ground': uk_losing,
        })

    comp_df = pl.DataFrame(comparisons).to_pandas()

    st.dataframe(
        comp_df,
        hide_index=True,
        column_config={
            "vs": st.column_config.TextColumn("Comparison", width=120),
            "UK ahead": st.column_config.NumberColumn(format="%d", help="UK growing, bloc not"),
            "Bloc ahead": st.column_config.NumberColumn(format="%d", help="Bloc growing, UK not"),
            "Both growing": st.column_config.NumberColumn(format="%d"),
            "UK losing ground": st.column_config.NumberColumn(format="%d", help="UK declining, bloc not"),
        }
    )

    # Interpretation breakdown
    st.markdown("---")
    st.markdown("### Strategic Context Distribution")
    st.caption("Classification of UK's position based on bloc dynamics")

    interp_counts = bloc_trends.group_by("interpretation").len().sort("len", descending=True)

    interp_labels = {
        'structural_shift': '🌍 Structural Shift (all blocs declining)',
        'european_pattern': '🇪🇺 European Pattern (following EU trend)',
        'china_consolidating': '🇨🇳 China Consolidating (China growing, others not)',
        'uk_specific_decline': '🚨 UK-Specific (UK underperforming peers)',
        'shared_growth': '📈 Shared Growth (UK growing with others)',
        'competing_with_china': '🏁 Competing with China (UK & China both growing)',
        'counter_trend_growth': '💪 Counter-Trend (UK growing against tide)',
        'mixed': '❓ Mixed Pattern',
    }

    interp_df = interp_counts.to_pandas()
    interp_df['Context'] = interp_df['interpretation'].map(lambda x: interp_labels.get(x, x))
    interp_df = interp_df.rename(columns={'len': 'Topics'})

    col1, col2 = st.columns([2, 1])

    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=interp_df['Context'][::-1],
            x=interp_df['Topics'][::-1],
            orientation='h',
            marker_color='#1976D2',
            text=interp_df['Topics'][::-1],
            textposition='outside'
        ))
        fig3.update_layout(
            xaxis_title="Number of Topics",
            yaxis_title="",
            height=350,
            margin=dict(l=280, r=50, t=30, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.dataframe(
            interp_df[['Context', 'Topics']],
            hide_index=True,
            height=350
        )

    # Field breakdown by bloc trend
    st.markdown("---")
    st.markdown("### Fields Where China is Growing")
    st.caption("Topics where China's share is increasing (potential competitive pressure)")

    china_growing = bloc_trends.filter(pl.col("china_trend") == "GROWING")
    china_by_field = china_growing.group_by("field_name").len().sort("len", descending=True)

    if len(china_by_field) > 0:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            y=china_by_field["field_name"].to_list()[:10][::-1],
            x=china_by_field["len"].to_list()[:10][::-1],
            orientation='h',
            marker_color='#E53935',
            text=china_by_field["len"].to_list()[:10][::-1],
            textposition='outside'
        ))
        fig4.update_layout(
            xaxis_title="Number of Topics",
            yaxis_title="",
            height=400,
            margin=dict(l=200, r=50, t=30, b=50)
        )
        st.plotly_chart(fig4, use_container_width=True)


# =============================================================================
# VIEW 4: UK Strengths Dashboard
# =============================================================================

def uk_strengths_dashboard(data):
    """UK strengths analysis with bloc context."""
    st.header("UK Strengths Dashboard")
    st.markdown("Where is UK research gaining momentum, and in what competitive context?")

    trajectories = data['trajectories']
    bloc_trends = data['bloc_trends']

    # Join trajectory patterns with bloc trends
    combined = trajectories.join(
        bloc_trends.select([
            "topic_id", "europe_trend", "usa_trend", "china_trend", "uk_trend", "interpretation"
        ]),
        on="topic_id",
        how="left"
    )

    # Summary metrics
    st.markdown("### UK Momentum Overview")
    st.caption("Based on trajectory patterns comparing recent (2016-24) vs early (2010-17) rolling windows")

    # Count trajectory patterns
    pattern_counts = trajectories.group_by("trajectory_pattern").len()
    pattern_dict = {row['trajectory_pattern']: row['len'] for row in pattern_counts.iter_rows(named=True)}

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "🚀 Accelerating",
            f"{pattern_dict.get('ACCELERATING', 0):,}",
            help="UK share growing faster recently than historically"
        )

    with m2:
        st.metric(
            "📈 Recovering",
            f"{pattern_dict.get('RECOVERING', 0):,}",
            help="UK share was declining, now growing"
        )

    with m3:
        st.metric(
            "📉 Decelerating",
            f"{pattern_dict.get('DECELERATING', 0):,}",
            help="UK share growth is slowing"
        )

    with m4:
        st.metric(
            "🚨 Rapid Retreat",
            f"{pattern_dict.get('RAPID_RETREAT', 0):,}",
            help="UK share actively declining"
        )

    # Accelerating topics by bloc context
    st.markdown("---")
    st.markdown("### UK Accelerating Topics: Competitive Context")
    st.caption("Where UK is accelerating, what are the three blocs doing?")

    uk_accelerating = combined.filter(pl.col("trajectory_pattern") == "ACCELERATING")

    # Cross-tabulate with China trend
    accel_by_china = uk_accelerating.group_by("china_trend").len().sort("len", descending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By China trend:**")
        for row in accel_by_china.iter_rows(named=True):
            trend = row['china_trend'] or 'Unknown'
            icon = {'GROWING': '🇨🇳↑', 'STABLE': '🇨🇳→', 'DECLINING': '🇨🇳↓'}.get(trend, '?')
            st.write(f"{icon} China {trend.lower()}: **{row['len']:,}** topics")

    with col2:
        # Cross-tabulate with Europe trend
        accel_by_europe = uk_accelerating.group_by("europe_trend").len().sort("len", descending=True)
        st.markdown("**By Europe trend:**")
        for row in accel_by_europe.iter_rows(named=True):
            trend = row['europe_trend'] or 'Unknown'
            icon = {'GROWING': '🇪🇺↑', 'STABLE': '🇪🇺→', 'DECLINING': '🇪🇺↓'}.get(trend, '?')
            st.write(f"{icon} Europe {trend.lower()}: **{row['len']:,}** topics")

    # Key strategic categories
    st.markdown("---")
    st.markdown("### Strategic Opportunities")

    # UK accelerating while China also growing (competitive race)
    competing_china = uk_accelerating.filter(pl.col("china_trend") == "GROWING")
    st.markdown(f"#### 🏁 Competing with China ({len(competing_china)} topics)")
    st.markdown("UK accelerating in topics where China is also growing — active competitive spaces.")

    if len(competing_china) > 0:
        competing_df = competing_china.select([
            "topic_name", "field_name", "current_uk_papers"
        ]).sort("current_uk_papers", descending=True).head(15).to_pandas()
        competing_df.columns = ["Topic", "Field", "UK Papers"]

        st.dataframe(
            competing_df,
            hide_index=True,
            column_config={
                "Topic": st.column_config.TextColumn(width=300),
                "UK Papers": st.column_config.NumberColumn(format="%d"),
            }
        )

    # UK accelerating while China declining (opportunity)
    china_retreat = uk_accelerating.filter(pl.col("china_trend") == "DECLINING")
    st.markdown(f"#### 💪 China Retreating, UK Advancing ({len(china_retreat)} topics)")
    st.markdown("UK accelerating while China is declining — potential to capture share.")

    if len(china_retreat) > 0:
        retreat_df = china_retreat.select([
            "topic_name", "field_name", "current_uk_papers"
        ]).sort("current_uk_papers", descending=True).head(15).to_pandas()
        retreat_df.columns = ["Topic", "Field", "UK Papers"]

        st.dataframe(
            retreat_df,
            hide_index=True,
            column_config={
                "Topic": st.column_config.TextColumn(width=300),
                "UK Papers": st.column_config.NumberColumn(format="%d"),
            }
        )

    # UK accelerating by field
    st.markdown("---")
    st.markdown("### UK Accelerating Topics by Field")

    accel_by_field = uk_accelerating.group_by("field_name").agg([
        pl.len().alias("topics"),
        pl.col("current_uk_papers").sum().alias("uk_papers")
    ]).sort("topics", descending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=accel_by_field["field_name"].to_list()[:12][::-1],
        x=accel_by_field["topics"].to_list()[:12][::-1],
        orientation='h',
        marker_color=COLORS['uk_advantage'],
        text=accel_by_field["topics"].to_list()[:12][::-1],
        textposition='outside'
    ))

    fig.update_layout(
        xaxis_title="Number of Topics where UK is Accelerating",
        yaxis_title="",
        height=450,
        margin=dict(l=200, r=50, t=30, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top accelerating topics
    st.markdown("---")
    st.markdown("### Top UK Accelerating Topics")
    st.caption("Topics where UK momentum is strongest (by recent slope)")

    top_accel = combined.filter(
        pl.col("trajectory_pattern") == "ACCELERATING"
    ).sort("recent_uk_share_slope", descending=True).head(25)

    # Add bloc indicators
    top_df = top_accel.select([
        "topic_name",
        "field_name",
        "current_uk_papers",
        "recent_uk_share_slope",
        "europe_trend",
        "usa_trend",
        "china_trend",
    ]).to_pandas()

    # Format bloc trends
    def format_trend(t):
        return {'GROWING': '↑', 'STABLE': '→', 'DECLINING': '↓'}.get(t, '?') if t else '?'

    top_df["🇪🇺"] = top_df["europe_trend"].apply(format_trend)
    top_df["🇺🇸"] = top_df["usa_trend"].apply(format_trend)
    top_df["🇨🇳"] = top_df["china_trend"].apply(format_trend)
    top_df["Slope"] = top_df["recent_uk_share_slope"] * 100  # Convert to pp

    top_df = top_df[["topic_name", "field_name", "current_uk_papers", "Slope", "🇪🇺", "🇺🇸", "🇨🇳"]]
    top_df.columns = ["Topic", "Field", "UK Papers", "UK Trend (pp)", "🇪🇺", "🇺🇸", "🇨🇳"]

    st.dataframe(
        top_df,
        hide_index=True,
        column_config={
            "Topic": st.column_config.TextColumn(width=280),
            "Field": st.column_config.TextColumn(width=150),
            "UK Papers": st.column_config.NumberColumn(format="%d", width=80),
            "UK Trend (pp)": st.column_config.NumberColumn(format="%+.3f", width=100),
            "🇪🇺": st.column_config.TextColumn(width=40),
            "🇺🇸": st.column_config.TextColumn(width=40),
            "🇨🇳": st.column_config.TextColumn(width=40),
        }
    )

    # Explore by field
    st.markdown("---")
    st.markdown("### Explore Accelerating Topics by Field")

    fields = sorted([f for f in uk_accelerating["field_name"].unique().to_list() if f is not None])
    selected_field = st.selectbox("Select field", fields)

    field_topics = uk_accelerating.filter(pl.col("field_name") == selected_field).sort("recent_uk_share_slope", descending=True)

    field_df = field_topics.select([
        "topic_name",
        "current_uk_papers",
        "recent_uk_share_slope",
        "europe_trend",
        "usa_trend",
        "china_trend",
    ]).to_pandas()

    field_df["🇪🇺"] = field_df["europe_trend"].apply(format_trend)
    field_df["🇺🇸"] = field_df["usa_trend"].apply(format_trend)
    field_df["🇨🇳"] = field_df["china_trend"].apply(format_trend)
    field_df["Slope"] = field_df["recent_uk_share_slope"] * 100

    field_df = field_df[["topic_name", "current_uk_papers", "Slope", "🇪🇺", "🇺🇸", "🇨🇳"]]
    field_df.columns = ["Topic", "UK Papers", "UK Trend (pp)", "🇪🇺", "🇺🇸", "🇨🇳"]

    st.dataframe(
        field_df,
        hide_index=True,
        column_config={
            "Topic": st.column_config.TextColumn(width=300),
            "UK Papers": st.column_config.NumberColumn(format="%d", width=80),
            "UK Trend (pp)": st.column_config.NumberColumn(format="%+.3f", width=100),
            "🇪🇺": st.column_config.TextColumn(width=40),
            "🇺🇸": st.column_config.TextColumn(width=40),
            "🇨🇳": st.column_config.TextColumn(width=40),
        }
    )


# =============================================================================
# VIEW 5: Country Analysis (individual country comparison)
# =============================================================================

def country_analysis(data):
    """Individual country-by-country analysis using 46 comparator countries."""
    st.header("Country Analysis")
    st.markdown("Compare research trajectory patterns across 46 comparator countries.")

    country_traj = data['country_trajectories']
    country_profiles = data['country_profiles']

    # Calculate country summaries from trajectory data
    country_summary = country_traj.group_by("country_code").agg([
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "ACCELERATING").len().alias("accelerating"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "RECOVERING").len().alias("recovering"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "STEADY").len().alias("steady"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "DECELERATING").len().alias("decelerating"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "RAPID_RETREAT").len().alias("rapid_retreat"),
        pl.col("current_papers").sum().alias("total_papers"),
        pl.col("recent_slope").mean().alias("avg_recent_slope"),
    ])

    # Join with country names
    country_summary = country_summary.join(
        country_profiles.select(["country_code", "country_name"]),
        on="country_code",
        how="left"
    )

    # Calculate momentum score (growing - declining patterns) - cast to signed int
    country_summary = country_summary.with_columns([
        (pl.col("accelerating").cast(pl.Int64) + pl.col("recovering").cast(pl.Int64)
         - pl.col("decelerating").cast(pl.Int64) - pl.col("rapid_retreat").cast(pl.Int64)).alias("momentum_score"),
        (pl.col("accelerating") + pl.col("recovering")).alias("growing_topics"),
        (pl.col("decelerating") + pl.col("rapid_retreat")).alias("declining_topics"),
        (pl.col("accelerating") + pl.col("recovering") + pl.col("steady")
         + pl.col("decelerating") + pl.col("rapid_retreat")).alias("total_topics"),
    ])

    # Overview table
    st.markdown("### Country Overview")
    st.caption("Number of topics (out of ~4,500) where each country shows each trajectory pattern. "
               "Patterns compare rolling 5-year windows: recent (2016-20 to 2020-24) vs early (2010-14 to 2015-19).")

    # Sort by momentum score
    summary_sorted = country_summary.sort("momentum_score", descending=True)

    display_df = summary_sorted.select([
        "country_name",
        "total_topics",
        "accelerating",
        "recovering",
        "steady",
        "decelerating",
        "rapid_retreat",
        "momentum_score",
    ]).to_pandas()

    display_df.columns = ["Country", "Topics", "Accelerating", "Recovering", "Steady", "Decelerating", "Rapid Retreat", "Net Score"]

    # Highlight UK row
    def highlight_uk(row):
        if row['Country'] == 'United Kingdom':
            return ['background-color: #E3F2FD'] * len(row)
        return [''] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_uk, axis=1),
        height=500,
        hide_index=True,
        column_config={
            "Country": st.column_config.TextColumn(width=150),
            "Topics": st.column_config.NumberColumn(format="%d", width=70, help="Total topics this country is active in"),
            "Accelerating": st.column_config.NumberColumn(format="%d", width=95, help="Topics where country share is growing faster recently"),
            "Recovering": st.column_config.NumberColumn(format="%d", width=85, help="Topics where decline reversed to growth"),
            "Steady": st.column_config.NumberColumn(format="%d", width=70, help="Topics with stable trajectory"),
            "Decelerating": st.column_config.NumberColumn(format="%d", width=95, help="Topics where growth is slowing"),
            "Rapid Retreat": st.column_config.NumberColumn(format="%d", width=100, help="Topics where share is actively declining"),
            "Net Score": st.column_config.NumberColumn(format="%+d", width=80, help="(Accelerating + Recovering) - (Decelerating + Rapid Retreat)"),
        }
    )

    # Momentum ranking chart
    st.markdown("---")
    st.markdown("### Momentum Ranking")
    st.markdown("""
    **Net score** = (topics accelerating + recovering) − (topics decelerating + in rapid retreat)

    Positive = more topics gaining momentum than losing it. Negative = more topics losing momentum.
    """)

    top_25 = summary_sorted.head(25)

    fig = go.Figure()

    colors = ['#1976D2' if name == 'United Kingdom' else ('#4CAF50' if score > 0 else '#F44336')
              for name, score in zip(top_25['country_name'].to_list(), top_25['momentum_score'].to_list())]

    fig.add_trace(go.Bar(
        y=top_25['country_name'].to_list()[::-1],
        x=top_25['momentum_score'].to_list()[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{s:+,d}" for s in top_25['momentum_score'].to_list()[::-1]],
        textposition='outside'
    ))

    fig.add_vline(x=0, line_color="black", line_width=1)

    fig.update_layout(
        xaxis_title="Net Score (topics gaining − topics losing momentum)",
        yaxis_title="",
        height=600,
        margin=dict(l=150, r=80, t=30, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Country selector for detailed comparison
    st.markdown("---")
    st.markdown("### Detailed Country Comparison")

    available_countries = summary_sorted['country_name'].to_list()

    selected_countries = st.multiselect(
        "Select countries to compare",
        available_countries,
        default=["United Kingdom", "Germany", "France", "United States", "China", "Netherlands"]
    )

    if selected_countries:
        # Get selected country data
        selected_codes = country_profiles.filter(
            pl.col("country_name").is_in(selected_countries)
        )['country_code'].to_list()

        selected_data = country_summary.filter(pl.col("country_code").is_in(selected_codes))

        # Calculate percentages for fair comparison
        selected_data = selected_data.with_columns([
            (pl.col("accelerating") * 100.0 / pl.col("total_topics")).alias("pct_accelerating"),
            (pl.col("recovering") * 100.0 / pl.col("total_topics")).alias("pct_recovering"),
            (pl.col("steady") * 100.0 / pl.col("total_topics")).alias("pct_steady"),
            (pl.col("decelerating") * 100.0 / pl.col("total_topics")).alias("pct_decelerating"),
            (pl.col("rapid_retreat") * 100.0 / pl.col("total_topics")).alias("pct_rapid_retreat"),
        ])

        # Stacked bar chart showing pattern distribution as percentages
        st.caption("Percentage of topics in each trajectory pattern (countries active in different numbers of topics)")

        fig2 = go.Figure()

        countries = selected_data.sort("momentum_score", descending=True)['country_name'].to_list()
        patterns = [
            ('pct_accelerating', 'Accelerating', '#2E7D32'),
            ('pct_recovering', 'Recovering', '#4CAF50'),
            ('pct_steady', 'Steady', '#9E9E9E'),
            ('pct_decelerating', 'Decelerating', '#FF9800'),
            ('pct_rapid_retreat', 'Rapid Retreat', '#C62828'),
        ]

        for pattern_col, pattern_name, color in patterns:
            values = selected_data.sort("momentum_score", descending=True)[pattern_col].to_list()
            fig2.add_trace(go.Bar(
                name=pattern_name,
                y=countries,
                x=values,
                orientation='h',
                marker_color=color,
                text=[f"{v:.0f}%" for v in values],
                textposition='inside',
                textfont_size=10,
            ))

        fig2.update_layout(
            barmode='stack',
            xaxis_title="% of Topics",
            xaxis=dict(range=[0, 100]),
            yaxis_title="",
            height=50 + 50 * len(countries),
            margin=dict(l=150, r=50, t=30, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Simple summary table
        st.markdown("#### Summary")

        detail_df = selected_data.sort("momentum_score", descending=True).select([
            "country_name",
            "total_topics",
            "growing_topics",
            "declining_topics",
            "momentum_score",
        ]).to_pandas()

        detail_df.columns = ["Country", "Active Topics", "Growing", "Declining", "Net Score"]

        st.dataframe(
            detail_df,
            hide_index=True,
            column_config={
                "Active Topics": st.column_config.NumberColumn(format="%d", help="Topics where this country has presence"),
                "Growing": st.column_config.NumberColumn(format="%d", help="Accelerating + Recovering"),
                "Declining": st.column_config.NumberColumn(format="%d", help="Decelerating + Rapid Retreat"),
                "Net Score": st.column_config.NumberColumn(format="%+d"),
            }
        )

    # Rapid retreat comparison
    st.markdown("---")
    st.markdown("### Rapid Retreat Comparison")
    st.caption("Which countries have the most topics in rapid retreat?")

    retreat_ranked = country_summary.sort("rapid_retreat", descending=True).head(15)

    fig3 = go.Figure()

    colors = ['#1976D2' if code == 'GB' else '#C62828'
              for code in retreat_ranked['country_code'].to_list()]

    fig3.add_trace(go.Bar(
        y=retreat_ranked['country_name'].to_list()[::-1],
        x=retreat_ranked['rapid_retreat'].to_list()[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=retreat_ranked['rapid_retreat'].to_list()[::-1],
        textposition='outside'
    ))

    fig3.update_layout(
        xaxis_title="Topics in Rapid Retreat",
        yaxis_title="",
        height=450,
        margin=dict(l=150, r=50, t=30, b=50)
    )

    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="UK Research Position Explorer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Handle navigation requests (from buttons that want to switch views)
    views = ["Topic Browser", "Topic Detail", "Bloc Comparison", "Country Analysis", "UK Strengths Dashboard"]

    default_idx = 0
    if 'navigate_to' in st.session_state:
        try:
            default_idx = views.index(st.session_state['navigate_to'])
        except ValueError:
            default_idx = 0
        del st.session_state['navigate_to']

    # Sidebar navigation
    st.sidebar.title("🔬 UK Research Explorer")
    st.sidebar.markdown("Explore UK research positioning across 4,516 topics.")

    # View selector
    view = st.sidebar.radio(
        "Select View",
        views,
        index=default_idx,
        key="view_selector"
    )

    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    Analysis of UK research positioning across 4,516 topics using OpenAlex data (2010-2024).

    Compares UK trends against three major science blocs: **Europe** (EU27+EFTA+Israel), **USA**, and **China**.
    """)

    st.sidebar.markdown("### Key Findings")
    st.sidebar.markdown("""
    **UK Momentum:**
    - 1,337 topics accelerating
    - 1,136 topics in rapid retreat
    - 1,438 topics steady

    **Context matters:**
    - 73% of topics show UK-specific patterns
    - 9% reflect structural shifts (multiple blocs declining)
    - 5% show China consolidating share

    **Method:** Rolling 5-year windows comparing recent period (2016-24) vs early period (2010-17).
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("*UK Research Position Study*")

    # Load data
    with st.spinner("Loading data..."):
        data = load_data()

    # Route to view
    if view == "Topic Browser":
        topic_browser(data)
    elif view == "Topic Detail":
        topic_detail(data)
    elif view == "Bloc Comparison":
        country_comparison(data)
    elif view == "Country Analysis":
        country_analysis(data)
    elif view == "UK Strengths Dashboard":
        uk_strengths_dashboard(data)


if __name__ == "__main__":
    main()
