#!/usr/bin/env python3
"""
UK Research Position Interactive Explorer

A Streamlit app for exploring UK research positioning across 4,516 topics.

Usage:
    Deployed version for Streamlit Community Cloud.

Views:
    1. Topic Browser - Filterable table with search
    2. Topic Detail - Deep dive with trajectory chart + country comparison
    3. Country Comparison - Side-by-side analysis
    4. UK Strengths Dashboard - Advantages summary
"""

import json
import numpy as np
import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "data"
IND_STRAT_DIR = Path(__file__).parent / "data" / "ind_strat"

# Set Plotly default font to Roboto Mono
PLOTLY_FONT = dict(family="Roboto Mono, monospace", size=12, weight=200)
PLOTLY_TITLE_FONT = dict(family="Roboto Mono, monospace", size=16, weight=400)

pio.templates["roboto_mono"] = go.layout.Template(
    layout=go.Layout(
        font=PLOTLY_FONT,
        title_font=PLOTLY_TITLE_FONT,
        xaxis=dict(tickfont=PLOTLY_FONT, title_font=PLOTLY_FONT),
        yaxis=dict(tickfont=PLOTLY_FONT, title_font=PLOTLY_FONT),
        legend=dict(font=PLOTLY_FONT),
    )
)
pio.templates.default = "plotly+roboto_mono"

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
    'CONSOLIDATING': '📊',
    'RECOVERING': '📈',
    'STEADY': '➡️',
    'STABILISING': '⏸️',
    'DECELERATING': '📉',
    'DECLINING': '⬇️',
    'RAPID_RETREAT': '🚨',
}

# Bloc context interpretation labels (user-friendly names)
# These describe how UK's *recent* bloc trend compares to other blocs
# Note: this is separate from trajectory pattern (which compares recent vs historical)
INTERPRETATION_LABELS = {
    'uk_lagging_blocs': 'UK lagging bloc growth',
    'structural_shift': 'Structural shift (multiple blocs declining)',
    'european_pattern': 'Following European pattern',
    'china_consolidating': 'China consolidating',
    'shared_growth': 'Shared growth',
    'competing_with_china': 'Competing with China',
    'counter_trend_growth': 'Counter-trend (UK growing, blocs declining)',
    'mixed': 'Mixed pattern',
}

INTERPRETATION_DESCRIPTIONS = {
    'uk_lagging_blocs': "UK's recent growth rate is below the threshold for 'GROWING' while Europe/USA are not declining. UK may still be growing - just not as fast as peers. This is about absolute recent growth rate, separate from trajectory (which compares to UK's own history).",
    'structural_shift': "Multiple major blocs are declining together - suggests field-level contraction, not UK-specific.",
    'european_pattern': "UK trend matches broader European pattern.",
    'china_consolidating': "China is growing its share while other blocs are flat or declining.",
    'shared_growth': "UK is growing along with other major blocs.",
    'competing_with_china': "Both UK and China are growing - active competitive space.",
    'counter_trend_growth': "UK is growing while multiple other blocs are declining.",
    'mixed': "Complex pattern not fitting other categories.",
}

# Industrial Strategy sector colors
SECTOR_COLORS = {
    'Life Sciences': '#4CAF50',
    'Cross-cutting': '#2196F3',
    'Financial Services': '#9C27B0',
    'Defence': '#607D8B',
    'Professional and Business Services': '#FF9800',
    'Creative Industries': '#E91E63',
    'Advanced Manufacturing': '#795548',
    'Digital and Technologies': '#00BCD4',
    'Clean Energy Industries': '#8BC34A',
}

# Alignment status colors and labels
ALIGNMENT_COLORS = {
    'strong_alignment': '#2E7D32',
    'moderate_alignment': '#81C784',
    'uk_strength_declining_field': '#FFCA28',
    'growing_field_uk_weakness': '#FF9800',
    'strategic_concern': '#C62828',
}

ALIGNMENT_LABELS = {
    'strong_alignment': 'Strong Alignment',
    'moderate_alignment': 'Moderate Alignment',
    'uk_strength_declining_field': 'UK Strength (Declining Field)',
    'growing_field_uk_weakness': 'Growing Field (UK Weakness)',
    'strategic_concern': 'Strategic Concern',
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

    # Industrial Strategy mapping data
    data['ind_strat_sectors'] = pl.read_parquet(IND_STRAT_DIR / "analysis" / "sector_summary.parquet")
    data['ind_strat_priorities'] = pl.read_parquet(IND_STRAT_DIR / "analysis" / "priority_alignment.parquet")
    data['ind_strat_mappings'] = pl.read_parquet(IND_STRAT_DIR / "analysis" / "core_mappings_detail.parquet")

    # Load priority descriptions from JSON
    with open(IND_STRAT_DIR / "extracted" / "priorities.json") as f:
        priorities_json = json.load(f)
    data['ind_strat_priority_details'] = {
        p['priority_id']: p for p in priorities_json['priorities']
    }

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
    ind_strat_mappings = data['ind_strat_mappings']
    priority_details = data['ind_strat_priority_details']

    # Join bloc trends to trajectories
    df = df.join(bloc_trends.select([
        "topic_id", "europe_trend", "usa_trend", "china_trend", "global_trend", "interpretation"
    ]), on="topic_id", how="left")

    # Create topic-to-sector mapping (a topic can map to multiple sectors, take first)
    topic_sectors = ind_strat_mappings.select(['topic_id', 'priority_id', 'strategic_concern']).join(
        pl.DataFrame([
            {'priority_id': pid, 'parent_sector': pdata.get('parent_sector', 'Unknown')}
            for pid, pdata in priority_details.items()
        ]),
        on='priority_id',
        how='left'
    ).group_by('topic_id').agg([
        pl.col('parent_sector').first().alias('ind_strat_sector'),
        pl.col('strategic_concern').max().alias('is_strategic_concern')  # True if any mapping is concern
    ])

    # Join sector info to main df
    df = df.join(topic_sectors, on='topic_id', how='left')

    # Fill nulls for unmapped topics
    df = df.with_columns([
        pl.col('ind_strat_sector').fill_null('Not Mapped'),
        pl.col('is_strategic_concern').fill_null(False)
    ])

    # Filters - now 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fields = ["All Fields"] + sorted([f for f in df["field_name"].unique().to_list() if f is not None])
        selected_field = st.selectbox("Field", fields)

    with col2:
        # Interpretation filter (bloc-based context) with friendly labels
        raw_interps = sorted([i for i in df["interpretation"].unique().to_list() if i is not None])
        interp_options = {INTERPRETATION_LABELS.get(i, i): i for i in raw_interps}
        interp_display = ["All Contexts"] + list(interp_options.keys())
        selected_interp_display = st.selectbox(
            "Bloc Context", interp_display,
            help="How UK's recent share trend compares to Europe, USA, and China. This is about absolute recent growth rate, separate from trajectory pattern (which compares recent vs historical UK performance)."
        )
        selected_interp = interp_options.get(selected_interp_display, None)

    with col3:
        # Define pattern order and display labels
        pattern_order = ['ACCELERATING', 'CONSOLIDATING', 'RECOVERING', 'STEADY',
                        'STABILISING', 'DECELERATING', 'DECLINING', 'RAPID_RETREAT']
        raw_patterns = [p for p in df["trajectory_pattern"].unique().to_list() if p is not None]
        # Filter to patterns that exist in data, maintaining order
        ordered_patterns = [p for p in pattern_order if p in raw_patterns]
        # Map raw to display labels
        pattern_display = {p: p.replace('_', ' ').title() for p in ordered_patterns}
        pattern_options = ["All Patterns"] + [pattern_display[p] for p in ordered_patterns]
        selected_pattern_display = st.selectbox("UK Momentum", pattern_options)
        # Map back to raw value for filtering
        display_to_raw = {v: k for k, v in pattern_display.items()}
        selected_pattern = display_to_raw.get(selected_pattern_display, None)

    with col4:
        # Industrial Strategy sector filter
        sector_order = ['Life Sciences', 'Cross-cutting', 'Financial Services', 'Defence',
                       'Professional and Business Services', 'Creative Industries',
                       'Advanced Manufacturing', 'Digital and Technologies', 'Clean Energy Industries']
        raw_sectors = [s for s in df["ind_strat_sector"].unique().to_list() if s is not None]
        ordered_sectors = [s for s in sector_order if s in raw_sectors]
        # Add 'Not Mapped' at the end if present
        if 'Not Mapped' in raw_sectors:
            ordered_sectors.append('Not Mapped')
        sector_options = ["All Sectors"] + ordered_sectors + ["Strategic Concerns Only"]
        selected_sector = st.selectbox(
            "Ind. Strategy Sector", sector_options,
            help="Filter by Industrial Strategy sector mapping. 'Strategic Concerns Only' shows topics where UK is losing ground in declining fields."
        )

    with col5:
        search = st.text_input("Search topics", "")

    # Apply filters
    filtered = df

    if selected_field != "All Fields":
        filtered = filtered.filter(pl.col("field_name") == selected_field)

    if selected_interp_display != "All Contexts" and selected_interp:
        filtered = filtered.filter(pl.col("interpretation") == selected_interp)

    if selected_pattern_display != "All Patterns" and selected_pattern:
        filtered = filtered.filter(pl.col("trajectory_pattern") == selected_pattern)

    if selected_sector == "Strategic Concerns Only":
        filtered = filtered.filter(pl.col("is_strategic_concern") == True)
    elif selected_sector != "All Sectors":
        filtered = filtered.filter(pl.col("ind_strat_sector") == selected_sector)

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
        "ind_strat_sector",
        "is_strategic_concern",
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
        return f"{PATTERN_ICONS.get(p, '?')} {p.replace('_', ' ').title()}"

    pandas_df["trajectory_pattern"] = pandas_df["trajectory_pattern"].apply(format_pattern)

    # Format sector with concern indicator
    def format_sector(row):
        sector = row['ind_strat_sector']
        is_concern = row['is_strategic_concern']
        if sector == 'Not Mapped':
            return "—"
        if is_concern:
            return f"⚠️ {sector}"
        return sector

    pandas_df["Sector"] = pandas_df.apply(format_sector, axis=1)

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

    # Drop original columns and rename
    pandas_df = pandas_df.drop(columns=["europe_trend", "usa_trend", "china_trend", "global_trend",
                                        "ind_strat_sector", "is_strategic_concern"])
    pandas_df = pandas_df.rename(columns={
        "topic_name": "Topic",
        "field_name": "Field",
        "current_uk_papers": "UK Papers",
        "current_uk_share": "UK Share %",
        "trajectory_pattern": "🇬🇧 Momentum",
    })

    # Reorder columns
    pandas_df = pandas_df[["Topic", "Field", "Sector", "UK Papers", "UK Share %", "🇬🇧 Momentum", "🇪🇺+", "🇺🇸", "🇨🇳", "🌍"]]

    # Clickable table
    event = st.dataframe(
        pandas_df,
        height=400,
        hide_index=True,
        column_config={
            "Topic": st.column_config.TextColumn(width=250),
            "Field": st.column_config.TextColumn(width=130),
            "Sector": st.column_config.TextColumn(width=160, help="Industrial Strategy sector (⚠️ = strategic concern)"),
            "UK Papers": st.column_config.NumberColumn(format="%d", width=70),
            "UK Share %": st.column_config.NumberColumn(format="%.2f%%", width=70),
            "🇬🇧 Momentum": st.column_config.TextColumn(width=130),
            "🇪🇺+": st.column_config.TextColumn(width=35),
            "🇺🇸": st.column_config.TextColumn(width=35),
            "🇨🇳": st.column_config.TextColumn(width=35),
            "🌍": st.column_config.TextColumn(width=35),
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
                st.session_state['_navigate_to'] = "Topic Detail"
                st.rerun()
    else:
        st.caption("Click a row above to select a topic, then navigate to details.")


# =============================================================================
# VIEW 2: Topic Detail
# =============================================================================

# Pattern interpretations for policy guidance
PATTERN_INTERPRETATIONS = {
    'ACCELERATING': {
        'icon': '🚀',
        'summary': 'Accelerating',
        'description': 'UK share is rising faster recently than historically.',
        'guidance': 'Success in progress — UK is gaining momentum.',
        'color': '#2E7D32',
    },
    'CONSOLIDATING': {
        'icon': '📊',
        'summary': 'Consolidating',
        'description': 'UK share has sustained positive growth across both periods.',
        'guidance': 'Established strength — continued investment is paying off.',
        'color': '#1976D2',
    },
    'RECOVERING': {
        'icon': '📈',
        'summary': 'Recovering',
        'description': 'UK share was declining but is now growing again.',
        'guidance': 'Monitor positively — turnaround in progress.',
        'color': '#4CAF50',
    },
    'STEADY': {
        'icon': '➡️',
        'summary': 'Steady',
        'description': 'UK share trajectory is flat over time.',
        'guidance': 'Evaluate strategically — stable but not growing.',
        'color': '#9E9E9E',
    },
    'STABILISING': {
        'icon': '⏸️',
        'summary': 'Stabilising',
        'description': 'UK share was declining but has now levelled off.',
        'guidance': 'Bleeding stopped — may need intervention to restore growth.',
        'color': '#78909C',
    },
    'DECELERATING': {
        'icon': '📉',
        'summary': 'Decelerating',
        'description': 'UK share growth is slowing — momentum is fading.',
        'guidance': 'Watch closely — early warning sign that may require attention.',
        'color': '#F57C00',
    },
    'DECLINING': {
        'icon': '⬇️',
        'summary': 'Declining',
        'description': 'UK share has sustained negative growth across both periods.',
        'guidance': 'Long-term erosion — may need strategic decision on whether to invest or exit.',
        'color': '#E65100',
    },
    'RAPID_RETREAT': {
        'icon': '🚨',
        'summary': 'Rapid Retreat',
        'description': 'UK share was stable or growing historically but is now actively declining.',
        'guidance': 'Investigate urgently — this was a UK strength that is now eroding.',
        'color': '#C62828',
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

    selected_topic = st.selectbox("Select Topic", topic_names, index=default_idx, key="topic_detail_selector")

    # Update session state when selection changes
    st.session_state['selected_topic'] = selected_topic

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

        bloc_html = []
        for bloc_name, bloc_label in [('europe', '🇪🇺 EU+'), ('usa', '🇺🇸 USA'), ('china', '🇨🇳 China'), ('global', '🌍 Global')]:
            trend = bloc_trends.get(bloc_name, {}).get('trend', 'STABLE')
            icon = trend_icons[trend]
            color = trend_colors[trend]
            bloc_html.append(f'<span style="color: {color}; font-weight: bold;">{bloc_label} {icon}</span>')

        st.markdown(f"""
        <div style="background-color: #f5f5f5; padding: 12px; border-radius: 5px; margin-top: 10px;">
            <div style="font-size: 0.8em; color: #666; margin-bottom: 8px;">Recent bloc trends:</div>
            <div style="display: flex; justify-content: space-around; font-size: 1.1em;">
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

    # Industrial Strategy Alignment section
    st.markdown("---")
    st.markdown("### Industrial Strategy Alignment")

    ind_strat_mappings = data['ind_strat_mappings']
    priority_details = data['ind_strat_priority_details']

    # Find all priorities this topic is mapped to
    topic_mappings = ind_strat_mappings.filter(pl.col("topic_id") == topic_id)

    if topic_mappings.height > 0:
        # Check if any mapping is strategic concern
        is_concern = topic_mappings.filter(pl.col("strategic_concern") == True).height > 0

        if is_concern:
            st.warning("**Strategic Concern**: This topic is flagged as a concern in at least one Industrial Strategy priority (declining momentum + UK losing ground).")

        # Display mapped priorities
        st.markdown(f"This topic maps to **{topic_mappings.height}** Industrial Strategy priorities:")

        for row in topic_mappings.iter_rows(named=True):
            priority_id = row['priority_id']
            priority_name = row['priority_name']
            relevance = row['relevance']
            pdetail = priority_details.get(priority_id, {})
            sector = pdetail.get('parent_sector', 'Unknown')
            color = SECTOR_COLORS.get(sector, '#757575')

            concern_badge = "⚠️ " if row.get('strategic_concern', False) else ""

            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.5rem; border-left: 3px solid {color}; background: #f9f9f9;">
                <strong>{concern_badge}{priority_name}</strong><br/>
                <span style="font-size: 0.85rem; color: #666;">
                    Sector: {sector} | Relevance: {relevance.title() if relevance else 'Core'}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("This topic is not currently mapped to any Industrial Strategy priorities.")


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

    # Use module-level labels with icons for display
    interp_labels = {
        'structural_shift': '🌍 ' + INTERPRETATION_LABELS['structural_shift'],
        'european_pattern': '🇪🇺 ' + INTERPRETATION_LABELS['european_pattern'],
        'china_consolidating': '🇨🇳 ' + INTERPRETATION_LABELS['china_consolidating'],
        'uk_lagging_blocs': '📊 ' + INTERPRETATION_LABELS['uk_lagging_blocs'],
        'shared_growth': '📈 ' + INTERPRETATION_LABELS['shared_growth'],
        'competing_with_china': '🏁 ' + INTERPRETATION_LABELS['competing_with_china'],
        'counter_trend_growth': '💪 ' + INTERPRETATION_LABELS['counter_trend_growth'],
        'mixed': '❓ ' + INTERPRETATION_LABELS['mixed'],
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
            "📊 Consolidating",
            f"{pattern_dict.get('CONSOLIDATING', 0):,}",
            help="Sustained positive growth across both periods"
        )

    with m3:
        st.metric(
            "📈 Recovering",
            f"{pattern_dict.get('RECOVERING', 0):,}",
            help="UK share was declining, now growing"
        )

    with m4:
        st.metric(
            "➡️ Steady",
            f"{pattern_dict.get('STEADY', 0):,}",
            help="UK share flat over time"
        )

    n1, n2, n3, n4 = st.columns(4)

    with n1:
        st.metric(
            "⏸️ Stabilising",
            f"{pattern_dict.get('STABILISING', 0):,}",
            help="UK share was declining, now levelled off"
        )

    with n2:
        st.metric(
            "📉 Decelerating",
            f"{pattern_dict.get('DECELERATING', 0):,}",
            help="UK share growth is slowing"
        )

    with n3:
        st.metric(
            "⬇️ Declining",
            f"{pattern_dict.get('DECLINING', 0):,}",
            help="Sustained negative growth across both periods"
        )

    with n4:
        st.metric(
            "🚨 Rapid Retreat",
            f"{pattern_dict.get('RAPID_RETREAT', 0):,}",
            help="UK share actively declining from previous strength"
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
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "CONSOLIDATING").len().alias("consolidating"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "RECOVERING").len().alias("recovering"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "STEADY").len().alias("steady"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "STABILISING").len().alias("stabilising"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "DECELERATING").len().alias("decelerating"),
        pl.col("trajectory_pattern").filter(pl.col("trajectory_pattern") == "DECLINING").len().alias("declining"),
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

    # Calculate momentum score (positive - negative patterns) - cast to signed int
    country_summary = country_summary.with_columns([
        (pl.col("accelerating").cast(pl.Int64) + pl.col("consolidating").cast(pl.Int64) + pl.col("recovering").cast(pl.Int64)
         - pl.col("decelerating").cast(pl.Int64) - pl.col("declining").cast(pl.Int64) - pl.col("rapid_retreat").cast(pl.Int64)).alias("momentum_score"),
        (pl.col("accelerating") + pl.col("consolidating") + pl.col("recovering")).alias("growing_topics"),
        (pl.col("decelerating") + pl.col("declining") + pl.col("rapid_retreat")).alias("declining_topics"),
        (pl.col("accelerating") + pl.col("consolidating") + pl.col("recovering") + pl.col("steady")
         + pl.col("stabilising") + pl.col("decelerating") + pl.col("declining") + pl.col("rapid_retreat")).alias("total_topics"),
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
        "consolidating",
        "recovering",
        "steady",
        "stabilising",
        "decelerating",
        "declining",
        "rapid_retreat",
        "momentum_score",
    ]).to_pandas()

    display_df.columns = ["Country", "Topics", "🚀", "📊", "📈", "➡️", "⏸️", "📉", "⬇️", "🚨", "Net"]

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
            "Country": st.column_config.TextColumn(width=140),
            "Topics": st.column_config.NumberColumn(format="%d", width=60, help="Total topics this country is active in"),
            "🚀": st.column_config.NumberColumn(format="%d", width=50, help="Accelerating: growing faster recently than historically"),
            "📊": st.column_config.NumberColumn(format="%d", width=50, help="Consolidating: sustained positive growth"),
            "📈": st.column_config.NumberColumn(format="%d", width=50, help="Recovering: was declining, now growing"),
            "➡️": st.column_config.NumberColumn(format="%d", width=50, help="Steady: flat trajectory"),
            "⏸️": st.column_config.NumberColumn(format="%d", width=50, help="Stabilising: was declining, now levelled off"),
            "📉": st.column_config.NumberColumn(format="%d", width=50, help="Decelerating: growth is slowing"),
            "⬇️": st.column_config.NumberColumn(format="%d", width=50, help="Declining: sustained negative growth"),
            "🚨": st.column_config.NumberColumn(format="%d", width=50, help="Rapid Retreat: actively declining from strength"),
            "Net": st.column_config.NumberColumn(format="%+d", width=60, help="(Accelerating + Consolidating + Recovering) − (Decelerating + Declining + Rapid Retreat)"),
        }
    )

    # Momentum ranking chart
    st.markdown("---")
    st.markdown("### Momentum Ranking")
    st.markdown("""
    **Net score** = (accelerating + consolidating + recovering) − (decelerating + declining + rapid retreat)

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

        # Calculate percentages for fair comparison (all 8 trajectory patterns)
        selected_data = selected_data.with_columns([
            (pl.col("accelerating") * 100.0 / pl.col("total_topics")).alias("pct_accelerating"),
            (pl.col("consolidating") * 100.0 / pl.col("total_topics")).alias("pct_consolidating"),
            (pl.col("recovering") * 100.0 / pl.col("total_topics")).alias("pct_recovering"),
            (pl.col("steady") * 100.0 / pl.col("total_topics")).alias("pct_steady"),
            (pl.col("stabilising") * 100.0 / pl.col("total_topics")).alias("pct_stabilising"),
            (pl.col("decelerating") * 100.0 / pl.col("total_topics")).alias("pct_decelerating"),
            (pl.col("declining") * 100.0 / pl.col("total_topics")).alias("pct_declining"),
            (pl.col("rapid_retreat") * 100.0 / pl.col("total_topics")).alias("pct_rapid_retreat"),
        ])

        # Stacked bar chart showing pattern distribution as percentages
        st.caption("Percentage of topics in each trajectory pattern (countries active in different numbers of topics)")

        fig2 = go.Figure()

        countries = selected_data.sort("momentum_score", descending=True)['country_name'].to_list()
        # Order: positive patterns (green) -> neutral (gray) -> negative patterns (orange/red)
        patterns = [
            ('pct_accelerating', 'Accelerating', '#2E7D32'),
            ('pct_consolidating', 'Consolidating', '#66BB6A'),
            ('pct_recovering', 'Recovering', '#4CAF50'),
            ('pct_steady', 'Steady', '#9E9E9E'),
            ('pct_stabilising', 'Stabilising', '#78909C'),
            ('pct_decelerating', 'Decelerating', '#FF9800'),
            ('pct_declining', 'Declining', '#EF5350'),
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
# STRATEGIC ALIGNMENT VIEW
# =============================================================================

def strategic_alignment(data):
    """Industrial Strategy alignment analysis."""
    st.header("Strategic Alignment")
    st.markdown("How UK research activity aligns with Industrial Strategy priorities.")

    # Methodology note
    with st.expander("How priorities were mapped", expanded=False):
        st.markdown("""
        **Source Documents**: 10 UK Industrial Strategy documents (main strategy + 8 sector plans)

        **Priority Extraction**: An LLM extracted 209 unique policy priorities from the documents,
        capturing priority names, descriptions, keywords, and source references.

        **Topic Matching**: Each priority was matched to OpenAlex research topics using:
        1. Text embeddings (all-MiniLM-L6-v2) of priorities and topics
        2. Top-75 candidate topics selected per priority by cosine similarity
        3. LLM validation classifying each candidate as Core, Peripheral, or Not Relevant

        **Result**: 1,681 core topic mappings across 203 priorities, covering ~15% of the OpenAlex topic space.
        """)

    # Explanation of metrics
    with st.expander("Understanding the metrics", expanded=False):
        st.markdown("""
        **Both metrics are UK-specific** — they measure different aspects of UK research performance.

        **UK Momentum** measures whether UK's trajectory is improving or worsening:
        - Compares UK's **recent** performance (2016-24) against its **early** performance (2010-17)
        - Calculated as: (Accelerating + Consolidating topics) − (Declining + Rapid Retreat topics)
        - **Positive values** (e.g., +10) mean UK is performing better recently than it was historically
        - **Negative values** (e.g., −5) mean UK's trajectory has worsened — declining faster or growing slower than before
        - ACCELERATING = UK improving faster recently; RAPID_RETREAT = UK declining faster recently

        **UK Share Trend** measures the overall direction of UK's share:
        - Calculated as: (Topics where UK share is increasing) − (Topics where UK share is declining)
        - **Positive values** mean UK is gaining share in more topics than it's losing
        - **Negative values** mean UK is losing share in more topics than it's gaining
        - INCREASING/STABLE/DECLINING based on the slope of UK's share over time

        **Strategic Concern** flags priorities where *both* metrics are negative — UK's trajectory is worsening *and* UK is losing share overall.
        """)

    sectors = data['ind_strat_sectors']
    priorities = data['ind_strat_priorities'].unique(subset=['priority_id'])  # Deduplicate
    mappings = data['ind_strat_mappings']
    priority_details = data['ind_strat_priority_details']

    # ---------------------------------------------------------------------
    # SECTOR OVERVIEW - UK POSITION FOCUS
    # ---------------------------------------------------------------------
    st.subheader("Sector Overview: UK Position")
    st.markdown("How is UK research share changing across Industrial Strategy sectors?")

    # Sort sectors by UK position (most important for UK policy)
    sectors_sorted = sectors.sort("avg_uk_position", descending=True)

    # Create two columns: chart and metrics
    col1, col2 = st.columns([2, 1])

    with col1:
        # Grouped bar chart: UK Share Trend and UK Momentum side by side
        fig_sectors = go.Figure()

        sector_names = sectors_sorted['parent_sector'].to_list()
        uk_pos_vals = sectors_sorted['avg_uk_position'].to_list()
        momentum_vals = sectors_sorted['avg_momentum'].to_list()

        # Get sector colors (reversed to match bar order)
        sector_colors = [SECTOR_COLORS.get(s, '#757575') for s in sector_names[::-1]]

        # UK Share Trend bars (primary - overall direction) - full sector color
        fig_sectors.add_trace(go.Bar(
            y=sector_names[::-1],
            x=uk_pos_vals[::-1],
            orientation='h',
            name='UK Share Trend',
            marker_color=sector_colors,
            text=[f"{v:+.1f}" for v in uk_pos_vals[::-1]],
            textposition='outside'
        ))

        # UK Momentum bars (secondary - trajectory change) - lighter/transparent
        fig_sectors.add_trace(go.Bar(
            y=sector_names[::-1],
            x=momentum_vals[::-1],
            orientation='h',
            name='UK Momentum',
            marker_color=sector_colors,
            marker_opacity=0.5,
            text=[f"{v:+.1f}" for v in momentum_vals[::-1]],
            textposition='outside'
        ))

        # Add vertical line at x=0
        fig_sectors.add_vline(x=0, line_dash="dash", line_color="#757575")

        fig_sectors.update_layout(
            title=dict(
                text="UK Share Trend vs UK Momentum by Sector<br><sup>Top bar: UK Momentum (faded) · Bottom bar: UK Share Trend (solid)</sup>",
                font=dict(size=16)
            ),
            xaxis_title="Score (positive = improving, negative = worsening)",
            yaxis_title="",
            height=700,
            margin=dict(l=200, r=80, t=80, b=50),
            barmode='group',
            showlegend=False
        )

        st.plotly_chart(fig_sectors, use_container_width=True)

    with col2:
        st.markdown("**Sector Summary**")
        st.caption("Sorted by UK Share Trend (best to worst)")
        # Display key metrics
        for row in sectors_sorted.iter_rows(named=True):
            sector = row['parent_sector']
            momentum = row['avg_momentum']
            uk_pos = row['avg_uk_position']
            concerns = row['concern_count']
            color = SECTOR_COLORS.get(sector, '#757575')

            # Icon based on UK share trend
            uk_icon = "🟢" if uk_pos > 0.5 else ("🟡" if uk_pos >= -0.5 else "🔴")
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem; padding: 0.3rem; border-left: 3px solid {color}; padding-left: 0.5rem;">
                <strong>{sector}</strong><br/>
                <span style="font-size: 0.85rem;">
                    {uk_icon} Share: {uk_pos:+.1f} | Momentum: {momentum:+.1f} | {concerns} concerns
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # PREPARE PRIORITY DATA FOR VISUALIZATIONS
    # ---------------------------------------------------------------------
    # Prepare data for all charts
    scatter_data = priorities.select([
        'priority_id', 'priority_name', 'core_topic_count',
        'total_uk_papers', 'avg_pattern_momentum', 'avg_uk_momentum'
    ]).join(
        # Get parent_sector from priority_details
        pl.DataFrame([
            {'priority_id': pid, 'parent_sector': pdata.get('parent_sector', 'Unknown')}
            for pid, pdata in priority_details.items()
        ]),
        on='priority_id',
        how='left'
    )

    # Calculate momentum_balance and uk_position for each priority
    priority_metrics = priorities.select([
        'priority_id',
        'accelerating_count', 'consolidating_count',
        'declining_count', 'rapid_retreat_count',
        'uk_increasing_count', 'uk_declining_count'
    ])

    priority_metrics = priority_metrics.with_columns([
        (pl.col('accelerating_count') + pl.col('consolidating_count') -
         pl.col('declining_count') - pl.col('rapid_retreat_count')).alias('momentum_balance'),
        (pl.col('uk_increasing_count') - pl.col('uk_declining_count')).alias('uk_position')
    ])

    scatter_data = scatter_data.join(
        priority_metrics.select(['priority_id', 'momentum_balance', 'uk_position']),
        on='priority_id',
        how='left'
    )

    # ---------------------------------------------------------------------
    # CHART 1: BUBBLE CHART WITH PAPER VOLUME
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Priority Portfolio Map")
    st.markdown("""
    Each bubble is an Industrial Strategy priority. **Size = UK research volume** (papers).
    Hover to see details. Larger bubbles represent priorities with more UK research activity.
    """)

    fig_bubble = go.Figure()

    # Normalize bubble sizes (sqrt scale for area perception)
    max_papers = scatter_data['total_uk_papers'].max()
    min_size, max_size = 8, 50

    for sector in SECTOR_COLORS.keys():
        sector_df = scatter_data.filter(pl.col('parent_sector') == sector)
        if sector_df.height == 0:
            continue

        papers = sector_df['total_uk_papers'].to_list()
        sizes = [min_size + (max_size - min_size) * ((p / max_papers) ** 0.5) for p in papers]

        fig_bubble.add_trace(go.Scatter(
            x=sector_df['momentum_balance'].to_list(),
            y=sector_df['uk_position'].to_list(),
            mode='markers',
            name=sector,
            marker=dict(
                size=sizes,
                color=SECTOR_COLORS[sector],
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            text=sector_df['priority_name'].to_list(),
            customdata=list(zip(papers, sector_df['core_topic_count'].to_list())),
            hovertemplate="<b>%{text}</b><br>UK Momentum: %{x}<br>UK Share Trend: %{y}<br>UK Papers: %{customdata[0]:,}<br>Topics: %{customdata[1]}<extra></extra>"
        ))

    # Add quadrant lines
    fig_bubble.add_hline(y=0, line_dash="dash", line_color="#757575", opacity=0.5)
    fig_bubble.add_vline(x=0, line_dash="dash", line_color="#757575", opacity=0.5)

    fig_bubble.update_layout(
        xaxis_title="UK Momentum (positive = trajectory improving)",
        yaxis_title="UK Share Trend (positive = share increasing)",
        height=500,
        margin=dict(l=50, r=50, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    # ---------------------------------------------------------------------
    # CHART 2: RESIDUALS - UK PERFORMANCE VS EXPECTED
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("UK Performance Anomalies")
    st.markdown("""
    Since Momentum and UK Position are correlated, this chart shows **anomalies** —
    priorities where UK is doing better or worse than expected given the field's momentum.

    - **Positive residual** (green): UK outperforming — gaining more share than typical for this momentum level
    - **Negative residual** (red): UK underperforming — losing more share than typical for this momentum level
    """)

    # Calculate linear regression: uk_position = a * momentum + b
    momentum_vals = scatter_data['momentum_balance'].to_numpy()
    uk_pos_vals = scatter_data['uk_position'].to_numpy()

    # Simple linear regression
    n = len(momentum_vals)
    sum_x = momentum_vals.sum()
    sum_y = uk_pos_vals.sum()
    sum_xy = (momentum_vals * uk_pos_vals).sum()
    sum_x2 = (momentum_vals ** 2).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    # Calculate residuals (actual - expected)
    expected_uk_pos = slope * momentum_vals + intercept
    residuals = uk_pos_vals - expected_uk_pos

    scatter_data = scatter_data.with_columns([
        pl.Series('expected_uk_position', expected_uk_pos),
        pl.Series('residual', residuals)
    ])

    # Get top outperformers and underperformers
    top_outperformers = scatter_data.sort('residual', descending=True).head(10)
    top_underperformers = scatter_data.sort('residual').head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**UK Outperforming** (better than expected)")
        fig_out = go.Figure()
        fig_out.add_trace(go.Bar(
            y=top_outperformers['priority_name'].to_list()[::-1],
            x=top_outperformers['residual'].to_list()[::-1],
            orientation='h',
            marker_color='#4CAF50',
            text=[f"+{r:.1f}" for r in top_outperformers['residual'].to_list()[::-1]],
            textposition='outside'
        ))
        fig_out.update_layout(
            xaxis_title="Residual (UK Position vs Expected)",
            yaxis_title="",
            height=400,
            margin=dict(l=250, r=60, t=20, b=50)
        )
        st.plotly_chart(fig_out, use_container_width=True)

    with col2:
        st.markdown("**UK Underperforming** (worse than expected)")
        fig_under = go.Figure()
        fig_under.add_trace(go.Bar(
            y=top_underperformers['priority_name'].to_list()[::-1],
            x=top_underperformers['residual'].to_list()[::-1],
            orientation='h',
            marker_color='#C62828',
            text=[f"{r:.1f}" for r in top_underperformers['residual'].to_list()[::-1]],
            textposition='outside'
        ))
        fig_under.update_layout(
            xaxis_title="Residual (UK Position vs Expected)",
            yaxis_title="",
            height=400,
            margin=dict(l=250, r=60, t=20, b=50)
        )
        st.plotly_chart(fig_under, use_container_width=True)

    st.caption(f"Regression: UK Share Trend ≈ {slope:.2f} × UK Momentum + {intercept:.2f}")

    # ---------------------------------------------------------------------
    # OPPORTUNITIES VS CONCERNS TABLES
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Opportunities & Concerns")

    # Opportunities
    st.markdown("**Top Opportunities** — strongest positive UK momentum")
    opportunities = scatter_data.filter(
        pl.col('momentum_balance') > 0
    ).sort('momentum_balance', descending=True).head(10)

    if opportunities.height > 0:
        opp_display = opportunities.select([
            'priority_name', 'parent_sector', 'momentum_balance'
        ]).rename({
            'priority_name': 'Priority',
            'parent_sector': 'Sector',
            'momentum_balance': 'UK Momentum'
        })
        st.dataframe(
            opp_display.to_pandas(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn(width="large"),
                "Sector": st.column_config.TextColumn(width="medium"),
                "UK Momentum": st.column_config.NumberColumn(format="%+d"),
            }
        )
    else:
        st.info("No opportunities identified.")

    # Concerns
    st.markdown("**Top Concerns** — strongest negative UK momentum")
    concerns_chart = scatter_data.filter(
        pl.col('momentum_balance') < 0
    ).sort('momentum_balance').head(10)

    if concerns_chart.height > 0:
        concern_display = concerns_chart.select([
            'priority_name', 'parent_sector', 'momentum_balance'
        ]).rename({
            'priority_name': 'Priority',
            'parent_sector': 'Sector',
            'momentum_balance': 'UK Momentum'
        })
        st.dataframe(
            concern_display.to_pandas(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn(width="large"),
                "Sector": st.column_config.TextColumn(width="medium"),
                "UK Momentum": st.column_config.NumberColumn(format="%+d"),
            }
        )
    else:
        st.info("No concerns identified.")

    # ---------------------------------------------------------------------
    # STRATEGIC CONCERNS
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Strategic Concerns")
    st.markdown("Priorities where UK momentum **and** UK share trend are both negative.")

    # Filter to strategic concerns (both negative), deduplicate by priority_name
    concerns = scatter_data.filter(
        (pl.col('momentum_balance') < 0) & (pl.col('uk_position') < 0)
    ).unique(subset=['priority_name']).sort('momentum_balance')

    if concerns.height > 0:
        st.warning(f"**{concerns.height} priorities** identified as strategic concerns.")
        st.caption("Sorted by UK momentum (most negative first). These priorities may require intervention or managed exit.")

        # Display as table
        concerns_display = concerns.select([
            'priority_name', 'parent_sector', 'momentum_balance', 'core_topic_count'
        ]).rename({
            'priority_name': 'Priority',
            'parent_sector': 'Sector',
            'momentum_balance': 'UK Momentum',
            'core_topic_count': 'Topics'
        })

        st.dataframe(
            concerns_display.to_pandas(),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority": st.column_config.TextColumn(width="large"),
                "Sector": st.column_config.TextColumn(width="medium"),
                "UK Momentum": st.column_config.NumberColumn(format="%+d", help="UK trajectory: recent vs early"),
                "Topics": st.column_config.NumberColumn(format="%d"),
            }
        )
    else:
        st.success("No strategic concerns identified.")

    # ---------------------------------------------------------------------
    # PRIORITY BROWSER
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Priority Browser")
    st.markdown("Explore individual priorities and their mapped research topics.")

    # Sector filter
    selected_sector = st.selectbox(
        "Filter by Sector",
        options=["All Sectors"] + list(SECTOR_COLORS.keys()),
        index=0,
        key="priority_browser_sector"
    )

    # Get priorities with sector info
    priorities_with_sector = priorities.join(
        pl.DataFrame([
            {'priority_id': pid, 'parent_sector': pdata.get('parent_sector', 'Unknown')}
            for pid, pdata in priority_details.items()
        ]),
        on='priority_id',
        how='left'
    )

    if selected_sector != "All Sectors":
        priorities_with_sector = priorities_with_sector.filter(
            pl.col('parent_sector') == selected_sector
        )

    # Sort by momentum (worst first to highlight concerns)
    priorities_with_sector = priorities_with_sector.join(
        priority_metrics.select(['priority_id', 'momentum_balance', 'uk_position']),
        on='priority_id',
        how='left'
    ).sort('momentum_balance')

    # Priority selector
    priority_options = priorities_with_sector.select(['priority_id', 'priority_name']).to_dicts()
    priority_names = {p['priority_id']: p['priority_name'] for p in priority_options}

    if len(priority_names) == 0:
        st.info("No priorities found for selected sector.")
        return

    # Use session state to preserve selection
    priority_key = f"priority_select_{selected_sector}"
    selected_priority_id = st.selectbox(
        "Select Priority",
        options=list(priority_names.keys()),
        format_func=lambda x: priority_names[x],
        index=0,
        key=priority_key
    )

    # Display priority details
    priority_row = priorities_with_sector.filter(
        pl.col('priority_id') == selected_priority_id
    ).row(0, named=True)

    priority_detail = priority_details.get(selected_priority_id, {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        momentum = priority_row.get('momentum_balance', 0)
        momentum_label = "Improving" if momentum > 0 else ("Declining" if momentum < 0 else "Stable")
        st.metric("UK Momentum", momentum_label, delta=momentum)
    with col2:
        uk_pos = priority_row.get('uk_position', 0)
        uk_pos_label = "Gaining" if uk_pos > 0 else ("Losing" if uk_pos < 0 else "Stable")
        st.metric("UK Share Trend", uk_pos_label, delta=uk_pos)
    with col3:
        st.metric("Core Topics", priority_row.get('core_topic_count', 0))
    with col4:
        st.metric("UK Papers", format_number(priority_row.get('total_uk_papers', 0)))

    st.caption("**UK Momentum** = is UK's trajectory improving or worsening? (recent vs early period). **UK Share Trend** = is UK's share increasing or decreasing overall? These can differ: e.g. positive momentum + negative trend = recovering from decline.")

    # Show description
    if priority_detail.get('description'):
        st.markdown(f"**Description:** {priority_detail['description']}")

    # Show mapped topics
    st.markdown("#### Mapped Research Topics")
    st.caption("**UK Momentum** compares UK's recent trajectory (2016-24) vs early period (2010-17). 🚀 Accelerating = UK improving faster. 🚨 Rapid Retreat = UK declining faster.")

    priority_topics = mappings.filter(pl.col('priority_id') == selected_priority_id)

    if priority_topics.height > 0:
        # Format with icons to avoid Streamlit auto-styling and be more readable
        topics_df = priority_topics.to_pandas()

        # Map trajectory patterns to icons + text
        momentum_map = {
            'ACCELERATING': '🚀 Accelerating',
            'CONSOLIDATING': '📊 Consolidating',
            'RECOVERING': '📈 Recovering',
            'STEADY': '➡️ Steady',
            'STABILISING': '⏸️ Stabilising',
            'DECELERATING': '📉 Decelerating',
            'DECLINING': '⬇️ Declining',
            'RAPID_RETREAT': '🚨 Rapid Retreat',
        }
        topics_df['UK Momentum'] = topics_df['trajectory_pattern'].map(
            lambda x: momentum_map.get(x, x)
        )

        # Rename columns
        topics_df = topics_df.rename(columns={
            'topic_name': 'Topic',
            'field_name': 'Field',
            'current_uk_papers': 'UK Papers',
            'current_uk_share': 'UK Share %'
        })

        # Select columns for display (dropping uk_trend as requested)
        topics_display = topics_df[['Topic', 'Field', 'UK Momentum', 'UK Papers', 'UK Share %']]

        st.dataframe(
            topics_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Topic": st.column_config.TextColumn(width="large"),
                "Field": st.column_config.TextColumn(width="medium"),
                "UK Momentum": st.column_config.TextColumn(width="small", help="UK trajectory: comparing recent vs early period"),
                "UK Papers": st.column_config.NumberColumn(format="%d"),
                "UK Share %": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    else:
        st.info("No topics mapped to this priority.")


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

    # Custom CSS (font handled by .streamlit/config.toml)
    st.markdown("""
    <style>
        .block-container {padding-top: 2rem;}
        section[data-testid="stSidebar"] .block-container {padding-top: 0.5rem;}

        /* Reduce sidebar header space */
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: auto;
            min-height: 0;
            padding: 0.5rem 1rem;
        }

        /* Reduce heading padding in sidebar */
        section[data-testid="stSidebar"] .stHeading {
            padding-top: 0;
            padding-bottom: 0.5rem;
        }

        /* Smaller text in sidebar */
        section[data-testid="stSidebar"] .stMarkdown p {
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # View options
    views = ["Topic Browser", "Topic Detail", "Bloc Comparison", "Country-by-Country Analysis", "UK Strengths Dashboard", "Strategic Alignment"]

    # Handle navigation requests (must happen before widget is created)
    # Setting session state directly before widget creation works
    if '_navigate_to' in st.session_state:
        st.session_state['view_selector'] = st.session_state.pop('_navigate_to')
        st.session_state['_should_scroll'] = True  # Mark that we should scroll

    # Sidebar navigation
    st.sidebar.title("UK Research Explorer")
    st.sidebar.markdown("Explore UK research positioning across 4,516 topics.")

    # View selector
    view = st.sidebar.radio(
        "Select View",
        views,
        key="view_selector"
    )

    # Scroll to top only on view change
    from streamlit_scroll_to_top import scroll_to_here
    if st.session_state.get('_should_scroll', False):
        scroll_to_here(0, key='scroll_top')
        st.session_state['_should_scroll'] = False
    # Track current view for detecting changes
    prev_view = st.session_state.get('_prev_view', None)
    if prev_view != view:
        st.session_state['_prev_view'] = view
        if prev_view is not None:  # Don't scroll on initial load
            scroll_to_here(0, key='scroll_top_view_change')

    # About section
    st.sidebar.markdown("---")
    st.sidebar.caption("""
    UK research trends across 4,516 topics (OpenAlex 2010-24).
    Mapped to Industrial Strategy priorities.

    **Ben Johnson**, University of Strathclyde
    """)
    st.sidebar.caption("ben.johnson \\[at\\] strath.ac.uk")

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
    elif view == "Country-by-Country Analysis":
        country_analysis(data)
    elif view == "UK Strengths Dashboard":
        uk_strengths_dashboard(data)
    elif view == "Strategic Alignment":
        strategic_alignment(data)


if __name__ == "__main__":
    main()
