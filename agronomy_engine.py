import datetime
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from config import CROP_PROFILES

def calculate_gdd(t_max: float, t_min: float, t_base: float, t_upper: float = 35.0) -> float:
    """Calculate single day Growing Degree Days (GDD) with base and upper thresholds."""
    t_max_adj = min(t_max, t_upper)
    t_min_adj = max(t_min, t_base)
    t_avg = (t_max_adj + t_min_adj) / 2.0
    return max(0.0, t_avg - t_base)

def simulate_phenology_and_gdd(
    crop_name: str,
    sowing_date_str: str,
    current_gdd_override: float = None,
    daily_avg_temp: float = 24.0,
    daily_temp_swing: float = 12.0
) -> Dict[str, Any]:
    """
    Simulates cumulative GDD, stage progression, and projected harvest window.
    """
    profile = CROP_PROFILES.get(crop_name, CROP_PROFILES["Sweet Corn / Maize"])
    t_base = profile["base_temp_c"]
    t_upper = profile["upper_temp_c"]
    target_gdd = profile["target_gdd"]
    
    try:
        sowing_date = datetime.datetime.strptime(sowing_date_str, "%Y-%m-%d").date()
    except Exception:
        sowing_date = datetime.date.today() - datetime.timedelta(days=60)
    
    today = datetime.date.today()
    days_elapsed = max(1, (today - sowing_date).days)
    
    # Estimate standard daily GDD for typical season
    t_max = daily_avg_temp + (daily_temp_swing / 2.0)
    t_min = daily_avg_temp - (daily_temp_swing / 2.0)
    est_daily_gdd = calculate_gdd(t_max, t_min, t_base, t_upper)
    if est_daily_gdd <= 0.5:
        est_daily_gdd = 8.5  # safe minimum
    
    if current_gdd_override is not None and current_gdd_override > 0:
        accumulated_gdd = float(current_gdd_override)
    else:
        accumulated_gdd = round(days_elapsed * est_daily_gdd, 1)
        
    completion_pct = min(100.0, round((accumulated_gdd / target_gdd) * 100.0, 1))
    
    # Calculate phenological stage index
    stages = profile["stages"]
    num_stages = len(stages)
    stage_idx = min(num_stages - 1, int((completion_pct / 100.0) * num_stages))
    current_stage = stages[stage_idx]
    
    # Calculate days remaining
    remaining_gdd = max(0.0, target_gdd - accumulated_gdd)
    days_remaining = int(round(remaining_gdd / est_daily_gdd))
    
    predicted_harvest_date = today + datetime.timedelta(days=days_remaining)
    window_start = predicted_harvest_date - datetime.timedelta(days=2)
    window_end = predicted_harvest_date + datetime.timedelta(days=3)
    
    # Status categorisation for Jira
    if completion_pct >= 95.0 or days_remaining <= 3:
        jira_status = "OPTIMAL_HARVEST"
    elif completion_pct >= 75.0 or days_remaining <= 14:
        jira_status = "RIPENING"
    else:
        jira_status = "VEGETATIVE"
        
    readiness_score = int(min(100, max(10, completion_pct)))
    
    return {
        "crop_name": crop_name,
        "sowing_date": sowing_date.strftime("%Y-%m-%d"),
        "days_elapsed": days_elapsed,
        "accumulated_gdd": accumulated_gdd,
        "target_gdd": target_gdd,
        "completion_pct": completion_pct,
        "current_stage": current_stage,
        "daily_gdd_rate": round(est_daily_gdd, 1),
        "days_remaining": days_remaining,
        "predicted_harvest_date": predicted_harvest_date.strftime("%Y-%m-%d"),
        "window_start": window_start.strftime("%Y-%m-%d"),
        "window_end": window_end.strftime("%Y-%m-%d"),
        "jira_status": jira_status,
        "readiness_score": readiness_score,
        "target_metric_name": profile["target_metric_name"],
        "target_metric_ideal": profile["target_metric_ideal"]
    }

def create_jira_gdd_plot(analysis_data: Dict[str, Any]) -> go.Figure:
    """
    Creates a Jira-styled interactive GDD trajectory and maturity plot.
    """
    days_elapsed = analysis_data.get("days_elapsed", 60)
    days_remaining = analysis_data.get("days_remaining", 15)
    total_days = days_elapsed + days_remaining + 10
    daily_gdd = analysis_data.get("daily_gdd_rate", 12.0)
    target_gdd = analysis_data.get("target_gdd", 1400)
    accumulated_gdd = analysis_data.get("accumulated_gdd", 1000)
    
    # Timeline points
    past_days = list(range(0, days_elapsed + 1, 3))
    past_gdd = [min(accumulated_gdd, round(d * daily_gdd * (1.0 + (d % 5 - 2) * 0.03), 1)) for d in past_days]
    
    future_days = list(range(days_elapsed, total_days + 1, 3))
    future_gdd = [round(accumulated_gdd + (d - days_elapsed) * daily_gdd, 1) for d in future_days]
    
    fig = go.Figure()
    
    # CropHarvest modern command center palette
    ch_primary = "#34D399"
    ch_dark = "#FFFFFF"
    ch_accent = "#10B981"
    ch_amber = "#FBBF24"
    ch_gray = "#9EB5AC"
    
    # Target Threshold Line
    fig.add_hline(
        y=target_gdd,
        line_dash="dash",
        line_color=ch_accent,
        annotation_text=f"Target GDD: {target_gdd} (Peak Quality)",
        annotation_position="top left",
        annotation_font_color=ch_primary,
        annotation_font_size=12
    )
    
    # Historical GDD Trace
    fig.add_trace(go.Scatter(
        x=past_days,
        y=past_gdd,
        mode='lines+markers',
        name='Accumulated GDD (Past)',
        line=dict(color="#10B981", width=3),
        marker=dict(size=6, color="#34D399"),
        hovertemplate='Day %{x}: %{y:.1f} GDD<extra></extra>'
    ))
    
    # Projected Forecast Trace
    fig.add_trace(go.Scatter(
        x=future_days,
        y=future_gdd,
        mode='lines',
        name='AI Projection Trajectory',
        line=dict(color=ch_amber, width=3, dash='dot'),
        hovertemplate='Forecast Day %{x}: %{y:.1f} GDD<extra></extra>'
    ))
    
    # Optimal Window Highlight
    harvest_day = days_elapsed + days_remaining
    fig.add_vrect(
        x0=max(0, harvest_day - 2),
        x1=harvest_day + 3,
        fillcolor="rgba(16, 185, 129, 0.22)",
        opacity=0.22,
        layer="below",
        line_width=1,
        line_color="#34D399",
        annotation_text="🎯 Optimal Harvest Window",
        annotation_position="top right",
        annotation_font_color="#34D399"
    )
    
    # Current Day vertical marker
    fig.add_vline(
        x=days_elapsed,
        line_width=2,
        line_color="#FFFFFF",
        annotation_text="Today",
        annotation_position="bottom right",
        annotation_font_color="#FFFFFF"
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>Growing Degree Days (GDD) Trajectory</b> • {analysis_data.get('crop_name', 'Crop')}",
            font=dict(family="Outfit, Plus Jakarta Sans, sans-serif", size=15, color="#FFFFFF")
        ),
        xaxis=dict(
            title=dict(text="Days from Planting / Bloom", font=dict(color="#E6F1EC")),
            gridcolor="rgba(52, 211, 153, 0.12)",
            zerolinecolor="rgba(52, 211, 153, 0.12)",
            showline=True,
            linecolor="rgba(52, 211, 153, 0.25)",
            tickfont=dict(color="#9EB5AC")
        ),
        yaxis=dict(
            title=dict(text="Cumulative GDD (°C-days)", font=dict(color="#E6F1EC")),
            gridcolor="rgba(52, 211, 153, 0.12)",
            zerolinecolor="rgba(52, 211, 153, 0.12)",
            showline=True,
            linecolor="rgba(52, 211, 153, 0.25)",
            tickfont=dict(color="#9EB5AC")
        ),
        paper_bgcolor="#132822",
        plot_bgcolor="#081512",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#A7F3D0")
        ),
        margin=dict(l=50, r=40, t=55, b=40),
        height=320
    )
    
    return fig
