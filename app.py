import os
import datetime
from typing import Dict, Any, List, Tuple
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    CROP_PROFILES,
    INITIAL_TICKETS,
    GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_GROK_MODEL,
    DEFAULT_GROK_API_KEY
)
from agronomy_engine import simulate_phenology_and_gdd, create_jira_gdd_plot
from groq_agent import analyze_harvest_with_groq, ask_groq_copilot, get_groq_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "jira_theme.css")

if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        CUSTOM_CSS = f.read()
else:
    CUSTOM_CSS = ""

# Quick presets
PRESET_DATA = {
    "🍇 Cabernet Grapes": {
        "crop": "Wine Grapes (Cabernet/Pinot)",
        "date": (datetime.date.today() - datetime.timedelta(days=144)).strftime("%Y-%m-%d"),
        "region": "California (Sonoma Valley)",
        "notes": "Warm dry sunny weather, berries softening and reaching deep purple"
    },
    "🌽 Sweet Corn": {
        "crop": "Sweet Corn / Maize",
        "date": (datetime.date.today() - datetime.timedelta(days=77)).strftime("%Y-%m-%d"),
        "region": "Midwest (Iowa)",
        "notes": "Silks turning brown, ears filling out with milky kernels"
    },
    "🍅 Tomatoes": {
        "crop": "Processing Tomato",
        "date": (datetime.date.today() - datetime.timedelta(days=106)).strftime("%Y-%m-%d"),
        "region": "Central Valley, CA",
        "notes": "Fruit clusters turning 85%+ red, firm skin"
    },
    "🌾 Wheat": {
        "crop": "Winter / Spring Wheat",
        "date": (datetime.date.today() - datetime.timedelta(days=110)).strftime("%Y-%m-%d"),
        "region": "Great Plains (Kansas)",
        "notes": "Heads golden brown, grain firming up"
    }
}

def render_agent_output(ai_result: Dict[str, Any], crop: str, region: str) -> str:
    readiness = int(ai_result.get("readiness_score", 85))
    harvest_date = ai_result.get("predicted_harvest_date", "2026-09-14")
    window_start = ai_result.get("window_start", "2026-09-12")
    window_end = ai_result.get("window_end", "2026-09-16")
    days_rem = ai_result.get("days_remaining", 7)
    stage = ai_result.get("current_stage", "Maturation")
    quality = ai_result.get("predicted_quality", "Optimal ripeness reached.")
    risk = str(ai_result.get("risk_level", "Low"))
    mitigation = ai_result.get("risk_mitigation", "Monitor daily weather.")
    rationale = ai_result.get("ai_rationale", "GDD accumulation aligns with peak quality curve.")
    source = ai_result.get("source", "Groq AI Agent")
    accumulated_gdd = ai_result.get('accumulated_gdd', 0)
    target_gdd = ai_result.get('target_gdd', 0)

    # Status Pill Styling
    if readiness >= 90:
        status_pill_bg = "rgba(16, 185, 129, 0.2)"
        status_pill_text = "#34D399"
        status_border = "#10B981"
        status_label = "READY TO HARVEST"
        progress_color = "#10B981"
    elif readiness >= 75:
        status_pill_bg = "rgba(245, 158, 11, 0.2)"
        status_pill_text = "#FBBF24"
        status_border = "#F59E0B"
        status_label = "APPROACHING MATURITY"
        progress_color = "#F59E0B"
    else:
        status_pill_bg = "rgba(56, 189, 248, 0.2)"
        status_pill_text = "#38BDF8"
        status_border = "#0284C7"
        status_label = "IN DEVELOPMENT"
        progress_color = "#38BDF8"

    # Risk badge styling
    risk_lower = risk.lower()
    if "high" in risk_lower or "urgent" in risk_lower:
        risk_color = "#FCA5A5"
        risk_bg = "rgba(239, 68, 68, 0.2)"
        risk_border = "#EF4444"
    elif "mod" in risk_lower:
        risk_color = "#FDE68A"
        risk_bg = "rgba(245, 158, 11, 0.2)"
        risk_border = "#F59E0B"
    else:
        risk_color = "#A7F3D0"
        risk_bg = "rgba(16, 185, 129, 0.2)"
        risk_border = "#10B981"

    display_source = "Groq LPU Engine" if "groq api" in source.lower() else "Agronomy Simulation"

    return f'''
    <div style="background: #132822; border: 1px solid rgba(52, 211, 153, 0.2); border-radius: 12px; padding: 22px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        
        <!-- Status Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid rgba(52, 211, 153, 0.15); padding-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 16px;">🌱</span>
                <span style="font-size: 13px; font-weight: 800; color: #FFFFFF; text-transform: uppercase; letter-spacing: .5px;">
                    Field Decision Brief • {crop} ({region})
                </span>
            </div>
            <span style="background: {status_pill_bg}; color: {status_pill_text}; border: 1px solid {status_border}; font-size: 11.5px; font-weight: 800; padding: 4px 14px; border-radius: 9999px; letter-spacing: 0.3px;">
                ● {status_label}
            </span>
        </div>

        <!-- Target Date Hero Banner -->
        <div style="background: linear-gradient(135deg, #0d221c 0%, #15382e 100%); border: 1px solid rgba(52, 211, 153, 0.25); border-radius: 10px; padding: 18px 22px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            <div>
                <div style="font-size: 11px; font-weight: 700; color: #6EE7B7; text-transform: uppercase; letter-spacing: 0.6px;">Predicted Harvest Date</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 32px; font-weight: 800; color: #34D399; line-height: 1.1; margin-top: 4px; text-shadow: 0 0 16px rgba(52, 211, 153, 0.3);">{harvest_date}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; font-weight: 700; color: #6EE7B7; text-transform: uppercase; letter-spacing: 0.6px;">Optimal Window</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 700; color: #FFFFFF; margin-top: 3px;">{window_start} → {window_end}</div>
                <div style="display: inline-block; margin-top: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(52, 211, 153, 0.35); border-radius: 9999px; padding: 3px 12px; font-size: 12px; font-weight: 600; color: #A7F3D0;">⏱️ ~{days_rem} days remaining</div>
            </div>
        </div>

        <!-- 3 Metrics Grid -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px;">
            <!-- Metric 1: Readiness -->
            <div style="background: #0d1e19; border: 1px solid rgba(52, 211, 153, 0.18); border-radius: 10px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #6EE7B7; text-transform: uppercase; letter-spacing: 0.5px;">Readiness Score</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 800; color: {progress_color}; margin: 2px 0 6px 0;">{readiness}%</div>
                <!-- Progress Bar -->
                <div style="width: 100%; height: 6px; background: #081512; border-radius: 9999px; overflow: hidden; margin-bottom: 6px; border: 1px solid rgba(52, 211, 153, 0.15);">
                    <div style="width: {min(readiness, 100)}%; height: 100%; background: {progress_color}; border-radius: 9999px;"></div>
                </div>
                <div style="font-size: 11.5px; color: #9EB5AC; font-weight: 500;">Stage: <b style="color: #FFFFFF;">{stage}</b></div>
            </div>

            <!-- Metric 2: Thermal Units -->
            <div style="background: #0d1e19; border: 1px solid rgba(52, 211, 153, 0.18); border-radius: 10px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #6EE7B7; text-transform: uppercase; letter-spacing: 0.5px;">Thermal Units (GDD)</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 800; color: #FFFFFF; margin: 2px 0 6px 0;">{accumulated_gdd} <span style="font-size: 13px; font-weight: 600; color: #6EE7B7;">GDD</span></div>
                <div style="font-size: 11.5px; color: #9EB5AC; margin-top: 8px;">Target: <b style="color: #FFFFFF;">{target_gdd} GDD</b></div>
            </div>

            <!-- Metric 3: Risk Evaluation -->
            <div style="background: #0d1e19; border: 1px solid rgba(52, 211, 153, 0.18); border-radius: 10px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #6EE7B7; text-transform: uppercase; letter-spacing: 0.5px;">Risk Assessment</div>
                <div style="margin: 4px 0 8px 0;">
                    <span style="background: {risk_bg}; color: {risk_color}; border: 1px solid {risk_border}; font-size: 12.5px; font-weight: 800; padding: 4px 10px; border-radius: 6px; display: inline-block;">
                        {risk}
                    </span>
                </div>
                <div style="font-size: 11.5px; color: #9EB5AC; margin-top: 6px;">Weather & Field Impact</div>
            </div>
        </div>

        <!-- AI Agent Explanation -->
        <div style="background: #0d1e19; border: 1px solid rgba(52, 211, 153, 0.2); border-left: 4px solid #10B981; padding: 14px 18px; border-radius: 0 10px 10px 0; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 13px; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 6px;">
                    <span>🧠</span> Agronomic Reasoning
                </div>
                <span style="font-size: 11px; font-weight: 700; background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 2px 10px; border-radius: 9999px;">
                    {display_source}
                </span>
            </div>
            <div style="font-size: 13.5px; color: #E6F1EC; line-height: 1.6;">
                {rationale}
            </div>
        </div>

        <!-- Actionable Recommendations -->
        <div style="background: #181c13; border: 1px solid rgba(245, 158, 11, 0.3); border-left: 4px solid #F59E0B; border-radius: 0 10px 10px 0; padding: 14px 18px;">
            <div style="font-size: 13px; font-weight: 800; color: #FBBF24; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                <span>📋</span> Operational Action Plan
            </div>
            <div style="font-size: 13px; color: #FEF3C7; line-height: 1.6;">
                <div>• <b>Target Metric:</b> {quality}</div>
                <div>• <b>Field Logistics:</b> {mitigation}</div>
            </div>
        </div>

    </div>
    '''

def render_kanban_board(tickets: List[Dict[str, Any]]) -> str:
    """Generates a responsive Jira-style Kanban board of active field tickets."""
    col_optimal = [t for t in tickets if t.get("status") == "OPTIMAL_HARVEST"]
    col_ripening = [t for t in tickets if t.get("status") == "RIPENING"]
    col_vegetative = [t for t in tickets if t.get("status") == "VEGETATIVE"]

    def make_card(t: Dict[str, Any]) -> str:
        priority_class = f"priority-{t.get('priority', 'low').lower()}"
        readiness = t.get('readiness_score', 80)
        progress_bg = "#10B981" if readiness >= 90 else ("#F59E0B" if readiness >= 75 else "#3B82F6")
        assignee_name = t.get('assignee', 'Field Lead')
        initial = assignee_name[0] if assignee_name else 'F'

        return f'''
        <div class="kanban-card">
            <div class="kanban-card-top">
                <span class="ticket-id">{t['id']}</span>
                <span class="priority-pill {priority_class}">{t.get('priority', 'MEDIUM')}</span>
            </div>
            <div class="kanban-card-title">{t['summary']}</div>
            <div class="kanban-card-meta">
                <span>📍 {t['location']}</span>
                <span>📐 {t['story_points']} acres</span>
            </div>
            <div class="kanban-card-progress">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #536961;">
                    <span>Readiness: <b>{readiness}%</b></span>
                    <span>{t.get('current_gdd', 0)} / {t.get('target_gdd', 0)} GDD</span>
                </div>
                <div class="kanban-progress-bar">
                    <div class="kanban-progress-fill" style="width: {min(readiness, 100)}%; background: {progress_bg};"></div>
                </div>
            </div>
            <div style="background: #F8FAF9; border-left: 3px solid #10B981; padding: 6px 10px; font-size: 11.5px; border-radius: 4px; color: #1A2E26;">
                <b>Window:</b> {t.get('window_start', 'N/A')} → {t.get('window_end', 'N/A')}
            </div>
            <div class="kanban-card-footer">
                <div class="kanban-assignee">
                    <span class="avatar-circle">{initial}</span>
                    <span>{assignee_name}</span>
                </div>
                <span style="font-weight: 700; color: #184638;">🎯 {t.get('predicted_harvest_date', '')}</span>
            </div>
        </div>
        '''

    cards_optimal = "".join([make_card(t) for t in col_optimal]) or "<div style='color: #667872; font-size: 12px; padding: 12px;'>No active fields</div>"
    cards_ripening = "".join([make_card(t) for t in col_ripening]) or "<div style='color: #667872; font-size: 12px; padding: 12px;'>No active fields</div>"
    cards_vegetative = "".join([make_card(t) for t in col_vegetative]) or "<div style='color: #667872; font-size: 12px; padding: 12px;'>No active fields</div>"

    return f'''
    <div class="kanban-board-container">
        <!-- Column 1 -->
        <div class="kanban-col">
            <div class="kanban-col-header">
                <div class="kanban-col-title">🎯 Optimal Harvest Window</div>
                <span class="kanban-col-count">{len(col_optimal)}</span>
            </div>
            {cards_optimal}
        </div>

        <!-- Column 2 -->
        <div class="kanban-col">
            <div class="kanban-col-header">
                <div class="kanban-col-title">🌾 Maturation & Ripening</div>
                <span class="kanban-col-count">{len(col_ripening)}</span>
            </div>
            {cards_ripening}
        </div>

        <!-- Column 3 -->
        <div class="kanban-col">
            <div class="kanban-col-header">
                <div class="kanban-col-title">🌱 Vegetative Growth</div>
                <span class="kanban-col-count">{len(col_vegetative)}</span>
            </div>
            {cards_vegetative}
        </div>
    </div>
    '''

# Initial default run
init_data = analyze_harvest_with_groq(
    crop="Sweet Corn / Maize",
    variety="Pioneer Sweet Corn",
    sowing_date=(datetime.date.today() - datetime.timedelta(days=77)).strftime("%Y-%m-%d"),
    location="Midwest (Iowa)",
    current_gdd=0.0,
    recent_weather="Warm sunny weather, average 25°C",
    soil_condition="Standard loam",
    target_quality="Kernel moisture 72%",
    api_key=""
)
DEFAULT_OUTPUT_HTML = render_agent_output(init_data, "Sweet Corn / Maize", "Midwest (Iowa)")
DEFAULT_PLOT = create_jira_gdd_plot(init_data)
DEFAULT_KANBAN_HTML = render_kanban_board(INITIAL_TICKETS)

def on_predict_harvest(crop, sowing_date, region, notes, objective):
    field_context = f"Operational objective: {objective}. Field observations: {notes or 'No recent observations provided.'}"
    result = analyze_harvest_with_groq(
        crop=crop,
        variety="Standard",
        sowing_date=sowing_date,
        location=region,
        current_gdd=0.0,
        recent_weather=field_context,
        soil_condition="Well maintained",
        target_quality="",
    )
    brief_html = render_agent_output(result, crop, region)
    plot_fig = create_jira_gdd_plot(result)
    return brief_html, plot_fig

def on_load_preset(preset_key):
    p = PRESET_DATA[preset_key]
    result = analyze_harvest_with_groq(
        crop=p["crop"],
        variety="Standard",
        sowing_date=p["date"],
        location=p["region"],
        current_gdd=0.0,
        recent_weather=p["notes"],
        soil_condition="Well maintained",
        target_quality="",
    )
    output_html = render_agent_output(result, p["crop"], p["region"])
    plot_fig = create_jira_gdd_plot(result)
    return p["crop"], p["date"], p["region"], p["notes"], output_html, plot_fig

def on_ask_agent(message, history, crop, region):
    if not message.strip():
        return history, ""
    answer = ask_groq_copilot(message, INITIAL_TICKETS, history)
    history.append((message, answer))
    return history, ""

def test_api_status(api_key, model):
    key = api_key.strip() if api_key else DEFAULT_GROQ_API_KEY
    if not key:
        return "ℹ️ No API Key configured. System is currently running on the high-fidelity Local Agronomic Engine with zero external dependencies."
    try:
        from groq import Groq
        client = Groq(api_key=key)
        start = datetime.datetime.now()
        client.chat.completions.create(
            model=model or DEFAULT_GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        latency = (datetime.datetime.now() - start).total_seconds() * 1000
        return f"✅ Connected successfully to Groq API ({model or DEFAULT_GROQ_MODEL})! Ultra-fast LPU inference active ({latency:.0f}ms response)."
    except Exception as e:
        return f"⚠️ Connection attempt failed: {str(e)}"

# Gradio Interface
with gr.Blocks(title="CropHarvest AI - Harvest Decision Engine", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    
    # Product header and branding
    gr.HTML('''
    <div class="jira-header">
        <div class="jira-header-title">
            <span class="brand-mark">CH</span>
            <div>
                <div class="brand-title">CropHarvest</div>
                <small>Field decisions, made before the window closes</small>
            </div>
        </div>
        <div class="header-status-badge">
            <span class="status-dot"></span>
            <span>AI Agronomy Engine Active</span>
        </div>
    </div>
    <div class="problem-brief">
        <div class="eyebrow">The Field Challenge</div>
        <h1>Turn crop signals into the next harvest move.</h1>
        <p>Harvest too early and quality is left in the field. Wait too long and weather, labor shortages, and over-ripening erode value. Describe what is happening in the field and CropHarvest will estimate the window, surface risk, and recommend the next operational action.</p>
        <div class="brief-points">
            <span class="brief-point-item">🌿 01 · Readiness Scoring</span>
            <span class="brief-point-item">⚡ 02 · Risk Mitigation</span>
            <span class="brief-point-item">🚜 03 · Crew Logistics</span>
        </div>
    </div>
    ''')
    
    # Modern Navigation Tabs
    with gr.Tabs():
        
        # TAB 1: Harvest Predictor & GDD Trajectory
        with gr.TabItem("🌾 Harvest Decision & GDD Curve"):
            
            # Quick Presets Bar
            with gr.Row(elem_classes=["preset-row"]):
                preset_1 = gr.Button("🍇 Cabernet Grapes", size="sm", elem_classes=["preset-chip"])
                preset_2 = gr.Button("🌽 Sweet Corn", size="sm", elem_classes=["preset-chip"])
                preset_3 = gr.Button("🍅 Processing Tomatoes", size="sm", elem_classes=["preset-chip"])
                preset_4 = gr.Button("🌾 Winter Wheat", size="sm", elem_classes=["preset-chip"])
            
            with gr.Row():
                # Left: Input Form
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("### Build a field brief")
                        gr.Markdown("Give the agent enough context to make a decision your crew can act on today.", elem_classes=["section-helper"])
                        
                        crop_select = gr.Dropdown(
                            label="Crop Variety",
                            choices=list(CROP_PROFILES.keys()),
                            value="Sweet Corn / Maize"
                        )
                        
                        with gr.Row():
                            date_input = gr.Textbox(
                                label="Planting / Sowing Date",
                                value=(datetime.date.today() - datetime.timedelta(days=77)).strftime("%Y-%m-%d")
                            )
                            region_input = gr.Textbox(
                                label="Field Region / Microclimate",
                                value="Midwest (Iowa)"
                            )
                        
                        notes_input = gr.Textbox(
                            label="What is happening in the field?",
                            placeholder="e.g. Silks turning brown, rain expected Friday, buyer needs 72% moisture",
                            value="Silks turning brown, ears filling out with milky kernels, sunny 26°C",
                            lines=3
                        )

                        objective_input = gr.Dropdown(
                            label="What decision do you need to make?",
                            choices=[
                                "Choose the best harvest timing",
                                "Prioritize fields for crew deployment",
                                "Protect quality before incoming weather",
                                "Check whether the crop is ready"
                            ],
                            value="Choose the best harvest timing"
                        )
                        
                        predict_btn = gr.Button("⚡ Run field decision", variant="primary", size="lg", elem_classes=["btn-run-decision"])

                # Right: Output Brief & Interactive Chart
                with gr.Column(scale=7):
                    output_display = gr.HTML(DEFAULT_OUTPUT_HTML)
                    with gr.Group(elem_classes=["plot-card"]):
                        gr.Markdown("#### Growing Degree Days (GDD) Trajectory & Harvest Curve")
                        gdd_plot = gr.Plot(value=DEFAULT_PLOT, show_label=False)

        # TAB 2: Jira-Style Field Operations Kanban Board
        with gr.TabItem("📋 Field Operations Board (Kanban)"):
            gr.Markdown("### Live Field Operations Tracking")
            gr.Markdown("Real-time agronomic progression across monitored acreage blocks. Click any ticket to inspect harvest readiness and resource dispatch status.", elem_classes=["section-helper"])
            kanban_display = gr.HTML(DEFAULT_KANBAN_HTML)
            refresh_kanban_btn = gr.Button("🔄 Refresh Field Board Status", size="sm", elem_classes=["preset-chip"])

        # TAB 3: AI Field Copilot
        with gr.TabItem("💬 AI Field Copilot"):
            gr.Markdown("### Field Copilot Conversational Assistant")
            gr.Markdown("Ask agronomic questions across your active field tickets, microclimate forecasts, or harvesting crew scheduling.", elem_classes=["section-helper"])
            
            # Quick Prompts
            with gr.Row(elem_classes=["prompt-chips-wrap"]):
                prompt_btn_1 = gr.Button("🚨 Which fields need immediate harvester deployment?", size="sm", elem_classes=["prompt-chip-btn"])
                prompt_btn_2 = gr.Button("🌡️ How does upcoming heat affect the Cabernet block?", size="sm", elem_classes=["prompt-chip-btn"])
                prompt_btn_3 = gr.Button("📋 Give me an operational crew checklist for tomorrow", size="sm", elem_classes=["prompt-chip-btn"])

            chatbot = gr.Chatbot(
                value=[(
                    "When is the best time of day to harvest sweet corn?",
                    "**CropHarvest Agent:** For sweet corn, harvest in the early morning before sunrise or at dawn when field heat is lowest. This preserves sugar content (sucrose) and prevents it from rapidly converting to starch."
                )],
                height=320,
                show_label=False
            )
            with gr.Row(elem_classes=["chat-input-row"]):
                chat_input = gr.Textbox(placeholder="Ask about rain risk, crew priority, quality metrics, or timing...", scale=9, show_label=False)
                send_btn = gr.Button("Ask Copilot", variant="primary", scale=2, elem_classes=["btn-run-decision"])

        # TAB 4: Engine Settings & API Keys
        with gr.TabItem("⚙️ Engine & API Settings"):
            with gr.Group():
                gr.Markdown("### AI Inference & Model Configuration")
                gr.Markdown("CropHarvest operates autonomously using its built-in agronomic simulation engine. Optionally connect your Groq API key for ultra-fast LPU inference.", elem_classes=["section-helper"])
                
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Groq API Key",
                        placeholder="gsk_...",
                        value=DEFAULT_GROQ_API_KEY,
                        type="password"
                    )
                    model_select = gr.Dropdown(
                        label="Groq Model",
                        choices=GROQ_MODELS,
                        value=DEFAULT_GROQ_MODEL
                    )
                
                with gr.Row():
                    test_btn = gr.Button("Test API Connection", variant="secondary", size="sm")
                    status_text = gr.Markdown("Ready: Local Agronomy Engine active.")

    # Event Bindings
    predict_btn.click(
        fn=on_predict_harvest,
        inputs=[crop_select, date_input, region_input, notes_input, objective_input],
        outputs=[output_display, gdd_plot]
    )
    
    preset_1.click(
        fn=lambda: on_load_preset("🍇 Cabernet Grapes"),
        inputs=[],
        outputs=[crop_select, date_input, region_input, notes_input, output_display, gdd_plot]
    )
    preset_2.click(
        fn=lambda: on_load_preset("🌽 Sweet Corn"),
        inputs=[],
        outputs=[crop_select, date_input, region_input, notes_input, output_display, gdd_plot]
    )
    preset_3.click(
        fn=lambda: on_load_preset("🍅 Tomatoes"),
        inputs=[],
        outputs=[crop_select, date_input, region_input, notes_input, output_display, gdd_plot]
    )
    preset_4.click(
        fn=lambda: on_load_preset("🌾 Wheat"),
        inputs=[],
        outputs=[crop_select, date_input, region_input, notes_input, output_display, gdd_plot]
    )
    
    # Chat events
    send_btn.click(
        fn=on_ask_agent,
        inputs=[chat_input, chatbot, crop_select, region_input],
        outputs=[chatbot, chat_input]
    )
    chat_input.submit(
        fn=on_ask_agent,
        inputs=[chat_input, chatbot, crop_select, region_input],
        outputs=[chatbot, chat_input]
    )

    # Quick prompt buttons
    prompt_btn_1.click(
        fn=lambda h, c, r: on_ask_agent("Which fields need immediate harvester deployment?", h, c, r),
        inputs=[chatbot, crop_select, region_input],
        outputs=[chatbot, chat_input]
    )
    prompt_btn_2.click(
        fn=lambda h, c, r: on_ask_agent("How does upcoming heat affect the Cabernet block?", h, c, r),
        inputs=[chatbot, crop_select, region_input],
        outputs=[chatbot, chat_input]
    )
    prompt_btn_3.click(
        fn=lambda h, c, r: on_ask_agent("Give me an operational crew checklist for tomorrow morning.", h, c, r),
        inputs=[chatbot, crop_select, region_input],
        outputs=[chatbot, chat_input]
    )

    # Kanban refresh
    refresh_kanban_btn.click(
        fn=lambda: render_kanban_board(INITIAL_TICKETS),
        inputs=[],
        outputs=[kanban_display]
    )

    # Settings test
    test_btn.click(
        fn=test_api_status,
        inputs=[api_key_input, model_select],
        outputs=[status_text]
    )

# Top-level ASGI FastAPI app for Vercel and production deployments
app = FastAPI(title="AgriPulse Harvest OS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app = gr.mount_gradio_app(app, demo, path="/", allowed_paths=[BASE_DIR])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
