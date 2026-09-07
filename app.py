import os
import datetime
from typing import Dict, Any, List
import gradio as gr

from config import (
    CROP_PROFILES,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_GROQ_MODEL,
    GROQ_MODELS
)
from agronomy_engine import simulate_phenology_and_gdd
from groq_agent import analyze_harvest_with_groq, ask_groq_copilot

with open("jira_theme.css", "r") as f:
    CUSTOM_CSS = f.read()

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
    readiness = ai_result.get("readiness_score", 85)
    harvest_date = ai_result.get("predicted_harvest_date", "2026-09-14")
    window_start = ai_result.get("window_start", "2026-09-12")
    window_end = ai_result.get("window_end", "2026-09-16")
    days_rem = ai_result.get("days_remaining", 7)
    stage = ai_result.get("current_stage", "Maturation")
    quality = ai_result.get("predicted_quality", "Optimal ripeness reached.")
    risk = ai_result.get("risk_level", "Low")
    mitigation = ai_result.get("risk_mitigation", "Monitor daily weather.")
    rationale = ai_result.get("ai_rationale", "GDD accumulation aligns with peak quality curve.")
    source = ai_result.get("source", "Groq AI Agent")

    status_pill_bg = "#E3FCEF" if readiness >= 90 else ("#FFF0B3" if readiness >= 75 else "#DEEBFF")
    status_pill_text = "#006644" if readiness >= 90 else ("#172B4D" if readiness >= 75 else "#0747A6")
    status_label = "READY TO HARVEST" if readiness >= 90 else ("APPROACHING MATURITY" if readiness >= 75 else "IN VEGETATIVE GROWTH")

    return f'''
    <div style="background: #FFFFFF; border: 1px solid #DFE1E6; border-radius: 6px; padding: 20px; box-shadow: 0 1px 3px rgba(9,30,66,0.08);">
        
        <!-- Status Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #EBECF0; padding-bottom: 10px;">
            <span style="font-size: 13px; font-weight: 700; color: #0052CC; text-transform: uppercase;">
                AI Agent Verdict • {crop} ({region})
            </span>
            <span style="background: {status_pill_bg}; color: {status_pill_text}; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 12px;">
                ● {status_label}
            </span>
        </div>

        <!-- Target Date Hero Banner -->
        <div style="background: #F4F8FF; border: 1px solid #B3D4FF; border-radius: 6px; padding: 16px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 11px; font-weight: 700; color: #5E6C84; text-transform: uppercase;">Predicted Harvest Date</div>
                <div style="font-size: 26px; font-weight: 700; color: #0052CC; margin-top: 2px;">🎯 {harvest_date}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; font-weight: 700; color: #5E6C84; text-transform: uppercase;">Optimal Harvest Window</div>
                <div style="font-size: 16px; font-weight: 700; color: #006644; margin-top: 2px;">{window_start} ➔ {window_end}</div>
                <div style="font-size: 12px; color: #5E6C84; margin-top: 2px;">⏳ ~{days_rem} days remaining</div>
            </div>
        </div>

        <!-- 3 Simple Metrics -->
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px;">
            <div style="background: #FAFBFC; border: 1px solid #EBECF0; border-radius: 4px; padding: 12px;">
                <div style="font-size: 11px; font-weight: 600; color: #5E6C84; text-transform: uppercase;">Crop Readiness</div>
                <div style="font-size: 20px; font-weight: 700; color: {'#006644' if readiness >= 90 else '#0052CC'};">{readiness}%</div>
                <div style="font-size: 11px; color: #5E6C84;">Stage: {stage}</div>
            </div>
            <div style="background: #FAFBFC; border: 1px solid #EBECF0; border-radius: 4px; padding: 12px;">
                <div style="font-size: 11px; font-weight: 600; color: #5E6C84; text-transform: uppercase;">Thermal Units (GDD)</div>
                <div style="font-size: 18px; font-weight: 700; color: #172B4D;">{ai_result.get('accumulated_gdd', 0)} GDD</div>
                <div style="font-size: 11px; color: #5E6C84;">Target: {ai_result.get('target_gdd', 0)} GDD</div>
            </div>
            <div style="background: #FAFBFC; border: 1px solid #EBECF0; border-radius: 4px; padding: 12px;">
                <div style="font-size: 11px; font-weight: 600; color: #5E6C84; text-transform: uppercase;">Risk Evaluation</div>
                <div style="font-size: 18px; font-weight: 700; color: {'#DE350B' if 'high' in risk.lower() else '#36B37E'};">{risk}</div>
                <div style="font-size: 11px; color: #5E6C84;">Weather Impact</div>
            </div>
        </div>

        <!-- AI Agent Explanation -->
        <div style="background: #FAFBFC; border-left: 4px solid #0052CC; padding: 12px 16px; border-radius: 0 4px 4px 0; margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 700; color: #0747A6; margin-bottom: 4px;">
                🤖 Agent Agronomic Analysis ({source})
            </div>
            <div style="font-size: 13px; color: #172B4D; line-height: 1.45;">
                {rationale}
            </div>
        </div>

        <!-- Actionable Recommendations -->
        <div style="background: #FFFBE6; border: 1px solid #FFE380; border-radius: 4px; padding: 12px 16px;">
            <div style="font-size: 12px; font-weight: 700; color: #172B4D; margin-bottom: 4px;">
                📋 Harvest Execution Recommendations:
            </div>
            <div style="font-size: 12.5px; color: #172B4D; line-height: 1.45;">
                • <b>Quality Target:</b> {quality}<br/>
                • <b>Harvest Action:</b> {mitigation}
            </div>
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

def on_predict_harvest(crop, sowing_date, region, notes, api_key, model):
    result = analyze_harvest_with_groq(
        crop=crop,
        variety="Standard",
        sowing_date=sowing_date,
        location=region,
        current_gdd=0.0,
        recent_weather=notes or "Average seasonal temperatures",
        soil_condition="Well maintained",
        target_quality="",
        api_key=api_key,
        model=model
    )
    return render_agent_output(result, crop, region)

def on_load_preset(preset_key, api_key, model):
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
        api_key=api_key,
        model=model
    )
    output_html = render_agent_output(result, p["crop"], p["region"])
    return p["crop"], p["date"], p["region"], p["notes"], output_html

def on_ask_agent(message, history, crop, region, api_key, model):
    if not message.strip():
        return history, ""
    context_ticket = [{
        "id": "CURRENT-FIELD",
        "summary": f"{crop} in {region}",
        "crop": crop,
        "status": "MATURATION",
        "current_gdd": 1200,
        "target_gdd": 1350,
        "predicted_harvest_date": "Within 7-10 days",
        "risk_level": "Low"
    }]
    answer = ask_groq_copilot(message, context_ticket, history, api_key, model)
    history.append((message, answer))
    return history, ""

# Gradio Interface
with gr.Blocks(title="CropHarvest AI Agent - Harvest Time Prediction", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    
    # Jira Top Header Bar
    gr.HTML('''
    <div class="jira-header">
        <div class="jira-header-title">
            <span>🌾</span>
            <span>CropHarvest AI Agent</span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span class="jira-header-badge">⚡ Groq LPU: Active</span>
            <span style="font-size: 12px; color: #DEEBFF;">Harvest Time Prediction</span>
        </div>
    </div>
    ''')
    
    # Quick 1-Click Presets Bar
    with gr.Row():
        preset_1 = gr.Button("🍇 Cabernet Grapes", size="sm", variant="secondary")
        preset_2 = gr.Button("🌽 Sweet Corn", size="sm", variant="secondary")
        preset_3 = gr.Button("🍅 Tomatoes", size="sm", variant="secondary")
        preset_4 = gr.Button("🌾 Wheat", size="sm", variant="secondary")
    
    # Main Form & Agent Output
    with gr.Row():
        
        # Left: Clean Input Form
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("### 📋 Crop Details")
                
                crop_select = gr.Dropdown(
                    label="Select Crop",
                    choices=list(CROP_PROFILES.keys()),
                    value="Sweet Corn / Maize"
                )
                
                with gr.Row():
                    date_input = gr.Textbox(
                        label="Planting / Sowing Date",
                        value=(datetime.date.today() - datetime.timedelta(days=77)).strftime("%Y-%m-%d")
                    )
                    region_input = gr.Textbox(
                        label="Field Region / Climate",
                        value="Midwest (Iowa)"
                    )
                
                notes_input = gr.Textbox(
                    label="Current Crop Notes / Recent Weather (Optional)",
                    placeholder="e.g. Silks turning brown, sunny weather around 26°C",
                    value="Silks turning brown, ears filling out with milky kernels, sunny 26°C",
                    lines=2
                )
                
                with gr.Accordion("⚙️ Groq API Key (Optional)", open=False):
                    api_key_field = gr.Textbox(
                        label="Groq API Key",
                        placeholder="gsk_... (leave empty to use built-in agronomy simulation)",
                        type="password",
                        value=DEFAULT_GROQ_API_KEY
                    )
                    model_field = gr.Dropdown(
                        label="Model",
                        choices=GROQ_MODELS,
                        value=DEFAULT_GROQ_MODEL
                    )
                
                predict_btn = gr.Button("🚀 Run Harvest Time Prediction Agent", variant="primary", size="lg")

        # Right: Clean AI Agent Output
        with gr.Column(scale=7):
            output_display = gr.HTML(DEFAULT_OUTPUT_HTML)

    # Chat with the Agent
    with gr.Accordion("💬 Ask the AI Agent a Question About Your Harvest", open=True):
        chatbot = gr.Chatbot(
            value=[(
                "When is the best time of day to harvest sweet corn?",
                "**CropHarvest Agent:** For sweet corn, harvest in the early morning before sunrise or at dawn when field heat is lowest. This preserves sugar content (sucrose) and prevents it from rapidly converting to starch."
            )],
            height=200,
            show_label=False
        )
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Ask a question about this crop harvest (e.g. 'Can I harvest early?', 'What if it rains?')...", scale=9, show_label=False)
            send_btn = gr.Button("Ask Agent", variant="primary", scale=2)

    # Event Bindings
    predict_btn.click(
        fn=on_predict_harvest,
        inputs=[crop_select, date_input, region_input, notes_input, api_key_field, model_field],
        outputs=[output_display]
    )
    
    preset_1.click(
        fn=lambda key, mdl: on_load_preset("🍇 Cabernet Grapes", key, mdl),
        inputs=[api_key_field, model_field],
        outputs=[crop_select, date_input, region_input, notes_input, output_display]
    )
    preset_2.click(
        fn=lambda key, mdl: on_load_preset("🌽 Sweet Corn", key, mdl),
        inputs=[api_key_field, model_field],
        outputs=[crop_select, date_input, region_input, notes_input, output_display]
    )
    preset_3.click(
        fn=lambda key, mdl: on_load_preset("🍅 Tomatoes", key, mdl),
        inputs=[api_key_field, model_field],
        outputs=[crop_select, date_input, region_input, notes_input, output_display]
    )
    preset_4.click(
        fn=lambda key, mdl: on_load_preset("🌾 Wheat", key, mdl),
        inputs=[api_key_field, model_field],
        outputs=[crop_select, date_input, region_input, notes_input, output_display]
    )
    
    send_btn.click(
        fn=on_ask_agent,
        inputs=[chat_input, chatbot, crop_select, region_input, api_key_field, model_field],
        outputs=[chatbot, chat_input]
    )
    chat_input.submit(
        fn=on_ask_agent,
        inputs=[chat_input, chatbot, crop_select, region_input, api_key_field, model_field],
        outputs=[chatbot, chat_input]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
