import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration (Ultra-fast LPU inference)
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# xAI Grok (Alternative)
DEFAULT_GROK_API_KEY = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "")
GROK_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
DEFAULT_GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")

# Crop Agronomic Database (GDD Base, Target GDD, Target Quality Metrics)
CROP_PROFILES: Dict[str, Dict[str, Any]] = {
    "Wine Grapes (Cabernet/Pinot)": {
        "base_temp_c": 10.0,
        "upper_temp_c": 35.0,
        "target_gdd": 1450,
        "growing_days_typical": 165,
        "target_metric_name": "Sugar Content (Brix)",
        "target_metric_ideal": "23.5 - 25.5 °Bx",
        "target_moisture_or_brix": 24.5,
        "stages": ["Budbreak", "Bloom", "Veraison (Color Change)", "Phenolic Ripening", "Optimal Harvest"],
        "critical_risks": ["Botrytis Bunch Rot", "Late Frost", "Heatwave Desiccation", "Sugar Spike vs Acid Drop"]
    },
    "Sweet Corn / Maize": {
        "base_temp_c": 10.0,
        "upper_temp_c": 30.0,
        "target_gdd": 1300,
        "growing_days_typical": 85,
        "target_metric_name": "Kernel Moisture %",
        "target_metric_ideal": "70% - 74% (Fresh) / 15-18% (Grain)",
        "target_moisture_or_brix": 72.0,
        "stages": ["Emergence", "V6 Vegetative", "Tasseling (VT)", "Silking (R1)", "Milk/Dough (R3-R4)", "Dent/Physiological Maturity"],
        "critical_risks": ["Corn Earworm", "Drought Stress during Silking", "Late Hail Damage", "Early Fall Frost"]
    },
    "Winter / Spring Wheat": {
        "base_temp_c": 0.0,
        "upper_temp_c": 25.0,
        "target_gdd": 1850,
        "growing_days_typical": 120,
        "target_metric_name": "Grain Moisture %",
        "target_metric_ideal": "13.0% - 14.5%",
        "target_moisture_or_brix": 13.5,
        "stages": ["Germination", "Tillering", "Stem Elongation (Jointing)", "Booting & Heading", "Grain Fill", "Hard Dough (Maturity)"],
        "critical_risks": ["Fusarium Head Blight", "Pre-harvest Sprouting", "Lodging from Heavy Wind", "Rust Infestation"]
    },
    "Processing Tomato": {
        "base_temp_c": 10.0,
        "upper_temp_c": 32.0,
        "target_gdd": 1600,
        "growing_days_typical": 110,
        "target_metric_name": "Brix & Color Index (a/b)",
        "target_metric_ideal": "5.2 - 5.8 °Bx / 90%+ Red",
        "target_moisture_or_brix": 5.5,
        "stages": ["Transplant", "Vegetative Growth", "First Flower Cluster", "Fruit Set", "Breaker / Color Break", "Full Field Red"],
        "critical_risks": ["Early/Late Blight", "Sunscald", "Fruit Split from Heavy Rain", "Blossom End Rot"]
    },
    "Soybeans": {
        "base_temp_c": 10.0,
        "upper_temp_c": 30.0,
        "target_gdd": 1400,
        "growing_days_typical": 125,
        "target_metric_name": "Seed Moisture %",
        "target_metric_ideal": "13.0% - 14.0%",
        "target_moisture_or_brix": 13.0,
        "stages": ["VE Emergence", "V4 Flowering Initiation", "R1 Beginning Bloom", "R5 Seed Fill", "R7 Beginning Maturity", "R8 Full Maturity"],
        "critical_risks": ["Soybean Rust", "Pod Shatter in Low Humidity", "Sudden Death Syndrome", "Frost before R7"]
    },
    "Apples (Honeycrisp/Gala)": {
        "base_temp_c": 4.0,
        "upper_temp_c": 28.0,
        "target_gdd": 1550,
        "growing_days_typical": 140,
        "target_metric_name": "Starch-Iodine Index & Firmness",
        "target_metric_ideal": "Starch 5-6 / 15-17 lbs Firmness",
        "target_moisture_or_brix": 14.0,
        "stages": ["Dormancy/Silver Tip", "Pink Bud", "Full Bloom", "Fruitlet Cell Division", "Color Flush & Starch Conversion", "Maturity"],
        "critical_risks": ["Apple Scab", "Sunburn & Bitter Pit", "Pre-harvest Fruit Drop", "Freezing Temperatures"]
    },
    "Paddy Rice": {
        "base_temp_c": 10.0,
        "upper_temp_c": 32.0,
        "target_gdd": 1750,
        "growing_days_typical": 130,
        "target_metric_name": "Paddy Grain Moisture %",
        "target_metric_ideal": "20.0% - 22.0% (Field) / 14% (Stored)",
        "target_moisture_or_brix": 21.0,
        "stages": ["Nursery / Transplant", "Tillering", "Panicle Initiation", "Heading & Anthesis", "Milk/Dough Stage", "Golden Ripe"],
        "critical_risks": ["Rice Blast", "Brown Planthopper", "Lodging in Monsoons", "Grain Cracking from Rapid Drying"]
    },
    "Strawberries": {
        "base_temp_c": 3.0,
        "upper_temp_c": 26.0,
        "target_gdd": 750,
        "growing_days_typical": 45,
        "target_metric_name": "Sugar/Acid Ratio (Brix)",
        "target_metric_ideal": "8.5 - 10.5 °Bx / 95% Red Surface",
        "target_moisture_or_brix": 9.2,
        "stages": ["Crown Growth", "Bloom Clusters", "Green Berry", "White/Pink Berry", "Full Red Maturation"],
        "critical_risks": ["Grey Mould (Botrytis)", "Thrips", "Rain Damage / Softening", "Over-ripening in Heat"]
    },
    "Potatoes (Russet/Table)": {
        "base_temp_c": 7.0,
        "upper_temp_c": 28.0,
        "target_gdd": 1500,
        "growing_days_typical": 115,
        "target_metric_name": "Specific Gravity & Skin Set",
        "target_metric_ideal": "Specific Gravity 1.085+ / 95% Skin Set",
        "target_moisture_or_brix": 18.0,
        "stages": ["Sprout Development", "Vegetative Growth", "Tuber Initiation", "Tuber Bulking", "Vine Senescence", "Skin Set"],
        "critical_risks": ["Late Blight", "Hollow Heart", "Bruising in Cold Soils", "Wet Rot during Rain"]
    }
}

# Jira Board Columns & Status Definitions
JIRA_STATUSES = {
    "VEGETATIVE": {
        "label": "🌱 Vegetative / Active Growth",
        "badge_class": "jira-badge-inprogress",
        "desc": "Crop is accumulating biomass and heat units."
    },
    "RIPENING": {
        "label": "🌾 Maturation & Ripening",
        "badge_class": "jira-badge-review",
        "desc": "Crop is in final phase (veraison, grain fill, color break)."
    },
    "OPTIMAL_HARVEST": {
        "label": "🎯 Optimal Harvest Window",
        "badge_class": "jira-badge-ready",
        "desc": "Peak quality and yield reached. Ideal window for harvesting."
    },
    "ARCHIVED": {
        "label": "✅ Harvested & Archived",
        "badge_class": "jira-badge-done",
        "desc": "Harvest operation concluded and yield logged."
    }
}

# Initial Default Jira Tickets (Real-world style farm fields)
INITIAL_TICKETS: List[Dict[str, Any]] = [
    {
        "id": "AGRI-101",
        "summary": "Sonoma Block 4 - Cabernet Sauvignon Harvest",
        "crop": "Wine Grapes (Cabernet/Pinot)",
        "variety": "Cabernet Sauvignon Clone 337",
        "location": "Sonoma Valley, CA (38.3°N, 122.5°W)",
        "sowing_or_bloom_date": "2026-04-12",
        "days_elapsed": 144,
        "status": "OPTIMAL_HARVEST",
        "priority": "HIGH",
        "story_points": 45,  # acres
        "assignee": "Alex Mercer (Senior Viticulturist)",
        "current_gdd": 1410,
        "target_gdd": 1450,
        "predicted_harvest_date": "2026-09-08",
        "window_start": "2026-09-06",
        "window_end": "2026-09-11",
        "readiness_score": 96,
        "predicted_quality": "24.6 °Bx, pH 3.58, optimal anthocyanin extraction",
        "risk_level": "Moderate (Late week heat spike 36°C)",
        "risk_mitigation": "Initiate night mechanical harvesting on Sep 7 to preserve acidity and prevent phenolic over-extraction.",
        "ai_rationale": "Accumulated 1410 GDD against 1450 threshold. Daily GDD accumulation is 12.5 GDD/day. Veraison occurred 42 days ago. Sugar accumulation curve matches vintage 2019 profile."
    },
    {
        "id": "AGRI-102",
        "summary": "Midwest Field 12 - Hybrid Sweet Corn Batch A",
        "crop": "Sweet Corn / Maize",
        "variety": "Pioneer Hi-Bred 1197",
        "location": "Story County, Iowa (42.0°N, 93.6°W)",
        "sowing_or_bloom_date": "2026-06-18",
        "days_elapsed": 77,
        "status": "RIPENING",
        "priority": "MEDIUM",
        "story_points": 120,
        "assignee": "Sarah Jenkins (Agronomy Lead)",
        "current_gdd": 1190,
        "target_gdd": 1300,
        "predicted_harvest_date": "2026-09-14",
        "window_start": "2026-09-12",
        "window_end": "2026-09-17",
        "readiness_score": 88,
        "predicted_quality": "Kernel moisture 72.8%, sugar/starch ratio 4.2",
        "risk_level": "Low (Favorable clear weather)",
        "risk_mitigation": "Mobilize combine harvesters on standby for Sep 13 morning.",
        "ai_rationale": "Corn is currently at late R3 (milk stage transition to early dough). 110 GDD remaining until optimal 72% kernel moisture. Favorable diurnal temp swing will optimize kernel sweetness."
    },
    {
        "id": "AGRI-103",
        "summary": "Central Valley Sector 9 - Roma Processing Tomatoes",
        "crop": "Processing Tomato",
        "variety": "Heinz 9665 Processing",
        "location": "Fresno, California (36.7°N, 119.8°W)",
        "sowing_or_bloom_date": "2026-05-20",
        "days_elapsed": 106,
        "status": "OPTIMAL_HARVEST",
        "priority": "URGENT",
        "story_points": 80,
        "assignee": "Carlos Ramirez (Field Operations)",
        "current_gdd": 1585,
        "target_gdd": 1600,
        "predicted_harvest_date": "2026-09-05",
        "window_start": "2026-09-04",
        "window_end": "2026-09-07",
        "readiness_score": 98,
        "predicted_quality": "5.6 °Bx, 94% fruit color uniformity, firmness 88/100",
        "risk_level": "High (Sunscald risk if delayed past Sep 7)",
        "risk_mitigation": "Immediate priority dispatch for harvester team Alpha.",
        "ai_rationale": "GDD completion at 99.1%. Color break is at 94% full red. Further delay will degrade viscosity and increase sunburn percentage."
    },
    {
        "id": "AGRI-104",
        "summary": "Willamette Orchard 3 - Honeycrisp Block",
        "crop": "Apples (Honeycrisp/Gala)",
        "variety": "Royal Honeycrisp",
        "location": "Hood River, Oregon (45.7°N, 121.5°W)",
        "sowing_or_bloom_date": "2026-05-01",
        "days_elapsed": 125,
        "status": "VEGETATIVE",
        "priority": "LOW",
        "story_points": 30,
        "assignee": "Emily Thorne (Orchard Specialist)",
        "current_gdd": 1280,
        "target_gdd": 1550,
        "predicted_harvest_date": "2026-09-24",
        "window_start": "2026-09-22",
        "window_end": "2026-09-28",
        "readiness_score": 74,
        "predicted_quality": "Starch conversion 3.2/8, background ground color still chartreuse",
        "risk_level": "Low (Standard maturation progression)",
        "risk_mitigation": "Apply calcium foliar spray to strengthen cell walls against bitter pit.",
        "ai_rationale": "Orchard requires ~270 additional GDD. Crispness index progressing on target with cool night temperatures encouraging anthocyanin red striping."
    }
]
