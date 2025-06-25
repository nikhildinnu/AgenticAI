import gradio as gr
import json
import re
import requests
import feedparser
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- Existing Travel Functions ---

# LLM model for attractions, restaurants, transportation
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

travel_prompt = PromptTemplate.from_template("""
You are a travel assistant. Given the following city name, list the top 3 tourist attractions, activity, adventure sports and a short description of each.
For each of the following tourist landmarks in {city}, list 2 top-rated restaurants (preferably 5-star Google-rated or highly rated) nearby. For each restaurant, include the name, cuisine type, and a one-line description.
For each of the following tourist landmarks in {city}, suggest transportation and estimated cast for public transsport and cab's cast.

City: {city}
""")

def travel_guide(city):
    prompt_text = travel_prompt.format(city=city)
    full_response = model.invoke(prompt_text).content
    attraction_pattern = re.findall(r"(?:1\\.|2\\.|3\\.|\\-)\\s*([A-Z][\\w\\s,'&\\-]+)", full_response)
    attractions = list(dict.fromkeys([a.strip() for a in attraction_pattern if len(a.strip()) > 3]))
    return {
        "city": city,
        "attractions": attractions[:3],
        "full_guide": full_response
    }

def calamity_forecast(city, start_date, end_date):
    geo_url = "https://nominatim.openstreetmap.org/search"
    geo_params = {"q": city, "format": "json"}
    headers = {"User-Agent": "CalamityForecast/1.0 (your@email.com)"}
    geo_response = requests.get(geo_url, params=geo_params, headers=headers)
    try:
        geo_data = geo_response.json()
    except Exception:
        return {"error": "Failed to parse geolocation response. Possibly blocked or invalid."}
    if not geo_data:
        return {"error": f"Location '{city}' not found."}
    lat = float(geo_data[0]["lat"])
    lon = float(geo_data[0]["lon"])

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }
    weather_response = requests.get(weather_url, params=weather_params)
    try:
        weather_data = weather_response.json().get("daily", {})
    except Exception:
        weather_data = "Weather API failed or returned invalid data"

    usgs_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    usgs_params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": 4.5
    }
    usgs_response = requests.get(usgs_url, params=usgs_params)
    earthquakes = []
    try:
        usgs_data = usgs_response.json()
        for eq in usgs_data.get("features", []):
            place_info = eq["properties"]["place"]
            if city.lower() in place_info.lower():
                earthquakes.append({
                    "place": place_info,
                    "magnitude": eq["properties"]["mag"],
                    "time": datetime.utcfromtimestamp(eq["properties"]["time"] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "url": eq["properties"]["url"]
                })
    except Exception:
        earthquakes = "USGS data unavailable or failed to decode"

    try:
        gdacs_feed = feedparser.parse("https://www.gdacs.org/xml/rss.xml")
        gdacs_alerts = []
        for entry in gdacs_feed.entries:
            if city.lower() in entry.title.lower() or city.lower() in entry.summary.lower():
                gdacs_alerts.append({
                    "event": entry.title,
                    "details": entry.summary,
                    "link": entry.link
                })
        if not gdacs_alerts:
            gdacs_alerts = "No GDACS disaster alerts found."
    except Exception:
        gdacs_alerts = "Failed to fetch GDACS alerts"

    return (city, lat, lon, weather_data, gdacs_alerts)

# Hotel Recommender
prompt = PromptTemplate.from_template("""
You are a travel assistant. Suggest 5 top-rated hotels in {city}.

- 2 should be premium luxury hotels (7-star or 5-star),
- 2 should be mid-range 3-star hotels (with swimming pool, restaurants, etc.),
- 1 should be a budget economy hotel.

For each hotel, provide:
- Hotel Name
- Location (area or landmark)
- Cost per night (local currency)

Format the output as a JSON list like this:
[
  {{
    "name": "Hotel Name",
    "location": "Area, City",
    "cost_per_night": "₹XXXX or $XXX"
  }},
  ...
]
""")

def hotel_recommender(city: str):
    input_text = prompt.format(city=city)
    response = model.invoke(input_text).content
    return response

# Itinerary Generator
itinerary_model = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.7
)

itinerary_prompt = PromptTemplate.from_template("""
You are a professional travel planner. Create a detailed itinerary for a trip to {city} for {days} days.

Must include:
- Top tourist attractions: {attractions}
- Adventure sports or activities: {adventure_sports}
- Suggested schedule for each day (morning, afternoon, evening)
- Include meal/restaurant recommendations, travel tips, and rest time
- Focus on realistic travel distances and pacing
- Mention estimated cost per day (budget-friendly range)

Format the output clearly per day.

Example:
Day 1:
- Morning: ...
- Afternoon: ...
- Evening: ...
- Estimated cost: ₹XXXX
""")

def generate_itinerary(city: str, days: int, attractions: list[str], adventure_sports: list[str]):
    attractions_str = ", ".join(attractions)
    adventure_str = ", ".join(adventure_sports)
    input_prompt = itinerary_prompt.format(
        city=city,
        days=days,
        attractions=attractions_str,
        adventure_sports=adventure_str
    )
    itinerary = itinerary_model.invoke(input_prompt).content
    return itinerary

# --- Total Cost Calculation ---
def calculate_total_cost(itinerary_text):
    cost_lines = [line for line in itinerary_text.splitlines() if "Estimated cost" in line]
    total_cost = 0
    for line in cost_lines:
        match = re.search(r"[\u20B9\$](\d+)", line)
        if match:
            total_cost += int(match.group(1))
    return total_cost

# --- Trip Summary Generation ---
def generate_summary(city, days, attractions, hotel_info, total_cost):
    try:
        hotel_info = hotel_info.strip()
        if not hotel_info.startswith('['):
            raise ValueError("Hotel info is not in JSON format")
        hotel_data = json.loads(hotel_info)
        top_hotel = hotel_data[0]['name'] if hotel_data else "N/A"
    except Exception:
        top_hotel = "Hotel data unavailable"

    summary = f"""
Trip Summary:
City: {city}
Duration: {days} days
Top Attractions: {', '.join(attractions)}
Recommended Hotel: {top_hotel}
Estimated Total Trip Cost: ₹{total_cost}
"""
    return summary.strip()

# --- Main Function for Integration ---
def main_travel_planner(city, days):
    guide = travel_guide(city)
    attractions = guide['attractions']
    adventure_sports = ["Paragliding", "Jet Skiing"]
    itinerary = generate_itinerary(city, days, attractions, adventure_sports)
    hotel_data = hotel_recommender(city)
    total_cost = calculate_total_cost(itinerary)
    summary = generate_summary(city, days, attractions, hotel_data, total_cost)
    return guide['full_guide'], itinerary, hotel_data, f"₹{total_cost}", summary

# --- Gradio UI ---
def gradio_interface(city, days):
    return main_travel_planner(city, int(days))

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="City Name"),
        gr.Number(label="Number of Days", precision=0)
    ],
    outputs=[
        gr.Textbox(label="Travel Guide"),
        gr.Textbox(label="Itinerary"),
        gr.Textbox(label="Hotel Recommendations"),
        gr.Textbox(label="Estimated Total Cost"),
        gr.Textbox(label="Trip Summary")
    ],
    title="AI Travel Planner",
    description="Plan your trip with hotel, itinerary, attractions, activities and costs."
)

if __name__ == "__main__":
    iface.launch()
