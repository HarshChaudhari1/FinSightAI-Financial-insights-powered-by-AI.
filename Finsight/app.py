import os
import faiss
import pickle
import joblib
import pandas as pd
import numpy as np
import warnings
import re
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import google.generativeai as genai

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="A date index has been provided")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX


import time 

class FinancialDataFetcher:
    """Handles fetching live financial data and historical prices with retry logic."""
    def _fetch_with_retry(self, fetch_func, max_retries=3, delay=1):
        """Wrapper to retry a function call on failure."""
        for attempt in range(max_retries):
            try:
                return fetch_func()
            except Exception as e:
                print(f"    > WARN: Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    print(f"    > ERROR: All {max_retries} attempts failed.")
                    return None 

    def get_company_info(self, ticker_symbol):
        print(f"    > Fetching financial info for {ticker_symbol}")
        
        def fetch():
            ticker = yf.Ticker(ticker_symbol)
            return ticker.info

        info = self._fetch_with_retry(fetch)
        
        if not info: return {}

        return {
            "Market Cap": f"₹{info.get('marketCap', 0) / 1e7:.2f} Cr" if info.get('marketCap') else "N/A",
            "P/E Ratio": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A",
            "EPS": f"₹{info.get('trailingEps', 0):.2f}" if info.get('trailingEps') else "N/A",
        }

    def get_price_history(self, ticker, period="1y"):
        print(f"    > Fetching historical price data for {ticker}")

        def fetch():
            return yf.Ticker(ticker).history(period=period)

        history = self._fetch_with_retry(fetch)
        
        if history is None:
            return pd.DataFrame() 
        return history
    
class WebSearchEngine:
    """Performs targeted web searches using the Tavily API."""
    def __init__(self, api_key):
        if not api_key or "YOUR_API_KEY" in api_key: self.client = None
        else: self.client = TavilyClient(api_key=api_key)

    def search(self, queries: list, max_results=3):
        if not self.client: return []
        print(f"    > Performing web search for: {queries}")
        content = []
        for query in queries:
            try:
                response = self.client.search(query=query, search_depth="advanced", max_results=max_results)
                for r in response.get('results', []):
                    content.append({"title": r.get('title'), "url": r.get('url'), "content": r.get('content')})
            except Exception as e:
                print(f"    > ERROR during Tavily search for query '{query}': {e}")
        return content

class FactRetrievalEngine:
    """Handles loading and querying the local FAISS vector database."""
    def __init__(self, index_path, documents_path):
        print(f"    > Initializing Fact Retrieval Engine from: {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            with open(documents_path, 'rb') as f: self.documents = pickle.load(f)
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=None)
        except Exception as e:
            print(f"    > ERROR loading vector DB assets: {e}")
            self.index = None

    def retrieve(self, query, top_k=2):
        if not self.index: return []
        print(f"    > Searching Vector DB for facts related to: '{query}'")
        try:
            query_vector = self.model.encode([query])
            _, indices = self.index.search(query_vector.astype('float32'), top_k)
            return [self.documents[i] for i in indices[0]]
        except Exception as e:
            print(f"    > ERROR during vector search: {e}")
            return []

class HybridForecastingEngine:
    """PRODUCTION-READY: Intelligently selects models and forecasts from TODAY's date."""
    def __init__(self, gru_path, sarima_path, scaler_path, master_dataset_path, ticker, bias_path):
        print("     > Initializing Production-Ready Forecasting Engine.")
        self.master_dataset_path = master_dataset_path
        self.ticker_symbol = ticker
        self.data_fetcher = FinancialDataFetcher()
        self.bias = 0.0

        try: self.gru_model = load_model(gru_path)
        except Exception as e: self.gru_model = None; print(f"   > WARN: GRU model not loaded: {e}")
        try: self.original_sarima = joblib.load(sarima_path)
        except Exception as e: self.original_sarima = None; print(f"    > WARN: SARIMA model not loaded: {e}")
        try: self.scaler = joblib.load(scaler_path)
        except Exception as e: self.scaler = None; print(f"     > WARN: Scaler not loaded: {e}")

        try:
            with open(bias_path, 'r') as f:
                self.bias = json.load(f).get('bias', 0.0)
                print(f"    > Loaded prediction bias of: {self.bias:.2f}")
        except Exception as e:
            print(f"    > WARN: Bias file not loaded. Using 0.0 bias. Error: {e}")

    def _get_live_anchored_data(self):
        print("      > Anchoring data to today's date with timezone and frequency sync...")
        hist_df = pd.read_csv(self.master_dataset_path, index_col='date', parse_dates=True)
        live_df = self.data_fetcher.get_price_history(self.ticker_symbol, period="3mo")

        if live_df.empty:
            if hist_df.index.tz is None: hist_df = hist_df.tz_localize('UTC')
            return hist_df.asfreq('B').fillna(method='ffill')

        live_df.columns = [col.lower() for col in live_df.columns]
        if hist_df.index.tz is None: hist_df = hist_df.tz_localize(live_df.index.tz)

        combined_df = pd.concat([hist_df, live_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        return combined_df.asfreq('B').fillna(method='ffill')

    def get_forecast(self, horizon_days: int):
        if horizon_days <= 60: return self._get_gru_forecast(horizon_days)
        else: return self._get_sarima_forecast(horizon_days)

    def _get_gru_forecast(self, horizon_days, lookback=60):
        if not all([self.gru_model, self.scaler]): return {"model": "GRU", "projection": 0, "series": None}
        print(f"    > Generating {horizon_days}-day SHORT-TERM forecast with GRU from LIVE data.")
        try:
            original_features_df = pd.read_csv(self.master_dataset_path)
            original_numeric_cols = original_features_df.select_dtypes(include=np.number).columns.tolist()
            combined_data = self._get_live_anchored_data()
            for col in original_numeric_cols:
                if col not in combined_data.columns: combined_data[col] = 0
            features_df = combined_data[original_numeric_cols].astype('float64')
            if len(features_df) < lookback: return {"model": "GRU", "projection": 0, "series": None}
            recent_data = features_df.values[-lookback:]
            future_predictions = []
            for _ in range(horizon_days):
                scaled_window = self.scaler.transform(recent_data)
                X_test = np.array([scaled_window])
                pred_scaled = self.gru_model.predict(X_test, verbose=0)
                dummy_pred_array = np.zeros((1, len(original_numeric_cols)))
                close_price_index = original_numeric_cols.index('close')
                dummy_pred_array[0, close_price_index] = pred_scaled[0, 0]
                predicted_close_price = self.scaler.inverse_transform(dummy_pred_array)[0, close_price_index]
                future_predictions.append(predicted_close_price)
                next_input_row = recent_data[-1, :].copy()
                next_input_row[close_price_index] = predicted_close_price
                recent_data = np.vstack([recent_data[1:], next_input_row])

            final_price = future_predictions[-1]
            corrected_price = final_price + self.bias
            last_date = combined_data.index[-1]
            future_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, horizon_days + 1)])
            forecast_series = pd.Series([p + self.bias for p in future_predictions], index=future_dates)
            print(f"    > Raw Projection: {final_price:.2f}, Bias Corrected Projection: {corrected_price:.2f}")
            return {"model": "GRU (Bias Corrected)", "projection": corrected_price, "series": forecast_series}
        except Exception as e:
            traceback.print_exc(); print(f"    > ERROR during GRU forecast: {e}")
            return {"model": "GRU", "projection": 0, "series": None}

    def _get_sarima_forecast(self, horizon_days):
        if not self.original_sarima: return {"model": "SARIMA", "projection": 0, "series": None}
        print(f"    > Generating {horizon_days}-day LONG-TERM forecast with SARIMA from LIVE data.")
        try:
            combined_data = self._get_live_anchored_data()
            print("     > Re-fitting SARIMA model on up-to-date data...")
            refit_model = SARIMAX(combined_data['close'], order=self.original_sarima.model.order, seasonal_order=self.original_sarima.model.seasonal_order).fit(disp=False)
            forecast = refit_model.get_forecast(steps=horizon_days)
            final_price = forecast.predicted_mean.iloc[-1]
            return {"model": "SARIMA", "projection": final_price, "series": forecast.predicted_mean}
        except Exception as e:
            print(f"    > ERROR during SARIMA forecast: {e}")
            return {"model": "SARIMA", "projection": 0, "series": None}

class FinsightAgent:
    """AI agent"""
    def __init__(self, google_api_keys, tavily_api_key):
        print("--- Initializing Finsight Agent")
        if not google_api_keys:
            raise ValueError("Google API keys list cannot be empty.")
        self.google_api_keys = google_api_keys
        self.current_key_index = 0
        self.web_search_engine = WebSearchEngine(tavily_api_key)
        self.data_fetcher = FinancialDataFetcher()
        self.loaded_assets = {}

    def _execute_llm_call_with_rotation(self, prompt):
        """Attempts an LLM call, rotating API keys on failure."""
        for i in range(len(self.google_api_keys)):
            key_index_to_try = (self.current_key_index + i) % len(self.google_api_keys)
            current_key = self.google_api_keys[key_index_to_try]
            try:
                print(f"     > Attempting LLM call with API Key index: {key_index_to_try}")
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel('gemini-1.5-pro')
                response = model.generate_content(prompt)
                self.current_key_index = key_index_to_try
                return response
            except Exception as e:
                error_str = str(e).lower()
                if 'permissiondenied' in error_str or 'api_key_invalid' in error_str or 'api key not valid' in error_str:
                    print(f"     > WARN: API Key index {key_index_to_try} failed or is invalid. Trying next...")
                    continue
                else:
                    print(f"     > ERROR: Non-API key error during LLM call: {e}")
                    raise e
        raise Exception("All Google API keys failed. Please check your keys and quotas.")

    def _load_company_assets(self, company_name, asset_paths):
        if company_name in self.loaded_assets: return self.loaded_assets[company_name]
        print(f"--- Loading assets for {company_name} ---")
        assets = {
            "forecasting_engine": HybridForecastingEngine(
                asset_paths["gru_path"], asset_paths["sarima_path"],
                asset_paths["scaler_path"], asset_paths["master_dataset_path"],
                asset_paths["ticker"], asset_paths["bias_path"]
            ),
            "retrieval_engine": FactRetrievalEngine(
                asset_paths["faiss_index_path"], asset_paths["faiss_docs_path"]
            )
        }
        self.loaded_assets[company_name] = assets
        return assets

    def _get_structured_plan(self, query: str):
        print("     > Agent creating strategic, multi-horizon analysis plan...")
        prompt = f"""
        You are a planning agent. Your task is to analyze a user's query and create a multi-horizon forecasting plan.
        Your output MUST be a JSON object with one key: "time_horizons_days".

        **Your Logic:**
        1.  First, look for explicit timeframes mentioned by the user (e.g., "10 days", "6 months"). Convert them to days and add them to the list.
        2.  **If the user asks a strategic, open-ended question** like "what should I do?", "when should I sell?", or "is it a good investment?" and **does not specify a timeframe**, you must generate a default set of strategic horizons.
        3.  The default strategic horizons are: **[30, 90, 180]**.
        4.  If no future timeframe is mentioned and it's a simple factual query (e.g., "what is the news?"), return an empty list: [].

        **User Query:** "{query}"
        """
        try:
            response = self._execute_llm_call_with_rotation(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not match: raise json.JSONDecodeError("No JSON found", response.text, 0)
            plan = json.loads(match.group(0))
            horizons = plan.get('time_horizons_days', [])
            if not isinstance(horizons, list): horizons = [] if not horizons else [horizons]
            plan['time_horizons_days'] = sorted([int(h) for h in horizons if isinstance(h, (int, float))])
            return plan
        except Exception as e:
            print(f"    > ERROR parsing plan. Defaulting to empty list. Error: {e}")
            return {"time_horizons_days": []}

    def _synthesize_analysis(self, company_name, query, context):
        print("     > Agent synthesizing final strategic analysis...")
        
        # --- MODIFIED PROMPT ---
        prompt = f"""
        **Persona:** You are "Finsight," an elite AI Financial Analyst. Your analysis must be objective, data-driven, and detailed.

        **Your Thought Process (Chain of Thought):**
        1.  **Initial Assessment:** Start by looking at the `current_stock_price`.
        2.  **Quantitative Outlook:** Analyze the `quantitative_forecasts`. Do they predict an increase or decrease? Note the different time horizons (e.g., short-term GRU vs. long-term SARIMA). Are there any conflicts?
        3.  **Qualitative Context:** Read through the `live_web_search_results` and `internal_database_facts` (from annual reports). What is the story behind the numbers? Are there positive news catalysts, or negative headwinds?
        4.  **Synthesize and Justify:** Connect the qualitative story (from news and reports) to the quantitative data (from forecasts).
        5.  **Formulate Recommendation:** Based on this complete synthesis, decide on a clear `recommendation`. It must be one of: "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell".
        6.  **Summarize and Detail Drivers:** Write a comprehensive `analysis_summary` that explains your reasoning step-by-step. Extract the most critical points into a list of `key_drivers`.

        **Provided Context:**
        {json.dumps(context, indent=2)}

        **Required Output:** Respond ONLY with a valid JSON object inside a JSON markdown block.
        ```json
        {{
          "recommendation": "...",
          "analysis_summary": "...",
          "key_drivers": ["..."]
        }}
        ```
        """
        try:
            response = self._execute_llm_call_with_rotation(prompt)
            match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if not match:
                   raise json.JSONDecodeError("No JSON found in response", response.text, 0)
            
            return json.loads(match.group(1) if len(match.groups()) > 0 else match.group(0))

        except Exception as e:
            print(f"    > ERROR during final synthesis: {e}")
            return {"recommendation": "Error", "analysis_summary": "Failed to generate a valid analysis.", "key_drivers": []}

    def run(self, company_name: str, query: str, asset_paths: dict):
        plan = self._get_structured_plan(query)
        assets = self._load_company_assets(company_name, asset_paths)
        ticker = asset_paths.get("ticker")
        live_price_history = self.data_fetcher.get_price_history(ticker, period="1y")
        current_price = live_price_history['Close'].iloc[-1] if not live_price_history.empty else 0
        all_forecasts = []
        forecast_chart_data = []
        if plan['time_horizons_days']:
            for h in plan['time_horizons_days']:
                forecast = assets["forecasting_engine"].get_forecast(h)
                forecast['horizon'] = h
                all_forecasts.append(forecast)
                if forecast.get("series") is not None:
                    series = forecast["series"]
                    chart_item = {"horizon": h, "model": forecast.get("model"), "labels": series.index.strftime('%Y-%m-%d').tolist(), "data": series.values.tolist()}
                    forecast_chart_data.append(chart_item)
        
        web_results = self.web_search_engine.search([f'"{company_name}" ({ticker}) stock news {query}'])
        raw_db_facts = assets["retrieval_engine"].retrieve(query)
        db_facts = []
        for fact in raw_db_facts:
            content_str = ""
            if isinstance(fact, str):
                content_str = fact
            elif hasattr(fact, 'page_content'): 
                content_str = fact.page_content
            elif isinstance(fact, dict):
                content_str = fact.get('content', fact.get('text', str(fact)))
            else:
                content_str = str(fact)
            
            db_facts.append({"source_file": "From Vector DB", "content": content_str})
        serializable_forecasts = []
        for f in all_forecasts:
            forecast_copy = f.copy()
            if "series" in forecast_copy and forecast_copy["series"] is not None:
                series = forecast_copy["series"]
                forecast_copy["series_summary"] = {"start_date": series.index[0].strftime('%Y-%m-%d'), "end_date": series.index[-1].strftime('%Y-%m-%d'), "start_value": f"{series.iloc[0]:.2f}", "end_value": f"{series.iloc[-1]:.2f}"}
                del forecast_copy["series"]
            serializable_forecasts.append(forecast_copy)
        
        context = {"current_stock_price": f"₹{current_price:.2f}", "analysis_date": datetime.now().strftime("%Y-%m-%d"), "quantitative_forecasts": serializable_forecasts, "internal_database_facts": db_facts, "live_web_search_results": [item['content'] for item in web_results[:4]]}
        final_analysis_json = self._synthesize_analysis(company_name, query, context)
        
        forecast_display_list = []
        for f in all_forecasts:
            proj = f.get('projection', 0); horizon = f.get('horizon', 'N/A'); model = f.get('model', 'N/A')
            forecast_display_list.append(f"({model}) ~₹{proj:.2f} in {horizon} days")
        forecast_projection_string = " | ".join(forecast_display_list) if forecast_display_list else "N/A"
        
        sources_used = ["Live Market Data (Yahoo Finance)"]
        if all_forecasts: sources_used.append("Forecasting Models (GRU/SARIMA)")
        if web_results: sources_used.append("Web Search (Tavily AI)")
        if db_facts: sources_used.append("Annual Reports (Vector DB)")
        
        response_package = {"recommendation": final_analysis_json.get('recommendation', 'Data Insufficient'), "summary": final_analysis_json.get('analysis_summary', 'Could not generate summary.'),"drivers": final_analysis_json.get('key_drivers', []),"currentPrice": f"₹{current_price:.2f}","news": [{"title": r['title'], "url": r['url']} for r in web_results],"chartData": live_price_history,"financialRatios": self.data_fetcher.get_company_info(ticker),"forecastData": {"Projection": forecast_projection_string,"Model Used": "Hybrid" if all_forecasts else "N/A"},"forecastChartData": forecast_chart_data,"evidence": {"from_web_search": web_results,"from_vector_db": db_facts},"sources_used": sorted(list(set(sources_used)))}
        
        chart_data = response_package["chartData"]
        if chart_data is not None and not chart_data.empty:
            response_package["chartData"] = {"labels": chart_data.index.strftime('%Y-%m-%d').tolist(), "data": chart_data['Close'].tolist()}
        else: response_package["chartData"] = {}
        
        return response_package


app = Flask(__name__)
CORS(app)

print("Initializing Finsight Engine")
TAVILY_API_KEY = "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"

GOOGLE_API_KEYS = [
    "AIzaSyDFcQGpC--iFE_EfXzKL3JO3W_cF3NpeU0"
]

BASE_FOLDER = "D:\Desktop\FINSIGHT\Companies"
COMPANY_ASSET_MAP = {}
TICKER_MAP = {
    "Bajaj": "BAJFINANCE.NS", "Bharti airtel": "BHARTIARTL.NS", "HDFC": "HDFCBANK.NS",
    "Hindustanunilever": "HINDUNILVR.NS", "ICICI": "ICICIBANK.NS", "Infosys": "INFY.NS",
    "ITC": "ITC.NS", "Reliance Industries": "RELIANCE.NS", "SBI": "SBIN.NS", "TCS": "TCS.NS",
    "Tech Mahindra": "TECHM.NS", "Tata Motors": "TATAMOTORS.NS", "PGCIL": "POWERGRID.NS",
    "ONGC": "ONGC.NS", "NTPC": "NTPC.NS", "Nestle": "NESTLEIND.NS",
    "JSW": "JSWSTEEL.NS", "Coal India": "COALINDIA.NS", "HDFC Life Insurance": "HDFCLIFE.NS",
    "Adani Enterprises": "ADANIENT.NS"
}

for name, ticker in TICKER_MAP.items():
    folder_name_sanitized = name.lower().replace(' ', '').replace('.', '')
    company_dir = os.path.join(BASE_FOLDER, folder_name_sanitized)
    if os.path.isdir(company_dir):
        prefix = folder_name_sanitized
        COMPANY_ASSET_MAP[name] = {
            "gru_path": os.path.join(company_dir, f"{prefix}_gru_model_tuned.keras"),
            "sarima_path": os.path.join(company_dir, f"{prefix}_sarima_model_tuned.pkl"),
            "scaler_path": os.path.join(company_dir, f"{prefix}_scaler.pkl"),
            "bias_path": os.path.join(company_dir, f"{prefix}_bias.json"),
            "master_dataset_path": os.path.join(company_dir, f"{folder_name_sanitized}_master_dataset.csv"),
            "faiss_index_path": os.path.join(company_dir, f"{prefix}_vector_store_faiss", f"{prefix}_faiss.index"),
            "faiss_docs_path": os.path.join(company_dir, f"{prefix}_vector_store_faiss", f"{prefix}_documents.pkl"),
            "ticker": ticker
        }

try:
    agent = FinsightAgent(google_api_keys=GOOGLE_API_KEYS, tavily_api_key=TAVILY_API_KEY)
    print("--- FINSIGHT AI ENGINE READY (USING GEMINI) ---")
except Exception as e:
    agent = None; print(f"FATAL: Could not initialize Finsight Agent. {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not agent: return jsonify({'error': 'Agent not initialized'}), 503
    data = request.get_json()
    company_name, user_query = data.get('company'), data.get('query')
    if not all([company_name, user_query]): return jsonify({'error': 'Missing company or query'}), 400
    if company_name not in COMPANY_ASSET_MAP: return jsonify({'error': f'Assets for "{company_name}" not found'}), 404
    try:
        result = agent.run(company_name, user_query, COMPANY_ASSET_MAP[company_name])
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected internal server error occurred: {e}'}), 500

if __name__ == '__main__':
    if not agent:
        print("\nCould not start Flask server because Finsight Agent failed to initialize.")
    else:
        app.run(host='0.0.0.0', port=5000)