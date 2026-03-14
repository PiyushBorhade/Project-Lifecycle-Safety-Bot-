"""
Construction Safety Bot - Complete Solution for Hackathon Evaluation
Meets all evaluation criteria including routing accuracy, answer grounding, CSV computations, etc.
"""

from __future__ import annotations

import os
import pandas as pd
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import zipfile
import glob
from datetime import datetime
import traceback

from openai import AzureOpenAI
import streamlit as st

# ============================================================================
# Azure OpenAI Configuration
# ============================================================================

AZURE_ENDPOINT = "https://vcon.openai.azure.com/"
AZURE_API_KEY = "EfwhbUa3MrbLrioEAyALwcjwffIAOQzr6rtdeRXbTvgROOyrl9pmJQQJ99CAACYeBjFXJ3w3AAABACOGUCez"
AZURE_DEPLOYMENT = "gpt-4o"
AZURE_API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION
)

def llm(messages: list[dict], max_tokens: int = 1024) -> str:
    """Call Azure OpenAI"""
    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()


# ============================================================================
# Data Store with Enhanced Topic Management
# ============================================================================

class DataStore:
    """Central data repository with comprehensive indexing"""
    
    def __init__(self, project_folder: str, baselines_csv: str, monthly_csv: str):
        self.txt_docs = {}
        self.txt_metadata = {}
        self.baselines_df = None
        self.monthly_df = None
        
        # Enhanced indices
        self.topic_name_to_id = {}      # Normalized name -> topic_id
        self.topic_id_to_name = {}      # topic_id -> canonical name
        self.topic_id_to_workstream = {} # topic_id -> workstream
        self.topic_id_to_baseline = {}   # topic_id -> baseline row
        self.available_periods = []       # List of months with data
        
        # Load data (silently, no prints)
        self._load_txt_files(project_folder)
        self._load_csv_files(baselines_csv, monthly_csv)
        self._build_indices()
    
    def _load_txt_files(self, folder_path: str):
        """Load all TXT files with metadata"""
        if not Path(folder_path).exists():
            return
        
        for txt_file in glob.glob(f"{folder_path}/**/*.txt", recursive=True):
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                filename = os.path.basename(txt_file)
                topic_name = filename.replace('.txt', '').replace('_', ' ')
                
                self.txt_docs[filename] = content
                self.txt_metadata[filename] = {
                    'topic': topic_name,
                    'filename': filename,
                    'folder': os.path.basename(os.path.dirname(txt_file)),
                    'path': txt_file,
                    'size': len(content)
                }
            except Exception:
                pass  # Silent fail
    
    def _load_csv_files(self, baselines_csv: str, monthly_csv: str):
        """Load CSV files with validation"""
        if Path(baselines_csv).exists():
            self.baselines_df = pd.read_csv(baselines_csv)
            self.baselines_df.columns = [c.strip() for c in self.baselines_df.columns]
        
        if Path(monthly_csv).exists():
            self.monthly_df = pd.read_csv(monthly_csv)
            self.monthly_df.columns = [c.strip() for c in self.monthly_df.columns]
            
            if 'period_month' in self.monthly_df.columns:
                self.available_periods = sorted(self.monthly_df['period_month'].unique())
    
    def _build_indices(self):
        """Build comprehensive lookup indices"""
        
        # From baselines
        if self.baselines_df is not None:
            for _, row in self.baselines_df.iterrows():
                if pd.notna(row.get('topic_id')) and pd.notna(row.get('topic_name')):
                    topic_id = str(row['topic_id'])
                    topic_name = str(row['topic_name']).strip()
                    
                    self.topic_name_to_id[topic_name.lower()] = topic_id
                    self.topic_id_to_name[topic_id] = topic_name
                    
                    if 'workstream' in row and pd.notna(row['workstream']):
                        self.topic_id_to_workstream[topic_id] = str(row['workstream'])
                    
                    self.topic_id_to_baseline[topic_id] = row.to_dict()
        
        # From monthly
        if self.monthly_df is not None:
            for _, row in self.monthly_df.iterrows():
                if pd.notna(row.get('topic_id')) and pd.notna(row.get('topic_name')):
                    topic_id = str(row['topic_id'])
                    topic_name = str(row['topic_name']).strip()
                    
                    if topic_id not in self.topic_id_to_name:
                        self.topic_id_to_name[topic_id] = topic_name
                    if topic_name.lower() not in self.topic_name_to_id:
                        self.topic_name_to_id[topic_name.lower()] = topic_id
    
    def find_topic(self, query: str) -> List[Tuple[str, str, float]]:
        """Find topics matching query with confidence scores"""
        results = []
        query_lower = query.lower()
        
        for name, topic_id in self.topic_name_to_id.items():
            if query_lower == name:
                results.append((topic_id, name, 1.0))
            elif query_lower in name:
                confidence = len(query_lower) / len(name)
                results.append((topic_id, name, confidence))
            else:
                query_words = set(query_lower.split())
                name_words = set(name.split())
                common = query_words & name_words
                if common:
                    confidence = len(common) / max(len(query_words), len(name_words))
                    results.append((topic_id, name, confidence))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]
    
    def get_txt_for_topic(self, topic_name: str) -> Optional[Dict]:
        """Get TXT document for a topic"""
        topic_lower = topic_name.lower()
        for filename, meta in self.txt_metadata.items():
            if topic_lower in meta['topic'].lower():
                return {
                    'content': self.txt_docs[filename],
                    'metadata': meta
                }
        return None


# ============================================================================
# Enhanced Query Router with Confidence Scoring
# ============================================================================

ROUTER_SYSTEM = """You are a query router for construction safety data. Analyze the question and return JSON.

Available data sources:
- TXT: Documents with procedures, descriptions, owners, fields
- CSV (baselines): Static metrics (risk scores, training hours, targets)
- CSV (monthly): Time-series metrics (inspections, findings, compliance)

Return format:
{
    "sources": ["txt"] or ["csv"] or ["txt", "csv"],
    "primary_topics": ["main topic names"],
    "secondary_topics": ["related topics"],
    "time_period": "YYYY-MM" or null,
    "time_range": {"start": "YYYY-MM", "end": "YYYY-MM"} or null,
    "metrics": ["column names"],
    "aggregation": "sum" or "avg" or "min" or "max" or "count" or null,
    "filters": {"workstream": "name"} or null,
    "confidence": 0.0-1.0,
    "reasoning": "why these choices"
}

Rules:
- descriptive (what/describe/list) → TXT
- numeric (what is/score/metric) → CSV
- trend (over time/compare months) → CSV monthly
- hybrid (describe AND show numbers) → TXT + CSV
"""

def route_query(question: str, store: DataStore) -> Dict:
    """Route query with confidence scoring"""
    
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    try:
        raw = llm(messages)
        raw = re.sub(r'```(?:json)?|```', '', raw).strip()
        route = json.loads(raw)
        
        if route.get('primary_topics'):
            matched_topics = []
            for topic in route['primary_topics']:
                matches = store.find_topic(topic)
                if matches:
                    matched_topics.append(matches[0][1])
            route['primary_topics'] = matched_topics
        
        return route
        
    except Exception:
        return {
            "sources": ["txt", "csv"],
            "primary_topics": [],
            "secondary_topics": [],
            "time_period": None,
            "time_range": None,
            "metrics": [],
            "aggregation": None,
            "filters": None,
            "confidence": 0.5,
            "reasoning": "Fallback routing"
        }


# ============================================================================
# Enhanced TXT Retriever with Section Extraction
# ============================================================================

def retrieve_txt(question: str, topics: List[str], store: DataStore) -> Dict:
    """Retrieve relevant TXT content with provenance"""
    
    if not store.txt_docs:
        return {"content": "", "sources": [], "confidence": 0}
    
    results = []
    
    for topic in topics:
        doc = store.get_txt_for_topic(topic)
        if doc:
            sections = extract_sections(doc['content'], question)
            results.append({
                'topic': topic,
                'filename': doc['metadata']['filename'],
                'folder': doc['metadata']['folder'],
                'content': sections,
                'confidence': 0.9
            })
    
    if not results:
        keywords = set(question.lower().split())
        for filename, content in store.txt_docs.items():
            score = 0
            content_lower = content.lower()
            for kw in keywords:
                if len(kw) > 3:
                    score += content_lower.count(kw)
            if score > 5:
                results.append({
                    'topic': store.txt_metadata[filename]['topic'],
                    'filename': filename,
                    'folder': store.txt_metadata[filename]['folder'],
                    'content': extract_sections(content, question)[:1000],
                    'confidence': min(score / 20, 0.8)
                })
    
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "content": "\n\n".join([f"From {r['filename']}:\n{r['content']}" for r in results[:3]]),
        "sources": [f"{r['filename']} (topic: {r['topic']})" for r in results[:3]],
        "confidence": results[0]['confidence'] if results else 0
    }


def extract_sections(content: str, question: str) -> str:
    """Extract most relevant sections from document"""
    paragraphs = content.split('\n\n')
    scored = []
    question_lower = question.lower()
    
    for p in paragraphs:
        if len(p.strip()) < 50:
            continue
        p_lower = p.lower()
        score = 0
        
        headers = ['typical data', 'fields', 'primary owners', 'overview', 'related']
        for h in headers:
            if h in p_lower[:200]:
                score += 20
            if h in question_lower:
                score += 10
        
        for term in question_lower.split():
            if len(term) > 3 and term in p_lower:
                score += 2
        
        if score > 0:
            scored.append((score, p))
    
    if scored:
        scored.sort(reverse=True)
        return "\n\n".join([p for _, p in scored[:3]])
    
    return content[:1000]


# ============================================================================
# Enhanced CSV Query Engine with Accurate Computations
# ============================================================================

def query_csv(route: Dict, store: DataStore) -> Dict:
    """Query CSV data with accurate aggregations and filters"""
    
    results = {
        "data": [],
        "sources": [],
        "computations": [],
        "confidence": 0
    }
    
    topics = route.get('primary_topics', [])
    if not topics:
        return results
    
    period = route.get('time_period')
    time_range = route.get('time_range')
    metrics = route.get('metrics', [])
    aggregation = route.get('aggregation')
    filters = route.get('filters', {})
    
    # Query baselines
    if store.baselines_df is not None:
        baseline_rows = []
        for topic in topics:
            matches = store.find_topic(topic)
            if matches:
                topic_id = matches[0][0]
                mask = store.baselines_df['topic_id'].astype(str) == topic_id
                rows = store.baselines_df[mask]
                if not rows.empty:
                    baseline_rows.append(rows)
        
        if baseline_rows:
            combined = pd.concat(baseline_rows)
            if metrics:
                available = [m for m in metrics if m in combined.columns]
                if available:
                    cols = ['topic_id', 'topic_name'] + available
                    cols = [c for c in cols if c in combined.columns]
                    combined = combined[cols]
            results['data'].extend(combined.to_dict('records'))
            results['sources'].append("baseline (static metrics)")
    
    # Query monthly
    if store.monthly_df is not None:
        monthly_rows = []
        for topic in topics:
            matches = store.find_topic(topic)
            if matches:
                topic_id = matches[0][0]
                mask = store.monthly_df['topic_id'].astype(str) == topic_id
                
                if period and 'period_month' in store.monthly_df.columns:
                    mask &= store.monthly_df['period_month'] == period
                if time_range:
                    if 'period_month' in store.monthly_df.columns:
                        start, end = time_range.get('start'), time_range.get('end')
                        if start and end:
                            mask &= (store.monthly_df['period_month'] >= start) & \
                                   (store.monthly_df['period_month'] <= end)
                
                if filters and 'workstream' in filters and 'workstream' in store.monthly_df.columns:
                    mask &= store.monthly_df['workstream'].str.contains(filters['workstream'], case=False, na=False)
                
                rows = store.monthly_df[mask]
                if not rows.empty:
                    monthly_rows.append(rows)
        
        if monthly_rows:
            combined = pd.concat(monthly_rows)
            
            if aggregation and metrics:
                computations = []
                for metric in metrics:
                    if metric in combined.columns and pd.api.types.is_numeric_dtype(combined[metric]):
                        if aggregation == 'sum':
                            val = combined[metric].sum()
                        elif aggregation == 'avg':
                            val = combined[metric].mean()
                        elif aggregation == 'min':
                            val = combined[metric].min()
                        elif aggregation == 'max':
                            val = combined[metric].max()
                        elif aggregation == 'count':
                            val = len(combined)
                        else:
                            continue
                        
                        computations.append({
                            'metric': metric,
                            'aggregation': aggregation,
                            'value': float(val) if isinstance(val, (int, float)) else val,
                            'period': period or f"{time_range}"
                        })
                
                if computations:
                    results['computations'] = computations
                    results['sources'].append(f"monthly ({aggregation} over {len(combined)} rows)")
            
            if metrics and not aggregation:
                available = [m for m in metrics if m in combined.columns]
                if available:
                    cols = ['topic_id', 'topic_name', 'period_month'] + available
                    cols = [c for c in cols if c in combined.columns]
                    combined = combined[cols]
            
            results['data'].extend(combined.to_dict('records'))
    
    results['confidence'] = 0.9 if results['data'] or results['computations'] else 0
    return results


# ============================================================================
# Response Synthesizer with Grounding
# ============================================================================

SYNTHESIS_SYSTEM = """You are a construction safety assistant. Answer using ONLY the provided evidence.

RULES:
1. Every claim must be traceable to provided evidence
2. Cite sources: document names, topic_ids, periods
3. For numbers, specify if from baseline or monthly data
4. If evidence missing, say "I don't have information about..."
5. No hallucinations - if unsure, say so
6. Format clearly with bullet points for lists

Structure:
- Direct answer to question
- Supporting evidence with citations
- Summary if multiple points
"""

def synthesize_response(
    question: str,
    route: Dict,
    txt_result: Dict,
    csv_result: Dict
) -> Tuple[str, Dict]:
    """Generate grounded response with traceability"""
    
    evidence = []
    
    if txt_result['content']:
        evidence.append(f"TXT DOCUMENTS:\n{txt_result['content']}")
        evidence.append(f"TXT SOURCES: {', '.join(txt_result['sources'])}")
    
    if csv_result['data']:
        evidence.append(f"CSV DATA:\n{json.dumps(csv_result['data'][:5], indent=2)}")
        if csv_result['computations']:
            evidence.append(f"COMPUTATIONS:\n{json.dumps(csv_result['computations'], indent=2)}")
        evidence.append(f"CSV SOURCES: {', '.join(csv_result['sources'])}")
    
    if not evidence:
        evidence.append("No relevant data found in knowledge base.")
    
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user", "content": f"""Question: {question}

Routing Decision:
{json.dumps(route, indent=2)}

Evidence:
{chr(10).join(evidence)}

Provide answer with citations:"""}
    ]
    
    answer = llm(messages)
    
    trace = {
        "routing": route,
        "txt_sources": txt_result['sources'],
        "csv_sources": csv_result['sources'],
        "computations": csv_result['computations'],
        "confidence": min(txt_result['confidence'], csv_result['confidence']) if csv_result['confidence'] else txt_result['confidence']
    }
    
    return answer, trace


# ============================================================================
# Main Bot Class
# ============================================================================

class LifecycleSafetyBot:
    """Main bot class with all evaluation features"""
    
    def __init__(self, store: DataStore):
        self.store = store
        self.conversation = []
        self.evaluation_log = []
    
    def ask(self, question: str, return_trace: bool = False):
        """Process question with optional trace return"""
        
        route = route_query(question, self.store)
        
        txt_result = retrieve_txt(question, route.get('primary_topics', []), self.store) \
            if 'txt' in route.get('sources', []) else {"content": "", "sources": [], "confidence": 0}
        
        csv_result = query_csv(route, self.store) \
            if 'csv' in route.get('sources', []) else {"data": [], "sources": [], "computations": [], "confidence": 0}
        
        answer, trace = synthesize_response(question, route, txt_result, csv_result)
        
        self.conversation.append({"question": question, "answer": answer})
        
        if return_trace:
            return answer, trace, route
        return answer


# ============================================================================
# Streamlit UI - Clean Textbox Interface
# ============================================================================

def main():
    st.set_page_config(
        page_title="Construction Safety Bot",
        page_icon="🏗️"
    )
    
    st.title("🏗️ Construction Safety Bot")
    
    @st.cache_resource
    def init_bot():
        store = DataStore(
            project_folder="Project",
            baselines_csv="construction_topic_baselines_numeric.csv",
            monthly_csv="construction_monthly_metrics_numeric.csv"
        )
        return LifecycleSafetyBot(store)
    
    bot = init_bot()
    
    question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What is the risk score for Confined Space?",
        key="question_input"
    )
    
    if question:
        with st.spinner("Finding answer..."):
            try:
                answer, trace, route = bot.ask(question, return_trace=True)
                
                st.markdown("### Answer")
                st.write(answer)
                
                with st.expander("View details"):
                    st.json({
                        "routing": route,
                        "sources": {
                            "txt": trace.get('txt_sources', []),
                            "csv": trace.get('csv_sources', [])
                        },
                        "computations": trace.get('computations', []),
                        "confidence": trace.get('confidence', 0)
                    })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()