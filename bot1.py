


from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import AzureOpenAI

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT",    "https://vcon.openai.azure.com/")
AZURE_API_KEY  = os.getenv("AZURE_OPENAI_KEY",         "EfwhbUa3MrbLrioEAyALwcjwffIAOQzr6rtdeRXbTvgROOyrl9pmJQQJ99CAACYeBjFXJ3w3AAABACOGUCez")
AZURE_API_VER  = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT     = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-4o")

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VER,
)


def llm(messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

class DataStore:

    def __init__(
        self,
        project_folder: Optional[str] = None,   # path to extracted TXT folder
        baselines_csv:  Optional[str] = None,
        monthly_csv:    Optional[str] = None,
    ):
        self.txt_docs: dict[str, str] = {}
        if project_folder and Path(project_folder).exists():
            self._load_folder(project_folder)

        self.baselines_df: Optional[pd.DataFrame] = None
        self.monthly_df:   Optional[pd.DataFrame] = None

        if baselines_csv and Path(baselines_csv).exists():
            self.baselines_df = pd.read_csv(baselines_csv)
            self.baselines_df.columns = self.baselines_df.columns.str.strip()

        if monthly_csv and Path(monthly_csv).exists():
            self.monthly_df = pd.read_csv(monthly_csv)
            self.monthly_df.columns = self.monthly_df.columns.str.strip()

        self.known_topic_ids:   set[str] = set()
        self.known_topic_names: dict[str, str] = {}  
        self._build_topic_index()

    def _load_folder(self, folder: str) -> None:
        
        base = Path(folder)
        for txt_file in base.rglob("*.txt"):
            relative_name = str(txt_file.relative_to(base))
            self.txt_docs[relative_name] = txt_file.read_text(
                encoding="utf-8", errors="replace"
            )

    def _build_topic_index(self) -> None:
        
        for df in [self.baselines_df, self.monthly_df]:
            if df is None:
                continue
            if "topic_id" in df.columns:
                self.known_topic_ids.update(df["topic_id"].dropna().unique())
            if "topic_id" in df.columns and "topic_name" in df.columns:
                pairs = df[["topic_id", "topic_name"]].drop_duplicates()
                for _, row in pairs.iterrows():
                    key = str(row["topic_name"]).lower().strip()
                    self.known_topic_names[key] = row["topic_id"]


    def topic_id_from_name(self, name: str) -> Optional[str]:
       
        lower = name.lower().strip()

        if lower in self.known_topic_names:
            return self.known_topic_names[lower]

        for k, v in self.known_topic_names.items():
            if lower in k or k in lower:
                return v

        return None

    def summary(self) -> str:
        lines = [
            f"TXT docs    : {len(self.txt_docs)} files",
            (f"Baselines   : {len(self.baselines_df)} rows"
             if self.baselines_df is not None else "Baselines   : not loaded"),
            (f"Monthly     : {len(self.monthly_df)} rows"
             if self.monthly_df is not None else "Monthly     : not loaded"),
            f"Known topics: {len(self.known_topic_ids)}",
        ]
        return "\n".join(lines)


ROUTER_SYSTEM = """\
You are a query router for a construction-project safety knowledge base.
Given a user question, output ONLY valid JSON with these keys:

  "intent"  : one of "descriptive" | "numeric" | "trend" | "hybrid"
  "sources" : list containing one or more of "txt" | "csv_baselines" | "csv_monthly"
  "topics"  : list of topic names mentioned or implied (use canonical names if possible)
  "period"  : ISO month string like "2025-03" if a specific month is mentioned, else null
  "filters" : any extra filter hints (e.g. workstream name), else null

Routing rules:
  "descriptive" → sources: ["txt"]
      Use for: what does X say, who are the owners, what fields are captured, what is the focus
  "numeric"     → sources: ["csv_baselines"] or ["csv_monthly"] or both
      Use for: scores, metrics, counts, percentages pulled from the CSV data
  "trend"       → sources: ["csv_monthly"]
      Use for: changes over time, month-by-month comparisons, nonzero values across months
  "hybrid"      → sources: ["txt", "csv_baselines"] or ["txt", "csv_monthly"] or all three
      Use for: questions asking for both procedural/descriptive context AND numeric data

Output nothing except the JSON object. No markdown, no explanation.
"""


def route_query(question: str, store: DataStore) -> dict:
    """
    Send the question to the LLM router and get back a routing decision as a dict.
    Includes known topic names as a hint so the router can identify topics accurately.
    """
    topic_hint = (
        "Known topic names: " + ", ".join(sorted(store.known_topic_names.keys())[:40])
        if store.known_topic_names else ""
    )
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user",   "content": f"{topic_hint}\n\nQuestion: {question}"},
    ]
    raw = llm(messages, max_tokens=300)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "intent":  "hybrid",
            "sources": ["txt", "csv_baselines", "csv_monthly"],
            "topics":  [],
            "period":  None,
            "filters": None,
        }


def _score_doc(text: str, keywords: list[str]) -> int:
    """Count total keyword occurrences in a document (case-insensitive)."""
    lower = text.lower()
    return sum(lower.count(kw.lower()) for kw in keywords if kw)


def retrieve_txt(
    question: str,
    topics:   list[str],
    store:    DataStore,
    top_k:    int = 3,
) -> list[dict]:
    
    if not store.txt_docs:
        return []

    keywords = topics + question.lower().split()
    scored = [
        (name, content, _score_doc(content, keywords))
        for name, content in store.txt_docs.items()
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    results = []
    for name, content, score in scored[:top_k]:
        snippet = content[:1200].strip()
        results.append({"source": name, "content": snippet, "score": score})
    return results



def query_baselines(topics: list[str], store: DataStore) -> dict:
   
    if store.baselines_df is None or not topics:
        return {"rows": [], "note": "Baselines CSV not available or no topics specified."}

    df = store.baselines_df
    matched_rows = []
    matched_ids  = []

    for topic in topics:
        tid = store.topic_id_from_name(topic) or topic

        mask = (
            (df["topic_id"].str.lower()   == tid.lower())   |
            (df["topic_name"].str.lower() == topic.lower())
        )
        sub = df[mask]
        if not sub.empty:
            matched_rows.append(sub)
            matched_ids.append(tid)

    if not matched_rows:
        leaves = df[df["is_leaf"] == 1] if "is_leaf" in df.columns else df
        return {
            "rows":           leaves.head(5).to_dict(orient="records"),
            "note":           f"No exact match for {topics}. Showing sample leaf topics.",
            "topic_ids_used": [],
        }

    result_df = pd.concat(matched_rows).drop_duplicates()
    return {
        "rows":           result_df.to_dict(orient="records"),
        "note":           f"Baseline data for topic(s): {matched_ids}",
        "topic_ids_used": matched_ids,
    }


def query_monthly(
    topics:  list[str],
    period:  Optional[str],
    store:   DataStore,
    filters: Optional[str] = None,
) -> dict:
    
    if store.monthly_df is None or not topics:
        return {"rows": [], "note": "Monthly CSV not available or no topics specified."}

    df = store.monthly_df
    matched_rows = []
    matched_ids  = []

    for topic in topics:
        tid = store.topic_id_from_name(topic) or topic

        mask = (
            (df["topic_id"].str.lower()   == tid.lower())   |
            (df["topic_name"].str.lower() == topic.lower())
        )
        sub = df[mask]
        if not sub.empty:
            # Apply period filter if provided
            if period:
                sub = sub[sub["period_month"] == period]
            matched_rows.append(sub)
            matched_ids.append(tid)

    if not matched_rows:
        return {
            "rows": [],
            "note": (
                f"No monthly data found for {topics}"
                + (f" in period {period}" if period else "")
            ),
            "topic_ids_used": [],
        }

    result_df = pd.concat(matched_rows).drop_duplicates()

    # Optional workstream keyword filter
    if filters and "workstream" in result_df.columns:
        ws_mask = result_df["workstream"].str.lower().str.contains(
            filters.lower(), na=False
        )
        if ws_mask.any():
            result_df = result_df[ws_mask]

    return {
        "rows":           result_df.to_dict(orient="records"),
        "note":           (
            f"Monthly data for {matched_ids}"
            + (f", period={period}" if period else " (all periods)")
        ),
        "topic_ids_used": matched_ids,
        "period_used":    period,
    }



SYNTHESIS_SYSTEM = """\
You are a construction-project safety assistant. Answer the user's question
using ONLY the evidence provided below. Follow these rules strictly:

1. Never invent or estimate numbers. Every numeric value must come from the CSV evidence.
2. Always cite your sources:
   - For TXT evidence : mention the document filename
   - For CSV evidence : mention the topic_id, which CSV (baselines or monthly), and period_month
3. If the evidence does not contain the answer, say so clearly — do not guess.
4. Keep answers concise and professional.
5. Use bullet points when listing multiple items.
"""


def synthesize(
    question:        str,
    route:           dict,
    txt_hits:        list[dict],
    baseline_result: dict,
    monthly_result:  dict,
) -> str:
    """
    Build an evidence block from all retrieved results and ask the LLM
    to produce a grounded, cited answer.
    """
    evidence_parts = []

    if txt_hits:
        txt_block = "\n\n".join(
            f"[TXT: {h['source']}]\n{h['content']}" for h in txt_hits
        )
        evidence_parts.append(f"=== TXT DOCUMENT EVIDENCE ===\n{txt_block}")

    if baseline_result.get("rows"):
        bl_text = json.dumps(baseline_result["rows"], indent=2)
        evidence_parts.append(
            f"=== BASELINES CSV ({baseline_result.get('note', '')}) ===\n{bl_text}"
        )

    if monthly_result.get("rows"):
        mo_text = json.dumps(monthly_result["rows"], indent=2)
        evidence_parts.append(
            f"=== MONTHLY METRICS CSV ({monthly_result.get('note', '')}) ===\n{mo_text}"
        )

    if not evidence_parts:
        evidence_parts.append(
            "No relevant evidence was retrieved for this question."
        )

    evidence = "\n\n".join(evidence_parts)

    # Trim to stay within gpt-4o context window
    if len(evidence) > 12000:
        evidence = evidence[:12000] + "\n... [truncated for length]"

    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Routing decision: {json.dumps(route)}\n\n"
                f"{evidence}"
            ),
        },
    ]
    return llm(messages, max_tokens=1024)

class LifecycleSafetyBot:
    """End-to-end pipeline: question → grounded answer."""

    def __init__(self, store: DataStore):
        self.store   = store
        self.history: list[dict] = []  # kept for multi-turn memory / UI

    def ask(self, question: str, verbose: bool = False) -> str:
        """
        Full pipeline for one question:
          Route → Retrieve TXT → Query CSVs → Synthesize → Return answer
        """

        route = route_query(question, self.store)
        if verbose:
            print(f"\n[ROUTER] {json.dumps(route, indent=2)}")

        sources = route.get("sources", [])
        topics  = route.get("topics",  [])
        period  = route.get("period")
        filters = route.get("filters")

        txt_hits = []
        if "txt" in sources:
            txt_hits = retrieve_txt(question, topics, self.store)
            if verbose:
                print(f"[TXT] Retrieved {len(txt_hits)} doc(s): "
                      f"{[h['source'] for h in txt_hits]}")

        baseline_result = {"rows": [], "note": "Not queried."}
        monthly_result  = {"rows": [], "note": "Not queried."}

        if "csv_baselines" in sources:
            baseline_result = query_baselines(topics, self.store)
            if verbose:
                print(f"[BASELINES] {baseline_result['note']} "
                      f"→ {len(baseline_result['rows'])} row(s)")

        if "csv_monthly" in sources:
            monthly_result = query_monthly(topics, period, self.store, filters)
            if verbose:
                print(f"[MONTHLY] {monthly_result['note']} "
                      f"→ {len(monthly_result['rows'])} row(s)")

        answer = synthesize(
            question, route, txt_hits, baseline_result, monthly_result
        )

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

EVAL_QUESTIONS = [
    # ── TXT-only ─────────────────────────────────────────────
    {
        "q":              "List the typical data / fields captured for Work at Height.",
        "type":           "txt",
        "check_keywords": ["data", "fields", "work at height"],
    },
    {
        "q":              "Who are the primary owners of Permit to Work?",
        "type":           "txt",
        "check_keywords": ["owner", "permit"],
    },
    {
        "q":              "What does Scaffold & Fall Protection Audits focus on?",
        "type":           "txt",
        "check_keywords": ["scaffold", "fall", "audit"],
    },
    # ── CSV-only ─────────────────────────────────────────────
    {
        "q":              "What is the inherent_risk_score for Confined Space in the baseline data?",
        "type":           "csv",
        "check_keywords": ["confined space", "inherent_risk_score", "9.4"],
    },
    {
        "q":              "Across all topics in 2025-08, what is the total inspections_completed?",
        "type":           "csv",
        "check_keywords": ["2025-08", "inspections_completed"],
    },
    {
        "q":              "In which months did TRIR / DART Tracking have a nonzero trir_value?",
        "type":           "csv",
        "check_keywords": ["trir", "dart"],
    },
    # ── Hybrid ───────────────────────────────────────────────
    {
        "q":              "For Confined Space, list the typical data / fields and provide baseline permit_required_pct and baseline_training_hours.",
        "type":           "hybrid",
        "check_keywords": ["confined space", "permit_required_pct", "baseline_training_hours"],
    },
    {
        "q":              "For Weekly Safety Inspections, list related items and report findings_opened and findings_closed in 2025-02.",
        "type":           "hybrid",
        "check_keywords": ["weekly safety", "findings_opened", "findings_closed", "2025-02"],
    },
    {
        "q":              "For Permit to Work, what are the primary owners in the file and how many permits_issued were recorded in 2025-03?",
        "type":           "hybrid",
        "check_keywords": ["permit to work", "permits_issued", "2025-03"],
    },
    {
        "q":              "For Excavation & Ground Disturbance, list typical data fields, then provide baseline inherent_risk_score and permit_required_pct, and finally report inspections_completed and compliance_score_pct in 2025-08.",
        "type":           "hybrid",
        "check_keywords": ["excavation", "inherent_risk_score", "compliance_score_pct", "2025-08"],
    },
]


def run_eval(bot: LifecycleSafetyBot) -> None:
    print("\n" + "=" * 65)
    print("EVALUATION HARNESS")
    print("=" * 65)

    passed = 0
    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[Q{i}] ({item['type'].upper()}) {item['q']}")
        answer  = bot.ask(item["q"])
        a_lower = answer.lower()

        hits   = [kw for kw in item["check_keywords"] if kw.lower() in a_lower]
        score  = len(hits) / len(item["check_keywords"])
        status = "PASS" if score >= 0.5 else "PARTIAL" if score > 0 else "FAIL"

        if score >= 0.5:
            passed += 1

        print(f"[{status}] keyword coverage {len(hits)}/{len(item['check_keywords'])}")
        print(textwrap.indent(
            answer[:400] + ("..." if len(answer) > 400 else ""), "  "
        ))
        bot.reset()

    print(f"\n{'=' * 65}")
    print(f"Result: {passed}/{len(EVAL_QUESTIONS)} questions passed (>=50% keyword match)")
    print("=" * 65)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Lifecycle Safety Bot")
    parser.add_argument(
        "--folder",
        default="Project",
        help="Path to the extracted project TXT folder (default: ./Project)",
    )
    parser.add_argument(
        "--baselines",
        default="construction_topic_baselines_numeric.csv",
        help="Path to baselines CSV",
    )
    parser.add_argument(
        "--monthly",
        default="construction_monthly_metrics_numeric.csv",
        help="Path to monthly metrics CSV",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the built-in 10-question evaluation suite",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print routing and retrieval debug info for each question",
    )
    args = parser.parse_args()

    print("Loading data sources...")
    store = DataStore(
        project_folder=args.folder    if Path(args.folder).exists()    else None,
        baselines_csv =args.baselines if Path(args.baselines).exists() else None,
        monthly_csv   =args.monthly   if Path(args.monthly).exists()   else None,
    )
    print(store.summary())

    bot = LifecycleSafetyBot(store)

    if args.eval:
        run_eval(bot)
        return

    print("\nLifecycle Safety Bot ready.")
    print("Commands: 'quit' to exit | 'reset' to clear conversation history\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not q:
            continue
        if q.lower() == "quit":
            break
        if q.lower() == "reset":
            bot.reset()
            print("Conversation history cleared.\n")
            continue

        answer = bot.ask(q, verbose=args.verbose)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
