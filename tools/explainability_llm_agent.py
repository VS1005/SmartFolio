from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRADING_AGENT_ROOT = REPO_ROOT / "trading_agent"
if TRADING_AGENT_ROOT.exists() and str(TRADING_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TRADING_AGENT_ROOT))

from tradingagents.llm_provider import ProviderError, generate_completion

SYSTEM_PROMPT = ("""
You are a Financial Analyst writing a simplified newsletter for retail investors. Your goal is to explain an AI's trading strategy in plain English, removing all technical jargon.

### Input Data
You will receive "Decision Rules" that look like this:
`If AARTIIND.NS::Volume > 2.0 AND BATAINDIA.NS::Close < -1.0`...

### Interpretation Guide (Z-Scores)
The input data uses Z-scores (normalized values), not raw prices:
- **Volume > 1**: High trading volume / Activity spike.
- **Volume < -1**: Low trading volume / Quiet.
- **Close > 1**: Price is high relative to recent history (Uptrend).
- **Close < -1**: Price is low (Downtrend / "Buying the Dip").
- **Returns > 0**: Positive momentum.

### Your Task
Translate the math into a story.
1.  **Identify the Vibe**: Is the AI chasing momentum (buying high)? Is it contrarian (buying the dip)?
2.  **Explain Relationships**: If the AI buys Stock A based on Stock B's movement, call it a "Cross-market signal" or "Sector correlation".

### Output Rules (Strict)
1.  **NO JARGON**: Do not use words like "thresholds", "nodes", "z-scores", "coefficients", "average weight".
2.  **NO RAW NUMBERS**: Do not quote the specific values (like `4.42` or `0.51`). Use descriptive terms like "significant spike", "moderate drop", "stable".
3.  **NO "AVERAGE WEIGHT"**: Do not mention the allocation percentages.
4.  **Short & Simple**: Use short sentences.

### Output Format
For each stock, provide exactly this structure:

**[Stock Name]**
* **The Strategy**: [A catchy 3-6 word summary, e.g., "Momentum Play backed by Sector Peers"]
* **The Logic**: [A plain-English paragraph explaining *why*. E.g., "The model is buying this stock because it sees a massive volume spike in its peer, Tata Steel. It seems to be hedging against weakness in the broader market, only buying when the sector is quiet."]

""")


@dataclass
class SnapshotContext:
    metadata: Dict[str, str]
    filtered_data: Dict[str, object]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SmartFolio explainability narrative via LLM provider.")
    p.add_argument("--snapshot", default="explainability_results/explain_tree_custom.joblib")
    p.add_argument("--llm", action="store_true", help="Enable LLM generation")
    p.add_argument("--llm-model", default="gpt-4.1-mini", help="LLM model name")
    p.add_argument("--output", default="explainability_results/explainability_narrative.md")
    p.add_argument("--print", action="store_true")
    return p.parse_args()

def safe_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(v) for v in obj]
    else:
        return safe_convert(obj)

def load_snapshot(path: Path) -> SnapshotContext:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise TypeError("Joblib file must contain a dict.")

    keep = ["per_stock", "avg_weights", "top_indices", "global_r2", "X_shape", "Y_shape"]
    filtered: Dict[str, object] = {}
    for k in keep:
        if k in data and data[k] is not None:
            filtered[k] = data[k]

    meta = {"model_path": str(path), "included_keys": list(filtered.keys())}
    return SnapshotContext(metadata=meta, filtered_data=filtered)

def assemble_prompt(ctx: SnapshotContext) -> str:
    """
    Builds a structured LLM-ready prompt emphasizing narrative over data.
    """

    example_block = (
        "Example output:\n\n"
        "**ICICIBANK.NS**\n"
        "* **The Strategy**: Sector Rotation based on Peers\n"
        "* **The Logic**: The model is aggressively buying ICICI Bank because it sees a major volume spike in SBI. It uses this as a confirmation signal for the entire banking sector. It avoids this trade if Infosys is also crashing, treating that as a broader market risk.\n"
        "\n"
    )

    instructions = (
        "You are given structured JSON data summarizing decision-tree surrogates. "
        "Feature names follow the pattern 'TICKER::t-<N>::Metric', where t-0 is the most recent trading day, t-1 is one day back, etc. "
        "Convert the 'rules' and 'feature_importances' for each stock into the plain English format defined in the system prompt. "
        "Ignore the 'avg_weight' field completely in your output.\n"
    )

    payload = {
        "system_instruction": SYSTEM_PROMPT,
        "metadata": ctx.metadata,
        "instructions": instructions,
        "example_output": example_block,
        "explainability_data": ctx.filtered_data,
    }
    print(payload)

    return json.dumps(convert_keys_to_str(payload), indent=2, default=safe_convert)

def llm_narrative(prompt: str, model="gpt-4.1-mini", retries=3, delay=4) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] LLM call attempt {attempt}/{retries}")
            return generate_completion(
                prompt,
                system_prompt=SYSTEM_PROMPT,
                model=model,
                temperature=0.4,
                top_p=0.9,
            )
        except ProviderError as exc:
            last_error = exc
            message = str(exc)
            if "429" in message and attempt < retries:
                print(f"[WARN] Rate limited. Retrying in {delay}s…")
                time.sleep(delay)
                continue
            print(f"[ERROR] LLM call failed: {exc}")
            return f"**LLM generation failed:** {exc}"
    return f"**LLM generation unavailable:** {last_error}" if last_error else "**LLM generation unavailable.**"

def fallback_narrative(ctx: SnapshotContext) -> str:
    d = ctx.filtered_data
    per_stock = d.get("per_stock")
    if isinstance(per_stock, dict):
        stock_count = len(per_stock)
    elif isinstance(per_stock, (list, tuple)):
        stock_count = len(per_stock)
    else:
        stock_count = 0

    return (
        f"Fallback summary — global R² {d.get('global_r2','n/a')}, "
        f"{stock_count} stocks analyzed. "
        "Detailed logic requires LLM generation."
    )

def main() -> None:
    args = parse_args()
    snap = Path(args.snapshot).expanduser()
    try:
        ctx = load_snapshot(snap)
        print(f"[INFO] Loaded keys: {ctx.metadata['included_keys']}")
    except Exception as e:
        print(f"[ERROR] Snapshot load failed: {e}")
        sys.exit(1)

    prompt = assemble_prompt(ctx)
    prompt_path = Path(args.output).with_name("input_tree_llm_prompt.json")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    print(f"[INFO] Prompt saved → {prompt_path}")

    if args.llm:
        if not any(
            os.environ.get(var)
            for var in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")
        ):
            print(
                "[WARN] --llm set but no API key detected; export OPENAI_API_KEY or configure LLM_PROVIDER.",
                file=sys.stderr,
            )
        output_text = llm_narrative(prompt, model=args.llm_model)
    else:
        output_text = fallback_narrative(ctx)

    out_path = Path(args.output)
    out_path.write_text(output_text, encoding="utf-8")

    if args.print:
        print("\n--- Generated Narrative ---\n")
        print(output_text)
    print(f"Narrative written to {out_path}")


if __name__ == "__main__":
    main()
