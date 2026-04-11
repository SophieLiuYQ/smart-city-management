"""
District Scoring via Qwen 3 80B — runs locally on GB10.

Reads:  data/gold/district_analysis.json (from analyze_districts.py)
Writes: data/gold/district_scores_qwen.json

Each district gets an AI-generated analysis with scores and reasoning.

Usage (on GB10):
    python AI/analyze_districts.py                              # Step 1: build profiles
    QWEN_URL=http://localhost:3000/v1 python AI/score_districts_qwen.py  # Step 2: score with Qwen
"""
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(BASE_DIR, "data", "gold")

QWEN_URL = os.getenv("QWEN_URL", "http://localhost:3000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-80b")

SYSTEM_PROMPT = """You are a NYC urban sustainability analyst. Given a community district profile, analyze it and return ONLY a JSON object (no markdown) with:

{
  "district_code": "BX01",
  "energy_score": 0-100,
  "waste_score": 0-100,
  "nexus_score": 0-100,
  "energy_analysis": "2-3 sentences on energy consumption, solar potential, BESS opportunity",
  "waste_analysis": "2-3 sentences on waste volume, diversion gap, organic waste in refuse",
  "waste_to_energy_analysis": "2-3 sentences on AD potential: tons of organics → MWh → homes powered → CO₂ avoided",
  "bess_recommendation": "Specific BESS deployment plan for this district: which buildings, what size, expected savings",
  "equity_note": "1 sentence on Environmental Justice status and implications",
  "top_3_actions": ["action 1 with specific numbers", "action 2", "action 3"]
}

Scoring guide:
- energy_score: high solar + high consumption + EJ + good roofs = high
- waste_score: high refuse + low diversion + high complaints + few compost sites = high
- nexus_score: high when BOTH energy and waste opportunities exist in same district, especially near AD facility

Use the ACTUAL NUMBERS from the data. Don't make up statistics."""


def main():
    print("=" * 65)
    print("  DISTRICT SCORING — Qwen 3 80B (Local on GB10)")
    print(f"  Endpoint: {QWEN_URL}")
    print("=" * 65)

    # Load district profiles
    input_path = os.path.join(GOLD_DIR, "district_analysis.json")
    if not os.path.exists(input_path):
        print(f"\n[ERROR] {input_path} not found. Run analyze_districts.py first.")
        sys.exit(1)

    with open(input_path) as f:
        districts = json.load(f)
    print(f"\n[LOADED] {len(districts)} districts from district_analysis.json")

    # Connect
    from openai import OpenAI
    client = OpenAI(base_url=QWEN_URL, api_key="not-needed")

    print(f"[TEST] Pinging {QWEN_URL}...")
    try:
        models = client.models.list()
        print(f"  → Connected. Models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"  → Failed: {e}")
        sys.exit(1)

    # Score each district
    results = []
    total = len(districts)
    start = time.time()

    for i, d in enumerate(districts):
        code = d["district_code"]

        # Build compact profile for LLM (skip full building list to save tokens)
        profile = {
            "district_code": code,
            "borough": d["borough"],
            "buildings_summary": d["buildings_summary"],
            "waste": d["waste"],
            "waste_to_energy": d["waste_to_energy"],
            "complaints": d["complaints"],
            "infrastructure": d["infrastructure"],
            # Include top 3 buildings for context
            "top_buildings": [
                {
                    "site": b["site"],
                    "agency": b["agency"],
                    "ej": b["ej"],
                    "energy": b["energy"],
                    "bess": b["bess_recommendation"],
                }
                for b in d["buildings"][:3]
            ],
        }

        print(f"  [{i+1}/{total}] {code}...", end="", flush=True)
        t0 = time.time()

        try:
            resp = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(profile, default=str)},
                ],
                temperature=0.2,
                max_tokens=1000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            text = resp.choices[0].message.content.strip()
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            score = json.loads(text)
            score["district_code"] = code  # ensure it's correct
            results.append(score)
            elapsed = time.time() - t0
            print(f" E:{score.get('energy_score','?')} W:{score.get('waste_score','?')} N:{score.get('nexus_score','?')} ({elapsed:.1f}s)")

        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "district_code": code,
                "energy_score": 50, "waste_score": 50, "nexus_score": 50,
                "energy_analysis": "Scoring failed",
                "waste_analysis": "Scoring failed",
                "waste_to_energy_analysis": "",
                "bess_recommendation": "",
                "equity_note": "",
                "top_3_actions": [],
            })

    # Save
    out_path = os.path.join(GOLD_DIR, "district_scores_qwen.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed_total = time.time() - start
    print(f"\n[SAVED] {out_path}")
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE — {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  {len(results)} districts scored by Qwen")
    print(f"  Rate: {len(results)/elapsed_total:.1f} districts/sec")
    print(f"{'=' * 65}")

    # Print all
    print(f"\n  {'Code':>6s}  {'E':>3s}  {'W':>3s}  {'N':>3s}  Top Action")
    print("  " + "─" * 70)
    for r in sorted(results, key=lambda x: x.get("nexus_score", 0), reverse=True):
        actions = r.get("top_3_actions", [])
        top = actions[0][:50] if actions else ""
        print(f"  {r['district_code']:>6s}  {r.get('energy_score',0):>3d}  {r.get('waste_score',0):>3d}  "
              f"{r.get('nexus_score',0):>3d}  {top}")


if __name__ == "__main__":
    main()
