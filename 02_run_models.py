"""
Stage 2 - Prompt models.

Reads goemotions_sample.csv and custom_challenge_set.csv, sends every
row through the SAME fixed prompt to THREE FREE models, and writes
predictions to predictions.csv.

Models used (all free, no credit card required):
  - Gemini (gemini-2.0-flash) via Google AI Studio free tier
  - Llama 3.1 8B via Groq free tier
  - Gemma 2 9B via Groq free tier

Requires env vars: GOOGLE_API_KEY, GROQ_API_KEY
  - Gemini key: aistudio.google.com/apikey (free, no card)
  - Groq key:   console.groq.com/keys (free, no card)

Run:  python 02_run_models.py
"""
import os
import re
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
LABEL_SET = set(LABELS)

PROMPT_TEMPLATE = """Classify the following text into exactly ONE emotion from this fixed list:
{labels}

Text: "{text}"

Respond with ONLY the single emotion word from the list above, nothing else."""


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.format(labels=", ".join(LABELS), text=text)


def extract_label(raw: str) -> str:
    """Pull the first recognized label out of a model's free-text reply."""
    if raw is None:
        return "PARSE_ERROR"
    cleaned = raw.strip().lower().strip(".\"'")
    if cleaned in LABEL_SET:
        return cleaned
    for lab in LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", cleaned):
            return lab
    return "PARSE_ERROR"


# ---------- model callers (each returns raw string reply) ----------

def call_gemini(client, text, model="gemini-2.0-flash"):
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=build_prompt(text),
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=10),
    )
    return resp.text


def call_groq(client, text, model):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=10,
        messages=[{"role": "user", "content": build_prompt(text)}],
    )
    return resp.choices[0].message.content


def call_with_retry(fn, *args, tries=5, base_sleep=3, **kwargs):
    """Retries with growing backoff -- important on free tiers, which
    rate-limit aggressively rather than reject outright."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == tries - 1:
                return f"ERROR: {e}"
            time.sleep(base_sleep * (attempt + 1))


def main():
    from google import genai
    from openai import OpenAI  # Groq speaks the OpenAI-compatible API

    gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    groq_client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    df = pd.concat(
        [pd.read_csv("goemotions_sample.csv"), pd.read_csv("custom_challenge_set.csv")],
        ignore_index=True,
    )

    tasks = {
        "gemini": (lambda t: call_with_retry(call_gemini, gemini_client, t), 2),
        "llama": (lambda t: call_with_retry(call_groq, groq_client, t, "llama-3.1-8b-instant"), 4),
        "gemma": (lambda t: call_with_retry(call_groq, groq_client, t, "gemma2-9b-it"), 4),
    }

    # RESUME: if predictions.csv already exists with a model's *_pred column
    # fully populated (no missing values), skip re-running that model.
    # This means a crash/restart never loses finished work again.
    done_cols = {}
    if os.path.exists("predictions.csv"):
        prev = pd.read_csv("predictions.csv")
        if len(prev) == len(df) and (prev["text"] == df["text"]).all():
            for name in tasks:
                col = f"{name}_pred"
                if col in prev.columns and prev[col].notna().all():
                    done_cols[name] = prev[col]
                    print(f"Skipping {name} -- already completed in predictions.csv")

    for name, (fn, workers) in tasks.items():
        if name in done_cols:
            df[f"{name}_pred"] = done_cols[name].values
            continue
        print(f"Running {name} on {len(df)} examples (max_workers={workers})...")
        raw_results = [None] * len(df)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, row.text): i for i, row in df.iterrows()}
            for fut in tqdm(as_completed(futures), total=len(futures)):
                i = futures[fut]
                raw_results[i] = fut.result()

        df[f"{name}_raw"] = raw_results
        df[f"{name}_pred"] = df[f"{name}_raw"].apply(extract_label)

        # CHECKPOINT: save after every model, not just at the very end
        df.to_csv("predictions.csv", index=False)
        print(f"Checkpoint saved -- {name} done, predictions.csv updated")

    print("All models done. Final predictions.csv is up to date.")


if __name__ == "__main__":
    main()

