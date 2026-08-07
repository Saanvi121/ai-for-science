# Do AI Language Models Recognize Emotion, or Just Surface Cues?

A science-fair experiment testing whether AI language models genuinely
recognize emotional meaning in text, or mainly rely on surface-level word
cues — by comparing model predictions against human labels on the public
**GoEmotions** benchmark and against a hand-written **custom challenge set**
of sarcasm, slang, emojis, figurative language, indirect wording, and mixed
emotions.


## TL;DR result

The hypothesis was **not confirmed**. The model tested (Llama 3.1 8B)
performed *worse* on the public GoEmotions benchmark (21.4% strict accuracy)
than on the hand-written custom challenge set (48.7% strict accuracy) — the
opposite of what was predicted. Full explanation in
[`CONCLUSIONS.md`](./CONCLUSIONS.md).

Two of three planned models (Gemini, Gemma) failed to return usable
predictions due to free-tier API rate limits — see "Where the Method Broke"
in [`CONCLUSIONS.md`](./CONCLUSIONS.md) for details.

## Repo contents

| File | What it is |
|---|---|
| `CONCLUSIONS.md` | Full methods, results, and "where the method broke" writeup |
| `VIDEO_SCRIPT.md` | Script used for the 10-minute video explanation |
| `custom_challenge_set.csv` | 150 hand-written examples (25 each: sarcasm, slang, emoji, figurative, indirect, mixed) with gold labels |
| `01_sample_goemotions.py` | Pulls + stratified-samples the GoEmotions test set |
| `02_run_models.py` | Sends every example through a fixed prompt to Gemini, Llama 3.1 8B, and Gemma 2 9B |
| `03_score_and_metrics.py` | Computes accuracy, precision, recall, macro-F1, per-emotion F1, confusion matrices |
| `04_error_analysis.py` | Breaks down error rate by language feature and text length |
| `requirements.txt` | Python dependencies |
| `goemotions_sample.csv` *(generated)* | Output of step 1 — the stratified GoEmotions sample |
| `predictions.csv` *(generated)* | Output of step 2 — raw model predictions |
| `metrics_summary.csv`, `per_emotion_f1_<model>.csv`, `confusion_matrix_<model>.csv` *(generated)* | Output of step 3 |
| `error_analysis.csv`, `misclassified_examples.csv` *(generated)* | Output of step 4 |

*(Files marked "generated" are only in this repo if the pipeline run that
produced them was still available to download — see the Limitations section
below for what was and wasn't recoverable.)*

## The 5-step approach

1. **Define labels** — fix the model to GoEmotions' 27 emotions + neutral
   (28 labels total), so it can't invent unsupported categories.
2. **Prompt models** — same fixed prompt, temperature = 0, sent to every
   model for every example.
3. **Score answers** — lenient (matches any human label) and strict
   (matches the primary label) scoring.
4. **Compute metrics** — accuracy, precision, recall, macro-F1, per-emotion
   F1, confusion matrices — split by GoEmotions vs. custom set.
5. **Analyze errors** — error rate by language feature (sarcasm, slang,
   emoji, figurative, indirect, mixed) and by text length.

## How to run it

Requires Python 3.10+, and free API keys for Google AI Studio (Gemini) and
Groq (Llama/Gemma) — neither requires a credit card.

```bash
pip install -r requirements.txt

export GOOGLE_API_KEY="your-gemini-key"   # aistudio.google.com/apikey
export GROQ_API_KEY="your-groq-key"       # console.groq.com/keys

python 01_sample_goemotions.py --n_per_label 4
python 02_run_models.py
python 03_score_and_metrics.py
python 04_error_analysis.py
```

Each script writes its own CSV(s) and reads from the previous step's output,
so you can inspect intermediate results at every stage. `02_run_models.py`
checkpoints after each model finishes and can resume if interrupted — safe
to re-run if it stops partway through.

**Note on pacing:** free-tier APIs rate-limit aggressively. The script paces
requests (4.5s/call for Gemini, 2.2s/call for Groq models) to stay under
published limits rather than bursting and retrying — expect the full run to
take ~35-40 minutes for ~260 examples across 3 models.

## Why free models instead of GPT/Claude

The original design planned to compare GPT, Gemini, and Claude. As a
student project with no API budget, this was swapped for three genuinely
free options: Gemini (Google AI Studio free tier), and Llama 3.1 8B +
Gemma 2 9B via Groq's free tier. This is a real scope limitation, documented
in `CONCLUSIONS.md` rather than hidden.

## Limitations (short version — full detail in CONCLUSIONS.md)

- Only 1 of 3 planned models produced usable results (Gemini, Gemma hit
  free-tier rate limits).
- GoEmotions was stratified-sampled (~112 examples) rather than scored on
  the full 5,427-example test set.
- Strict scoring used the first-listed label on GoEmotions' multi-label
  rows as "primary," which may not reflect true label salience.
- Some row-level output (specific misclassified examples) was lost to a
  Colab session disconnect before download; aggregate metrics were
  preserved from the run log.
