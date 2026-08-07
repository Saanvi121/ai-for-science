# Methods

**Scientific question.** Do AI language models recognize emotional meaning
in short text, or mainly rely on surface-level cues? We compared model
predictions against human labels on the GoEmotions benchmark and against a
custom challenge set targeting sarcasm, slang, emojis, figurative language,
indirect wording, and mixed emotions.

**Labels.** All models were restricted to the fixed GoEmotions label set: 27
emotion categories plus neutral (28 labels total). This prevents models from
inventing unsupported labels and keeps outputs directly comparable.

**Data.**
- *GoEmotions*: the public GoEmotions test split (58k Reddit comments, 27
  emotions + neutral, multi-label). We stratified-sampled ~4 examples per
  label (112 total) from the official test split, rather than scoring all
  5,427 test rows, to fit the available time.
- *Custom challenge set*: 150 newly written examples (25 each across
  sarcasm, slang, emojis, figurative language, indirect wording, and mixed
  emotions), hand-labeled against the same 28-label set. These examples
  cannot be memorized from a public dataset, so they test generalization
  rather than benchmark recall.

**Prompting.** Every example — from both sources — was sent through the
*same fixed prompt*, asking the model to return exactly one emotion label
from the list, at temperature = 0, to keep outputs deterministic and
comparable across models.

**Models tested.** Three free-access models were selected due to project
budget constraints (student project, no paid API access): Gemini
(gemini-2.0-flash, Google AI Studio free tier), Llama 3.1 8B (Groq free
tier), and Gemma 2 9B (Groq free tier). This substitutes for paid
frontier APIs (GPT, Claude) that require billing even for light use.

**Scoring.** Two accuracy definitions were used: *lenient* (a prediction
counts as correct if it matches any of the human-assigned labels for that
example) and *strict* (a prediction must match the primary/first-listed
label). Macro precision, recall, and F1 were computed using strict scoring.
Metrics were computed overall and separately for the GoEmotions subset vs.
the custom challenge subset, to directly measure any generalization gap.

**Error analysis.** Error rate was broken down by the custom set's language
feature tag (sarcasm, slang, emoji, figurative, indirect, mixed) and by text
length, to see whether mistakes cluster around specific language features
(systematic) or scatter evenly (random).

# Conclusions

## Hypothesis (recap)
We expected models to be strongest on clear, literal emotions and weaker on
subtle cases (sarcasm, ambiguity, mixed feelings) — and expected a large gap
between GoEmotions and our custom challenge set to suggest that public
benchmark scores overestimate real-world robustness.

## What we actually found
Of the three planned models, only **Llama 3.1 8B (via Groq)** returned usable
predictions on the full 262-example set (112 GoEmotions + 150 custom
challenge). Gemini and Gemma both failed on every example due to free-tier
API rate limiting (see "Where the method broke" below), so this analysis is
based on Llama 3.1 8B only.

**Overall accuracy:** 42.0% lenient (prediction matches any human-assigned
label) / 37.0% strict (prediction matches the primary label).

**The result reverses our hypothesis.** Llama performed *worse* on the public
GoEmotions benchmark than on our hand-written custom challenge set — the
opposite of what we predicted:

| Subset | Lenient accuracy | Strict accuracy |
|---|---|---|
| GoEmotions (n=112) | 25.9% | 21.4% |
| Custom challenge (n=150) | 54.0% | 48.7% |

Breaking the custom set down by the language feature it was designed to test
(error rate — lower is better):

| Feature | Error rate |
|---|---|
| Emoji | 32.0% |
| Indirect wording | 48.0% |
| Figurative language | 52.0% |
| Slang | 56.0% |
| Sarcasm | 60.0% |
| Mixed emotions | 60.0% |
| **GoEmotions (comparison)** | **78.6%** |

Every custom-challenge category — including sarcasm and mixed emotions, which
we expected to be hardest — had a *lower* error rate than GoEmotions itself.

**Why GoEmotions was harder, not easier:**
- GoEmotions comments are real Reddit text pulled out of their original
  thread — the model often lacks the context a human annotator had when
  labeling them, whereas our custom examples were written to be
  self-contained and clearly signal one target emotion (even when using
  sarcasm or slang to do it).
- GoEmotions uses 27 emotion categories with real semantic overlap (e.g.
  annoyance vs. disapproval vs. anger); the original dataset's own
  inter-annotator agreement is imperfect for exactly this reason, so a
  "reasonable" model answer can still miss the one specific label a human
  happened to choose.
- We also scored GoEmotions' strict metric against the *first* label listed
  per example, which is not necessarily the most salient one for
  multi-label rows — a possible scoring artifact worth flagging rather than
  hiding.
- Text length compounded this: error rate rose from 51.5% (short, <8 words)
  to 92.0% (long, >20 words), and GoEmotions comments trend longer and more
  rambling than our tightly-written challenge examples.

**Bottom line:** rather than confirming that benchmark scores overstate
real-world robustness, this run suggests the opposite risk — that a model's
GoEmotions score may *understate* its ability to read emotion from clearly
written text, because GoEmotions' difficulty comes more from missing context
and fine-grained/overlapping label choices than from the surface-level
complexity (sarcasm, slang, emojis) we set out to test. That's a genuinely
interesting finding worth stating plainly, even though it isn't the one we
predicted.

## Where the method broke
- **Gemini and Gemma (2 of 3 planned models) returned zero usable
  predictions.** Both hit persistent rate-limit / quota errors on their free
  API tiers, even after adding retry logic and pacing calls under the
  published per-minute limits. Average per-call latency (15.4s for Gemini,
  7.6s for Gemma, vs. a target of 4.5s/2.2s) shows requests were repeatedly
  failing and retrying rather than completing normally.
- **Practical implication:** the planned three-model comparison could not be
  completed within the available time and budget (student project, no paid
  API access). The experiment as run is a single-model case study rather
  than a cross-model comparison, and that scope limitation should be stated
  up front rather than implied.
- **Scoring limitation:** GoEmotions multi-label rows were reduced to a
  single "primary" label (the first one listed) for strict scoring, which
  may not reflect true label salience and could be deflating GoEmotions
  accuracy independent of the model's real performance.
- **Sample size:** GoEmotions was stratified-sampled to ~4 examples per
  label (112 total) rather than the full 5,427-example test set, for time
  reasons — results on the full test set could differ.
- **Row-level output was lost to a Colab session disconnect.** The free-tier
  Colab runtime disconnected before `predictions.csv`, the per-emotion
  confusion matrices, and `misclassified_examples.csv` were downloaded
  locally. Aggregate metrics survived (captured from the run log before
  disconnection), but specific misclassified examples could not be
  recovered. This is a direct consequence of running the full pipeline on
  free-tier infrastructure with no persistent storage — a real constraint
  of doing this project at zero cost within a fixed time window, not a
  flaw in the experimental design itself.