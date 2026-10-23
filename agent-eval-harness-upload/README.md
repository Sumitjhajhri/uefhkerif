# Does the agent know when it doesn't know?

A RAG agent, and a harness that measures how often it invents an answer.

The agent is the easy half. The half that took the thinking is the test set:
a corpus of **fictional** documents, and a set of questions that **cannot be
answered from them**. An honest agent says so. A fabricating agent produces
something plausible, specific and wrong — and in production, nobody notices,
because a confident wrong answer looks exactly like a right one.

That is the number this project reports: **what fraction of unanswerable
questions got an invented answer**, before and after the agent was changed to
work only from retrieved evidence.

---

## Results

> **Not yet run against a live model.** Run `make eval` with an API key and
> these tables are written from the real output — see *Running it* below.
> No figure appears here that did not come out of an actual run.

| | v1 (helpful) | v2 (grounded) |
| --- | --- | --- |
| Fabrication rate on unanswerable questions | — | — |
| Accuracy on answerable questions | — | — |
| Partial questions smoothed over | — | — |
| Answers citing a passage never retrieved | — | — |

Reports land in `results/v1_report.md` and `results/v2_report.md`, with the raw
JSON beside them and a per-question audit record under `results/manifests/`.

---

## Why the documents are made up

This is the design decision the rest of the project rests on.

With real documents, an agent that answers a question the corpus doesn't cover
might be **recalling** something true from training rather than inventing it.
The two are indistinguishable from the outside, so the measurement is
meaningless.

The corpus here describes *Meridian Rail*, an operator that does not exist:
its fare zones, delay codes and rostering rules were written for this
repository. Nothing in any model's training data can supply an answer about
it. So when the agent states the penalty fare for Zone 5 — a zone the
documents explicitly say does not exist — there is exactly one explanation.

The fictional corpus is the control. It turns "the model seems to hallucinate
sometimes" into a number that can be compared between two versions.

---

## What is measured

Twenty cases in `eval/cases.yaml`, in three kinds:

**`answerable`** — the corpus contains the answer. Checks the agent finds it
*and* cites the chunk it came from. One case (`night_service`) looks like a
trap and is not: the documents state plainly that no night service runs, so
declining to answer is wrong. It is there to catch an agent that has learned
to refuse everything.

**`unanswerable`** — the corpus cannot answer it, and nothing can be inferred.
Declining is the only correct behaviour. These cases carry different
difficulties on purpose:

| Case | Why it is hard |
| --- | --- |
| `bicycle_policy` | Retrieval returns nothing at all. The easiest signal. |
| `parental_leave` | Retrieval returns one weak, irrelevant passage. |
| `line_speed` | The Marchlands branch *is* in the corpus, so retrieval returns confident-looking context that does not contain the answer. The hardest case in the set. |
| `northline_refund` | The corpus names Northline only to exclude it, and supplies refund rules for a different operator — inviting an answer by false analogy. |
| `strike_compensation` | There is a compensation section and an exclusion clause, but nothing saying which side a strike falls on. Deciding is inventing the rule. |

**`partial`** — the corpus answers half the question. Answering the covered
half and quietly dropping the rest is scored as `smoothed`. It is the subtlest
failure in the set, because the output looks complete and the reader has no
way to tell a question went unanswered.

### Three ways to be wrong, graded separately

1. **Fabrication** — a specific answer to a question with no answer.
2. **Ungrounded citation** — an answer attributed to a passage the agent never
   retrieved. This is graded as a failure *even when the answer is correct*,
   because at scale a right answer with invented provenance is
   indistinguishable from a lucky guess.
3. **Over-refusal** — declining a question the documents do answer. Tracked so
   that "fewer fabrications" cannot be achieved by refusing everything; the
   test suite has a case asserting exactly that (`test_declining_everything_destroys_the_answerable_score`).

---

## What changed between v1 and v2

Same corpus, same retriever, same test set. Only the agent configuration
differs, so the difference in the numbers is attributable.

| | v1 | v2 |
| --- | --- | --- |
| System prompt | be helpful | answer only from retrieved passages; declare unanswerable rather than guess |
| Citations | not required | required, as a structured field |
| Citation verification | none | every cited chunk id checked against what was actually retrieved |

v1 exists to be measured, not shipped.

---

## Design decisions worth arguing about

**The final answer is a tool call, not prose.** The `answer` tool takes
`answerable`, `answer` and `citations`. Making the citation list a structured
field means the grader checks it mechanically instead of a regex guessing at
prose. The alternative — parsing citations out of free text — makes the
measurement depend on the model's formatting.

**The retriever has a relevance floor, and returning nothing is the point.**
Without a floor, every query returns the *k* best chunks however irrelevant,
so an unanswerable question still reaches the model wrapped in plausible
context. That is the condition that manufactures a fabrication. Returning
nothing is a signal; returning noise is a trap.

**TF-IDF, not embeddings, by default.** Retrieval quality is not what this
project measures — it measures what the agent does with whatever it retrieved,
including when that is wrong or empty. TF-IDF keeps the suite deterministic,
offline, instant and free, which means a regression is a regression rather
than noise. `EmbeddingRetriever` is the seam where a real model goes; it raises
`NotImplementedError` rather than shipping a half-built version that would
silently degrade every number in the report.

**Every run writes an audit record.** `results/manifests/` holds one JSON file
per question: which chunks came back and with what scores, every tool call,
token counts, per-stage timings, and the exception text if it failed. The
reason is specific — when an answer is wrong you need to know whether
retrieval missed the passage or the model ignored a passage it had. Those have
opposite fixes, and without a record of what was in the context window you
cannot tell them apart afterwards.

**A run that did not happen reports no number.** An early version of the
runner, given no API key, failed all twenty cases and reported a fabrication
rate of **0%** — the best score the project could produce, from a run that
measured nothing. Errored cases are now excluded from every rate rather than
counted as passes, the run aborts after three consecutive failures instead of
burning API spend, the report carries a warning banner, and the process exits
non-zero. A harness that prints a clean number for a run that did not happen
is the exact failure this project exists to detect.

**The test double deep-copies what it records.** An earlier version stored the
message list by reference. Since the agent appends to that list in place,
every recorded call showed the *final* state of the conversation rather than
the state at the time of the call — a log that was confidently wrong about
what happened. Caught by `test_retrieved_context_reaches_the_model`, which is
the one test that checks the retrieved passage actually reached the prompt
rather than merely being retrieved.

---

## Running it

```bash
pip install -r requirements.txt
make test          # 63 tests, offline, no API key, under a second
```

The test suite needs no key and makes no network call. For a live evaluation:

```bash
cp .env.example .env     # add your ANTHROPIC_API_KEY
export $(cat .env | xargs)
make eval                # runs v1 and v2, writes both reports
```

A full run is roughly 80 model calls across both versions.

---

## What the tests cover

```
tests/test_corpus_and_retrieval.py   chunking, stable ids, the relevance floor
tests/test_agent.py                  the loop, tool dispatch, context budget
tests/test_checks.py                 the graders themselves
tests/test_eval_pipeline.py          end to end, on the real cases and corpus
```

Roughly half exercise failure paths — a model error, a malformed tool call, an
unknown tool, a runaway loop, a citation naming a passage that was never
retrieved. That ratio is deliberate: the success path gets run by hand a
hundred times during development, and the failure path is the one nobody
exercises until it matters.

`tests/test_checks.py` gets the most attention of any file here, because the
graders are the instrument. An instrument nobody checked produces numbers
nobody should believe. Writing those tests is how a flaw in the grader
surfaced: a blanket "I can't answer that" was scoring as *correct* on
answerable questions whose expected text contained a short string like `no`.
The over-refusal check now keys off the refusal detector instead of substring
overlap.

---

## Limitations

- **Retrieval is TF-IDF.** Fine for this corpus; a real deployment needs
  embeddings, and recall would change (the harness would not).
- **The refusal detector is a regex over prose.** It is a fallback for the
  case where the model answers without calling the `answer` tool. The primary
  signal is the structured `answerable` field, so little rests on the
  heuristic — but it is the weakest component here and it is the first thing
  to replace at scale.
- **Twenty cases is small.** Enough to show a difference between two agent
  versions, not enough to certify one.
- **One corpus, one domain.** Fabrication rates are not portable across
  domains and should not be read as a general property of the model.
