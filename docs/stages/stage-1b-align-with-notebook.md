# Stage 1b: align with the proof notebook

## Goal

Make the scaffolded code behave like the proof notebook (same chunking, prompt, retrieval
and generation settings) so later results can be compared with it. Fix the scaffold's bugs
along the way and set up linting, logging and tests.

## Starting point

The Stage 1 scaffold already worked end to end from the command line:
`loader.py` (read + chunk), `embeddings.py` (MiniLM), `store.py` (Chroma), `generator.py`
(Qwen), `pipeline.py` (ingest/ask), `cli.py`, and 4 chunking tests. But several settings
didn't match the notebook, and the prompt was the strict style that failed in the notebook.

## What changed

| File | Change | Why |
|---|---|---|
| `rag/config.py` | `TOP_K` 5 → 3, new `MAX_NEW_TOKENS=200` and `LOG_LEVEL` | Match notebook (D3, D7) |
| `rag/loader.py` | Stopped flattening whitespace; chunk start now also snaps to a word; empty-PDF-page warning; invalid UTF-8 replaced (with a warning) instead of silently dropped; unsupported file types rejected | D2; failures should be loud |
| `rag/prompts.py` (new) | Notebook's prompt + guardrail line, generic wording; bare-question baseline | D4, D5, D9, D11 |
| `rag/generator.py` | Takes ready-made messages; `pick_device()` chooses cuda / mps / cpu with float16 on GPUs; no `device_map="auto"`; logs device and dtype; greedy decoding with Qwen's sampling defaults cleared | D6, D12 |
| `rag/store.py` | Returns `Hit` objects (text, source, chunk, score) | Typed results instead of loose dicts |
| `rag/pipeline.py` | Returns an `Answer` with sources in rank order | D10 |
| `cli.py` | Numbered sources with scores; our logs at `LOG_LEVEL`, library logs only at WARNING | Easier to read |
| `tests/` | Chunking tests rewritten and extended; new prompt, pipeline and device-selection tests | 32 tests, none load a model |
| `pyproject.toml`, `requirements-dev.txt` | ruff and pytest config; dev tools split from runtime deps; `accelerate` removed | Code standards; D12 |
| `docs/` | This file, `decisions.md`, `proof-notebook.md` | So the project can be explained or handed over |

## How a question flows through the code now

1. `cli.py ask "How much debt does Netflix have?"` calls `pipeline.ask()`.
2. `store.search()` embeds the question with MiniLM and asks Chroma for the 3 closest chunks.
   Each comes back as a `Hit` with its cosine score.
3. `prompts.build_rag_messages()` puts those chunk texts, best first, into the prompt template.
4. `generator.generate()` runs Qwen on that prompt with greedy decoding, max 200 new tokens.
5. `pipeline.ask()` returns an `Answer`: the text plus the 3 hits. The CLI prints both.

With `--no-rag`, steps 2 and 3 are skipped and the model just gets the bare question.

## Debugging the first real run

The unit tests passed, but the first real run on a 16 GB Mac showed two problems:

```
WARNING accelerate.big_modeling: Some parameters are on the meta device because they were offloaded to the disk.
INFO rag.generator: Loaded Qwen/Qwen2.5-1.5B-Instruct on mps:0 (torch.float32)

The context does not contain this information.
```

**Problem 1: the model half-loaded to disk.** The code only used float16 on NVIDIA GPUs, so on
the Mac it loaded in float32 (about 6.2 GB). Memory was already tight (swap nearly full), so
`device_map="auto"` put some weights on disk. In a second shell the same load crashed with a
segmentation fault. Fix: choose the device explicitly and use float16 on any GPU (3.09 GB,
loads in about 5 s). See D12.

**Problem 2: the model refused a question it could answer.** Retrieval was fine: the top
chunk contained "$14.5 billion aggregate principal amount of senior notes". So the failure was
in generation. Four things differed from the notebook's working run: the guardrail line added
in this stage, greedy decoding, the generic prompt wording, and float32 on an Apple GPU. Tested
one at a time on the same chunks:

| Change tested | Result |
|---|---|
| Our prompt as written | Refused (greedy and 5 of 5 sampled seeds) |
| Remove only the guardrail line | Correct: $14.5B senior notes |
| Change only the wording back to "Netflix's annual report" | Still refused |
| Notebook's exact prompt, sampled with 5 seeds | Correct every time, no $17.5B sum |

So the guardrail line caused the refusal. Removing it entirely broke the WBD-fee question
instead, so four wordings were compared across all 8 benchmark questions. The final wording
("Quote each figure exactly as it appears in the context, and do not add figures together to
make a new total.") was the only one with no refusals and no wrong answers. Full details and
caveats are in D5.

Lesson: passing unit tests says nothing about model behaviour. Prompt changes need to be run
against the model, one variable at a time, which is what Stage 5 automates.

## How to run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

ruff format --check . && ruff check .
pytest                                   # ~5 s, no model downloads

python cli.py ingest data/corpus/        # first run downloads MiniLM (~90 MB)
python cli.py ask "How much debt does Netflix have?"          # first run downloads Qwen (~3 GB)
python cli.py ask "How much debt does Netflix have?" --no-rag
```

## Numbers to remember

- 800-character chunks, 150 overlap, 35 chunks from the corpus (notebook: 37).
- Top 3 chunks per question; up to 200 generated tokens.
- Notebook benchmark: baseline 0/8, RAG 5/8 correct + 2 partial + 1 wrong (the debt sum).
- After Stage 1b (one greedy run, manual check): 6/8 correct + 2 partial + 0 wrong.
  Not yet an independent result, since the guardrail wording was chosen on these questions.
- Qwen 1.5B in float16: 3.09 GB (float32: about 6.2 GB). One CLI question: about 15 s on an
  Apple GPU, including loading both models.
- 32 tests, all without models.

## Interview talking points

- **"Why did you change the prompt?"** The scaffold's strict "answer ONLY from context or say
  you don't know" prompt is exactly what made the notebook refuse the debt question even
  though the right chunk was retrieved. I went back to the notebook's fix (soft refusal plus a
  synonym hint) and added one line against combining figures, because the next failure was
  the model inventing a total.
- **"Why greedy decoding?"** So the evaluation is repeatable. With sampling, you can't tell
  whether a changed answer came from your change or from randomness.
- **"Why snap chunks to whitespace?"** Financial answers are numbers. Slicing at a fixed
  character can split `$45,183,036` across two chunks, and then neither chunk has the number.
- **"How do you test an LLM app without the LLM?"** Keep prompt building in pure functions and
  test those directly; replace retrieval and generation with fakes to test the flow. But that
  only tests the plumbing: my guardrail passed every unit test and still broke the main
  question, which I only found by running the real model.
- **"Tell me about a bug you debugged."** The model refused a question whose answer was in the
  top retrieved chunk. I checked retrieval first (fine), listed the four differences from the
  last working version, and tested them one at a time. One prompt line I had added was the
  cause. I then compared wordings across the whole benchmark rather than just the failing
  question, because removing the line fixed one question and broke another.
- **"Why not device_map='auto'?"** On a memory-tight laptop it silently offloaded weights to
  disk and then crashed. For a model that fits on one device, explicit placement is simpler
  and fails loudly.

## Known limitations and what's next

- The guardrail wording (D5) was chosen on the benchmark questions and the $17.5B sum didn't
  reproduce, so there's no proof yet that it prevents over-synthesis. Stage 5 measures it on
  more questions.
- Operating margin: every prompt variant answers "up about 3 points" and misses 29.5%. Check
  in Stage 5 whether that's a retrieval miss (hit@k) or a generation miss.
- Re-ingesting a changed file can leave stale chunks (fixed in Stage 3).
- Models are global singletons (replaced with injected interfaces in Stage 2).
- No API yet; CLI only (Stage 2).

## Handover notes

- Needs `transformers>=4.56` (the code uses the `dtype=` argument; older versions call it
  `torch_dtype`).
- The vector index persists in `chroma_db/`. Delete that folder to re-index from scratch.
- Check the `Loaded ... on <device> (<dtype>)` log line. `mps:0 (torch.float16)` or
  `cuda:0 (torch.float16)` is expected; `cpu (torch.float32)` means no GPU was found and
  generation will be much slower.
- `accelerate` is no longer a dependency. If you add `device_map=` back, you'll need it again.
- If a prompt change "should" work but the model refuses, first check what was retrieved
  (the sources printed with the answer) before touching the prompt.
