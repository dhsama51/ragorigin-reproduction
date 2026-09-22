# RAGOrigin Edge Case Reproduction (Lightweight Replication)

A lightweight replication and edge-case investigation of [IEEE S&P'26]
*"Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in
RAG"* (RAGOrigin).

**The goal is not to reproduce SOTA-level numbers, but to replicate, on a
low-cost GPU with lightweight models, that Algorithm 1 (attribution scope
narrowing) and the Responsibility Score (Eq. 5-8) behave as described in
the paper, and to directly measure boundary conditions the paper does not
explicitly address.**

## Key Findings

### Early termination causes missed detection

Forcing the first segment to be poisoned text (Match) and the second
segment to be natural benign text (No match) causes the termination
condition to be satisfied immediately after the second segment, ending
scope narrowing. A separate poisoned text hidden much deeper in the
corpus (22 segments in) was missed in 5/5 cases — it never entered the
final scope and was therefore undetectable.

### Position-based coverage curve

A single poisoned text was inserted at rank [K-1, K, K+1, 2K-1, 2K, 2K+1]
(= 4, 5, 6, 9, 10, 11). **At rank = 2K+1 (= 11), all 5/5 cases failed to
terminate normally even after the maximum number of iterations (30).**
The termination condition requires the cumulative Match/No-match counts to
become equal, which is only mathematically possible at an even iteration
count. When the first Match occurs at an odd-numbered iteration and no
further Match ever occurs afterward, that gap can never close. The raw
log shows "included in final scope = True" for these cases, but this is
not a successful, well-terminated detection — it is the algorithm being
forced to scan almost the entire corpus up to the iteration cap. This
should be read as **a collapse of scope narrowing's efficiency**, not as
a detection success.

### Judgment LLM self-judgment bias

For the same set of events, the judgment LLM was swapped between Qwen
(self-judging its own generations) and Llama-3.2-1B (separate judge).
**Final DACC was identical (1.00) in both conditions** — once a text
enters the final scope, both configurations separate poisoned from benign
perfectly. However, **the average number of iterations differed sharply:
2.40 (self) vs. 4.00 (separate)**. Llama-3.2-1B judged Match more
strictly, requiring more segments to be scanned before the termination
condition was satisfied. This difference should be interpreted as "how
lenient the match judgment is affects the algorithm's efficiency (when it
terminates)," not as a difference in the ability to distinguish poisoned
text. (Whether this reflects genuine self-preference bias or simply a
general difference in judgment strictness between the two models cannot
be fully disentangled by this design alone.)

### Adaptive termination-forcing attack

Under the paper's strong-attacker assumption (the attacker has complete
knowledge of the attribution system's structure), the first segment was
set to poisoned (Match) and the second to benign (No match) to
deliberately force early termination, while the real payload was hidden
at ranks 200 / 500 / 1000 / 2000. **All 20/20 cases (5 questions x 4
depths) evaded detection**, regardless of how deep the payload was
hidden.

## Limitations

- **Poisoned texts were constructed with no optimization and are highly
  overt** (they repeat the question verbatim and explicitly state the
  wrong answer or an override instruction). Unlike the paper's
  gradient-based attacks or adaptive attacks that directly minimize the
  SC/GC scores, there was no attempt to evade detection at all. This is
  likely why a separate, still-in-progress event-count stability
  experiment showed DACC = 1.00 across all 10 events with zero variance
  — this does not mean RAGOrigin is flawless, but that this project's
  attack setup only covered trivially easy cases.
- Only 5 target questions and a 10,000-text corpus (vs. millions in the
  paper) — an extreme scale-down. Results are not directly comparable to
  the paper's SOTA numbers.
- Only 2 attack types implemented (vs. 9 in the paper).

## Default Paper Conditions vs. This Project

| Item | Paper default | This project |
|---|---|---|
| Dataset | 5 datasets (NQ/HotpotQA/MS-MARCO/BoolQ/SQuAD), large-scale (SQuAD alone has ~5.23M passages) | SQuAD only, 10,000 passages |
| Retriever | E5-base-v2, cosine similarity | e5-small-v2, cosine similarity (lightweight, same family) |
| K (scope-narrowing segment size) | 5 | 5 (same) |
| M (poisoned texts per question) | 5 | 5 (same) |
| RAG response generation LLM | GPT-4o-mini (default) | Qwen2.5-1.5B-Instruct |
| Proxy LLM (ES/SC/GC computation) | Llama-3.1-8B (default) | Qwen2.5-1.5B-Instruct (same model as the generation LLM — a combination not present in the paper) |
| Judgment LLM (match judging) | GPT-4o-mini (default) | baseline: Qwen2.5-1.5B-Instruct (self) / comparison: Llama-3.2-1B-Instruct (separate) |
| Attack types | 9 SOTA poisoning attacks | 2 custom implementations (targeted: PoisonedRAG-style explicit wrong-answer statement / dos: Jamming-style "ignore context" instruction) |
| Target answer construction | LLM-generated random answer different from the correct one | Same approach (generated by Qwen, retried until different from the correct answer) |
| Number of target questions | Up to 100 successful instances collected per (attack, dataset) pair | 5 (chosen based on the project's time budget) |
| Threshold determination | K-means (k=2) on RS(u) | Same |

Because both the corpus scale (5.23M -> 10K) and the model scale
(GPT-4o-mini/Llama-8B -> ~1.5B) were reduced drastically, directly
comparing this project's DACC/FPR/FNR numbers to the paper's is not
meaningful. The purpose here is structural verification and boundary-case
observation, not SOTA reproduction.

## Relationship to the Author's Code

The author's official implementation
(`external/RAG-Responsibility-Attribution`) was consulted to cross-check
the scope-narrowing, RS-computation, and K-means-threshold logic.
Differences found and corrected during this cross-check:

1. **Match-judging target**: the initial implementation compared against
   the attacker's target answer (t), but both the author's code and the
   paper's definition correctly compare against the actually observed
   incorrect response (r) — this was fixed.
2. **"I don't know" handling**: the author's prompt explicitly instructs
   "say I don't know if you can't find the answer," and any such response
   is automatically treated as No-match — this was matched.
3. **Skipping recomputation of the first scope-narrowing segment**: since
   that segment's top-K is identical to the one already used when
   constructing the misgeneration event, it is fixed to Match without
   regenerating, exactly as the author's code does (saves one LLM call).

The RS computation method (HF `loss`-based vs. directly gathering
log-softmax values) is mathematically equivalent, so it was left as
originally implemented.

## Pipeline Structure

```
scripts/
  benchmark_llm_latency.py         # measures LLM call latency, used for time-budget planning

  setup/
    01_build_corpus.py               # builds a 10,000-text corpus + 5 target questions + embeddings
    02_generate_poisoned_texts.py    # generates poisoned texts (targeted/dos x M=5 x 5 questions)
    03_build_misgeneration_events.py # verifies actual attack success -> builds (q, r) events
    03b_retry_failed_dos_attacks.py  # retries failed dos scenarios (to confirm genuine failure)
    03c_replace_failed_questions.py  # replaces questions that still fail after retry
    04_scope_narrowing_sanity_check.py       # verifies scope narrowing behaves correctly
    05_responsibility_score_and_threshold.py # verifies RS(u) + K-means threshold behave correctly

  early_termination_attack.py      # early-termination missed-detection experiment
  position_coverage_curve.py       # position-based detection-rate experiment
  self_judgment_bias.py            # judgment LLM self-judgment bias experiment
  adaptive_termination_attack.py   # adaptive termination-forcing evasion attack experiment
```

## DoS Attack ASR Observation (`results/observation_dos_asr_note.md`)

Among the initial 5 questions, the dos attack failed on 2 of them
(when the top-K results were not fully dominated by poisoned text,
Qwen2.5-1.5B tended to favor the real information over the "ignore the
context" instruction). This is lower than the paper's Jamming ASR on
SQuAD (0.85) — our result was 3/5 = 0.60. Retrying with more forceful
instruction text still failed deterministically (greedy decoding), so
rather than force the text to be stronger, the two failing questions
were replaced with different questions to finalize the 5 target
questions. See the observation note for details.

## Pipeline Sanity-Check Results

Across all 10 events (5 questions x 2 attack types), poisoned-text
recall within the final scope was 100%, scope narrowing terminated after
an average of 2.4 iterations, and the RS-based K-means threshold achieved
DACC = 1.00. This confirms scope narrowing and RS computation behave as
described in the paper.

## In Progress (Not Yet Uploaded)

The event-count DACC/ASR stability experiment still needs a larger
baseline pool and is excluded from this upload for now.