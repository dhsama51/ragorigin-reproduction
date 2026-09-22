"""
Phase 2 - 실험 2-2: poisoned text 위치별 탐지율 coverage curve.

target question 5개(targeted) 각각에 대해, poisoned text 1개(해당 이벤트의
poisoned text 중 1개만)를 corpus 자연 순위 안의 서로 다른 rank
[K-1, K, K+1, 2K-1, 2K, 2K+1] = [4,5,6,9,10,11] (1-indexed)에 삽입하고
Algorithm 1을 (i=1도 매번 실제로 생성/판정하여) 끝까지 실행.

Eq.(4)의 수학적 성질상, poison이 segment1 또는 segment2(rank<=10)에 있으면
그 세그먼트에서 1회 match가 나온 뒤 다음 세그먼트가 no-match면 i=2에서
반드시 종료됨. poison이 segment3(rank 11~15)에 있으면, match가 i=3에서야
나오는데 그 뒤로 추가 match가 없으면 누적합이 1에서 멈춘 채 i/2만 계속
커져서 조건을 영원히 못 맞추는(MAX_ITERS까지 도달) 구조적 현상이 예상됨.
이게 바로 논문이 지적한 [K-1, 2K-1] 경계의 실질적 의미를 정량화하는 지점.

출력:
  results/position_coverage_curve/coverage_curve_result.json
"""
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.expanduser("~/projects/ragorigin/data")
RESULTS_DIR = os.path.expanduser("~/projects/ragorigin/results/position_coverage_curve")
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-small-v2"
K = 5
POSITIONS = [K - 1, K, K + 1, 2 * K - 1, 2 * K, 2 * K + 1]  # 1-indexed rank: 4,5,6,9,10,11
MAX_ITERS = 30  # position 11류는 구조적으로 종료 안 될 수 있어 상한선으로 컷


def build_rag_prompt(contexts, question):
    ctx = "\n".join(f"- {c}" for c in contexts)
    return (
        "Answer the question using only the information in the context below. "
        "Your answer should be short and concise. If you cannot find the answer "
        "to the question, just say \"I don't know\".\n"
        f"Context:\n{ctx}\nQuestion: {question}\nAnswer:"
    )


def generate(model, tok, device, prompt, max_new_tokens=30):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge_match(model, tok, device, response, reference_response):
    if response.strip().lower().startswith("i don't know") or \
       response.strip().lower().startswith("i do not know"):
        return False
    prompt = (
        "Does the RESPONSE convey the same meaning as the REFERENCE, even if worded "
        "differently? Answer with only one word: Yes or No.\n"
        f"RESPONSE: {response}\nREFERENCE: {reference_response}\nAnswer:"
    )
    out = generate(model, tok, device, prompt, max_new_tokens=3)
    return out.strip().lower().startswith("yes")


def run_algorithm1(forced_order, combined_texts, is_poisoned_flags, question,
                    incorrect_response, gen_model, tok, device):
    iterations = []
    cumulative_match = 0
    scope_ids = []
    i = 1
    while i <= MAX_ITERS:
        start = (i - 1) * K
        end = i * K
        if start >= len(forced_order):
            break
        segment_ids = forced_order[start:end]
        n_poisoned = sum(1 for vid in segment_ids if is_poisoned_flags.get(vid, False))

        segment_contexts = [combined_texts[vid] for vid in segment_ids]
        prompt = build_rag_prompt(segment_contexts, question)
        response = generate(gen_model, tok, device, prompt)
        match = judge_match(gen_model, tok, device, response, incorrect_response)

        if match:
            cumulative_match += 1
        scope_ids.extend(segment_ids)
        terminate = (cumulative_match == i / 2)
        iterations.append({"i": i, "n_poisoned_in_segment": n_poisoned, "match": match,
                            "cumulative_match": cumulative_match, "terminate": terminate})
        if terminate:
            break
        i += 1
    return iterations, scope_ids


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        corpus = json.load(f)
    corpus_embeds = np.load(os.path.join(DATA_DIR, "passage_embeds.npy"))
    with open(os.path.join(DATA_DIR, "misgeneration_events.json")) as f:
        events = json.load(f)
    targeted_events = [e for e in events if e["attack_type"] == "targeted"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading models ({device}) ...")
    retriever = SentenceTransformer(RETRIEVER_MODEL)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    gen_model.eval()

    corpus_texts = [c["text"] for c in corpus]
    results = []

    for event in targeted_events:
        question = event["question"]
        incorrect_response = event["incorrect_response"]
        poison_text = event["injected_poison_texts"][0]  # 단 1개만 사용

        q_embed = retriever.encode(
            [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        corpus_sims = corpus_embeds @ q_embed
        natural_rank = [int(i) for i in np.argsort(-corpus_sims)]  # corpus 내부 id 순서

        combined_texts = corpus_texts + [poison_text]
        poison_vid = len(corpus)  # 가상 id

        for pos in POSITIONS:  # 1-indexed rank
            forced_order = natural_rank[: pos - 1] + [poison_vid] + natural_rank[pos - 1:]
            is_poisoned_flags = {poison_vid: True}

            iterations, scope_ids = run_algorithm1(
                forced_order, combined_texts, is_poisoned_flags, question,
                incorrect_response, gen_model, tok, device,
            )
            included = poison_vid in scope_ids
            terminated = iterations[-1]["terminate"] if iterations else False
            print(
                f"Q: {question[:35]:35s} | rank={pos:3d} | iteration {len(iterations)}회 "
                f"| 종료됨={terminated} | scope에 포함됨={included}"
            )
            results.append({
                "question": question, "position": pos,
                "n_iterations": len(iterations), "terminated": terminated,
                "final_scope_size": len(scope_ids), "poison_included_in_scope": included,
            })

    out_path = os.path.join(RESULTS_DIR, "coverage_curve_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("")
    print(f"saved to {out_path}")
    print("위치별 탐지율(5개 질문 평균):")
    for pos in POSITIONS:
        subset = [r for r in results if r["position"] == pos]
        rate = sum(r["poison_included_in_scope"] for r in subset) / len(subset)
        avg_iter = sum(r["n_iterations"] for r in subset) / len(subset)
        print(f"  rank={pos:3d}: 탐지율={rate:.0%}  평균 iteration={avg_iter:.1f}")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print("")
    print("총 실행 시간: " + f"{time.time() - _script_t0:.1f}" + "초")
