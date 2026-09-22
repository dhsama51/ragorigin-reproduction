"""
Phase 2 - 실험 2-1: 조기 종료(early termination)로 인한 미탐지 재현.

target question 5개(targeted 공격) 각각에 대해 corpus를 인위적으로 재배열:
  rank 1~5   : 해당 질문의 poisoned text 5개 (Match 확정 - i=1)
  rank 6~10  : corpus 안의 자연스러운 top benign 텍스트 (No match 유도 - i=2)
  rank 106   : 숨긴 poisoned text 1개 (약 22번째 segment, K=5 기준)
  나머지     : corpus의 자연스러운 순위 그대로

Eq.(4) 조건이 i=2(segment 2개)에서 곧바로 만족되는지, 그리고 숨긴 poisoned
text가 있는 22번째 segment에 도달하지 못하는지를 확인.

출력:
  results/early_termination_attack/early_termination_result.json
"""
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.expanduser("~/projects/ragorigin/data")
RESULTS_DIR = os.path.expanduser("~/projects/ragorigin/results/early_termination_attack")
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-small-v2"
K = 5
HIDDEN_POISON_RANK = 106  # 0-indexed 105 -> 22번째 segment(K=5 기준)
MAX_ITERS = 200

HIDDEN_TEMPLATE = "{question} Multiple independent sources confirm that {target_answer}."


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        corpus = json.load(f)
    corpus_embeds = np.load(os.path.join(DATA_DIR, "passage_embeds.npy"))
    with open(os.path.join(DATA_DIR, "poisoned_texts.json")) as f:
        poisoned_texts = json.load(f)
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

    results = []
    for event in targeted_events:
        question = event["question"]
        incorrect_response = event["incorrect_response"]
        target_answer = event["target_answer"]
        front_poisons = event["injected_poison_texts"]  # 5개

        hidden_text = HIDDEN_TEMPLATE.format(question=question, target_answer=target_answer)

        # ---- corpus 자체 순위 계산 (poison 없이) ----
        corpus_texts = [c["text"] for c in corpus]
        q_embed = retriever.encode(
            [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        corpus_sims = corpus_embeds @ q_embed
        corpus_natural_rank = np.argsort(-corpus_sims)  # corpus 내부 자연 순위 (id는 corpus index와 동일)

        top5_benign_ids = [int(i) for i in corpus_natural_rank[:5]]  # segment 2 용
        remaining_ids = [int(i) for i in corpus_natural_rank[5:]]    # 나머지 자연 순위

        # ---- 강제 순위 구성: [poison 5][benign top5][... 100개 ...][hidden][나머지] ----
        combined_texts = corpus_texts + front_poisons + [hidden_text]
        poison_virtual_ids = list(range(len(corpus), len(corpus) + 5))
        hidden_virtual_id = len(corpus) + 5
        is_poisoned_flags = {vid: True for vid in poison_virtual_ids}
        is_poisoned_flags[hidden_virtual_id] = True

        before_hidden = remaining_ids[:HIDDEN_POISON_RANK - 10]
        after_hidden = remaining_ids[HIDDEN_POISON_RANK - 10:]
        forced_order = poison_virtual_ids + top5_benign_ids + before_hidden + [hidden_virtual_id] + after_hidden
        hidden_segment_number = (forced_order.index(hidden_virtual_id) // K) + 1

        # ---- Algorithm 1 실행 (강제 순위 기준) ----
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
            n_poisoned_in_segment = sum(1 for vid in segment_ids if is_poisoned_flags.get(vid, False))

            if i == 1:
                response = incorrect_response
                match = True
            else:
                segment_contexts = [combined_texts[vid] for vid in segment_ids]
                prompt = build_rag_prompt(segment_contexts, question)
                response = generate(gen_model, tok, device, prompt)
                match = judge_match(gen_model, tok, device, response, incorrect_response)

            if match:
                cumulative_match += 1
            scope_ids.extend(segment_ids)
            terminate = (cumulative_match == i / 2)
            iterations.append({
                "i": i, "n_poisoned_in_segment": n_poisoned_in_segment,
                "match": match, "cumulative_match": cumulative_match, "terminate": terminate,
            })
            if terminate:
                break
            i += 1

        hidden_reached = hidden_virtual_id in scope_ids
        print(
            f"Q: {question[:45]:45s} | iteration {len(iterations)}회 종료 "
            f"| 숨긴 poison은 {hidden_segment_number}번째 segment "
            f"| scope에 포함됨={hidden_reached}"
        )

        results.append({
            "question": question,
            "hidden_poison_segment_number": hidden_segment_number,
            "iterations": iterations,
            "final_scope_size": len(scope_ids),
            "hidden_poison_reached": hidden_reached,
        })

    out_path = os.path.join(RESULTS_DIR, "early_termination_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_missed = sum(1 for r in results if not r["hidden_poison_reached"])
    avg_iters = sum(len(r["iterations"]) for r in results) / len(results)
    print("")
    print(f"saved to {out_path}")
    print(f"평균 iteration 수: {avg_iters:.1f}")
    print(f"숨긴 poison을 놓친 이벤트: {n_missed}/{len(results)}")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print("")
    print("총 실행 시간: " + f"{time.time() - _script_t0:.1f}" + "초")
