"""
Phase 1 - Step 4: Algorithm 1 (Attribution scope narrowing) 구현.

논문 5.2절/Algorithm 1을 그대로 따름:
  1. 미리 만들어진 10개 misgeneration event 각각에 대해,
     오염된 DB(corpus 1만개 + 해당 시나리오 poisoned text M개)를 재구성.
  2. 질문 q와 오염된 DB 전체의 코사인 유사도로 내림차순 정렬(D_rank).
  3. i=1부터 시작, D_rank[(i-1)*K+1 : i*K]를 순서대로 scope U에 추가하며
     judgment LLM으로 이 subset이 misgeneration event의 응답 r과
     match하는지 판정.
  4. Eq.(4): sum_{j=1}^{i} I(match_j) == i/2 를 만족하는 순간 종료.
     (첫 iteration부터 확인, i/2가 정수가 아니면 자동으로 불만족)
  5. 각 iteration의 (subset 내 poisoned 개수, match 여부, 누적 match,
     종료조건) 로그를 전부 남김 - 이게 Phase 2 실험의 원자료가 됨.

출력:
  results/sanity_check/scope_narrowing_log.json : [
    {"question":.., "attack_type":.., "iterations":[
        {"i":1, "segment_ids":[...], "n_poisoned_in_segment":int,
         "match":bool, "cumulative_match":int, "terminate":bool}, ...
     ], "final_scope_ids":[...], "final_scope_size":int,
     "n_poisoned_in_scope":int, "n_poisoned_total":int}
  ]
"""
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.expanduser("~/projects/ragorigin/data")
RESULTS_DIR = os.path.expanduser("~/projects/ragorigin/results/sanity_check")
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-small-v2"
K = 5
MAX_ITERS = 200  # 안전장치: corpus+poison 전체를 다 돌아도 끝나야 함 (10050/5=2010, 여유있게)


def generate(model, tok, device, prompt, max_new_tokens=30):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def judge_match(model, tok, device, response, reference_response):
    # 저자 코드(check())와 동일한 규칙: 응답이 "I don't know"면 무조건 No match
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


def build_rag_prompt(contexts, question):
    ctx = "\n".join(f"- {c}" for c in contexts)
    return (
        "Answer the question using only the information in the context below. "
        "Your answer should be short and concise. If you cannot find the answer "
        "to the question, just say \"I don't know\".\n"
        f"Context:\n{ctx}\nQuestion: {question}\nAnswer:"
    )


def narrow_scope(event, corpus, corpus_embeds, retriever, gen_model, tok, device):
    question = event["question"]
    incorrect_response = event["incorrect_response"]  # Eq.(3)/(4): t가 아니라 r과 비교
    poison_texts = event["injected_poison_texts"]

    poison_embeds = retriever.encode(
        [f"passage: {t}" for t in poison_texts],
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    combined_embeds = np.vstack([corpus_embeds, poison_embeds])
    combined_texts = [c["text"] for c in corpus] + poison_texts
    is_poisoned_flags = [False] * len(corpus) + [True] * len(poison_texts)

    q_embed = retriever.encode(
        [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]
    sims = combined_embeds @ q_embed
    ranked_idx = np.argsort(-sims)  # D_rank : 유사도 내림차순 전체 인덱스

    iterations = []
    cumulative_match = 0
    scope_ids = []
    i = 1
    while i <= MAX_ITERS:
        start = (i - 1) * K
        end = i * K
        if start >= len(ranked_idx):
            break  # 전체 DB를 다 썼는데도 종료 안 됨 (이례적 상황)
        segment_idx = ranked_idx[start:end]
        segment_ids = [int(x) for x in segment_idx]
        n_poisoned_in_segment = sum(1 for idx in segment_idx if is_poisoned_flags[idx])

        if i == 1:
            # 저자 코드와 동일: i=1의 top-K는 misgeneration event를 만들 때 이미
            # 사용한 top-K와 정확히 같으므로, incorrect_response를 그대로 재사용하고
            # match는 재호출 없이 True로 고정 (LLM 호출 절약 + 저자 구현과 정합성 유지)
            response = incorrect_response
            match = True
        else:
            segment_contexts = [combined_texts[idx] for idx in segment_idx]
            prompt = build_rag_prompt(segment_contexts, question)
            response = generate(gen_model, tok, device, prompt)
            match = judge_match(gen_model, tok, device, response, incorrect_response)

        if match:
            cumulative_match += 1
        scope_ids.extend(segment_ids)

        terminate = (cumulative_match == i / 2)
        iterations.append({
            "i": i,
            "segment_ids": segment_ids,
            "n_poisoned_in_segment": n_poisoned_in_segment,
            "match": match,
            "cumulative_match": cumulative_match,
            "terminate": terminate,
        })

        if terminate:
            break
        i += 1

    n_poisoned_in_scope = sum(1 for idx in scope_ids if is_poisoned_flags[idx])
    return {
        "question": question,
        "attack_type": event["attack_type"],
        "iterations": iterations,
        "final_scope_ids": scope_ids,
        "final_scope_size": len(scope_ids),
        "n_poisoned_in_scope": n_poisoned_in_scope,
        "n_poisoned_total": len(poison_texts),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        corpus = json.load(f)
    corpus_embeds = np.load(os.path.join(DATA_DIR, "passage_embeds.npy"))
    with open(os.path.join(DATA_DIR, "misgeneration_events.json")) as f:
        events = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading models ({device}) ...")
    retriever = SentenceTransformer(RETRIEVER_MODEL)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    gen_model.eval()

    logs = []
    for event in events:
        result = narrow_scope(event, corpus, corpus_embeds, retriever, gen_model, tok, device)
        logs.append(result)
        print(
            f"[{result['attack_type']:8s}] Q: {result['question'][:45]:45s} "
            f"| iteration {len(result['iterations'])}회 종료 "
            f"| scope 크기 {result['final_scope_size']} "
            f"| scope 내 poisoned {result['n_poisoned_in_scope']}/{result['n_poisoned_total']}"
        )

    out_path = os.path.join(RESULTS_DIR, "scope_narrowing_log.json")
    with open(out_path, "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print("")
    print("saved to " + out_path)
    avg_iters = sum(len(l["iterations"]) for l in logs) / len(logs)
    avg_recall = sum(
        l["n_poisoned_in_scope"] / l["n_poisoned_total"] for l in logs
    ) / len(logs)
    print(f"평균 iteration 수: {avg_iters:.1f}")
    print(f"평균 scope 내 poisoned 텍스트 재현율: {avg_recall:.2%}")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print("")
    print("총 실행 시간: " + f"{time.time() - _script_t0:.1f}" + "초")
