"""
Phase 1 - Step 5: Responsibility Score 계산 + K-means threshold 판정.

각 misgeneration event의 최종 scope(04번 결과) 안에 있는 모든 텍스트 u에 대해:
  ES(u) = cos_sim(q, u)                                   (Eq.5)
  SC(u) = avg log P(query 토큰 | Prompt1(context=u, query=q))   (Eq.6)
  GC(u) = avg log P(r 토큰 | Prompt2(context=u, query=q, answer=r)) (Eq.7)
  RS(u) = mean(zscore(ES), zscore(SC), zscore(GC))          (Eq.8, scope 내 z-score)

이후 scope 내 RS(u) 값들에 K-means(k=2)를 적용해 평균 RS가 높은 군집을
poisoned으로 판정(5.4절). 실제 정답(is_poisoned)과 비교해 이벤트별
DACC/FPR/FNR을 계산.

출력:
  results/sanity_check/rs_scores_and_threshold.json : [
    {"question":.., "attack_type":..,
     "scope":[{"id":int, "is_poisoned":bool, "ES":.., "SC":.., "GC":..,
               "RS":.., "predicted_poisoned":bool}, ...],
     "dacc":.., "fpr":.., "fnr":..}
  ]
"""
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = os.path.expanduser("~/projects/ragorigin/data")
RESULTS_DIR = os.path.expanduser("~/projects/ragorigin/results/sanity_check")
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-small-v2"


def zscore(values):
    arr = np.asarray(values, dtype=np.float64)
    mu, sigma = arr.mean(), arr.std() + 1e-8
    return ((arr - mu) / sigma).tolist()


def avg_logprob_of_suffix(model, tok, device, prefix_text, full_text):
    """full_text = prefix_text + suffix. suffix 토큰들의 평균 log P(token|prev)를 계산.
    (Eq.6/Eq.7의 근사 구현: prefix 길이로 토큰 경계를 나눔)"""
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids[0]
    full_ids = tok(full_text, return_tensors="pt").input_ids[0]
    b = prefix_ids.shape[0]
    if full_ids.shape[0] <= b:
        return float("-inf")  # suffix가 비어있는 이례적 상황

    input_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (L, V)

    log_probs = torch.log_softmax(logits[b - 1: full_ids.shape[0] - 1], dim=-1)
    target_ids = full_ids[b:].to(device)
    token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.mean().item()


def build_prompt1(context, question):
    return (
        "Below is a query from a user and a relevant context. "
        "Answer the question given the information in the context.\n"
        f"Context: {context}\nQuery: {question}"
    )


def build_prompt1_prefix(context):
    return (
        "Below is a query from a user and a relevant context. "
        "Answer the question given the information in the context.\n"
        f"Context: {context}\nQuery:"
    )


def build_prompt2(context, question, answer):
    return build_prompt1(context, question) + f"\nAnswer: {answer}"


def build_prompt2_prefix(context, question):
    return build_prompt1(context, question) + "\nAnswer:"


def evaluate_event(scope_entry, event, corpus, poison_texts, retriever, gen_model, tok, device):
    question = event["question"]
    incorrect_response = event["incorrect_response"]
    scope_ids = scope_entry["final_scope_ids"]

    combined_texts = [c["text"] for c in corpus] + poison_texts
    is_poisoned_flags = [False] * len(corpus) + [True] * len(poison_texts)

    q_embed = retriever.encode(
        [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]

    scope_texts = [combined_texts[i] for i in scope_ids]
    scope_flags = [is_poisoned_flags[i] for i in scope_ids]

    # ---- ES(u): 코사인 유사도 (배치 임베딩) ----
    u_embeds = retriever.encode(
        [f"passage: {t}" for t in scope_texts], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    ES_raw = (u_embeds @ q_embed).tolist()

    # ---- SC(u), GC(u): 텍스트 하나씩 forward pass ----
    SC_raw, GC_raw = [], []
    for u_text in scope_texts:
        sc = avg_logprob_of_suffix(
            gen_model, tok, device,
            build_prompt1_prefix(u_text), build_prompt1(u_text, question),
        )
        gc = avg_logprob_of_suffix(
            gen_model, tok, device,
            build_prompt2_prefix(u_text, question),
            build_prompt2(u_text, question, incorrect_response),
        )
        SC_raw.append(sc)
        GC_raw.append(gc)

    ES_z, SC_z, GC_z = zscore(ES_raw), zscore(SC_raw), zscore(GC_raw)
    RS = [(e + s + g) / 3.0 for e, s, g in zip(ES_z, SC_z, GC_z)]

    # ---- K-means(k=2) threshold ----
    if len(set(np.round(RS, 6))) < 2:
        # scope 내 RS 값이 사실상 다 동일한 이례적 상황 - 클러스터링 불가, 전부 benign 처리
        predicted = [False] * len(RS)
    else:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(np.array(RS).reshape(-1, 1))
        cluster_means = [np.mean([RS[i] for i in range(len(RS)) if labels[i] == c]) for c in [0, 1]]
        poisoned_cluster = int(np.argmax(cluster_means))
        predicted = [bool(labels[i] == poisoned_cluster) for i in range(len(RS))]

    tp = sum(1 for gt, pred in zip(scope_flags, predicted) if gt and pred)
    fp = sum(1 for gt, pred in zip(scope_flags, predicted) if not gt and pred)
    fn = sum(1 for gt, pred in zip(scope_flags, predicted) if gt and not pred)
    tn = sum(1 for gt, pred in zip(scope_flags, predicted) if not gt and not pred)
    n = len(scope_flags)
    dacc = (tp + tn) / n if n else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    scope_out = [
        {"id": int(scope_ids[i]), "is_poisoned": scope_flags[i],
         "ES": ES_raw[i], "SC": SC_raw[i], "GC": GC_raw[i], "RS": RS[i],
         "predicted_poisoned": predicted[i]}
        for i in range(len(scope_ids))
    ]
    return {
        "question": question, "attack_type": event["attack_type"],
        "scope": scope_out, "dacc": dacc, "fpr": fpr, "fnr": fnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    with open(os.path.join(DATA_DIR, "corpus.json")) as f:
        corpus = json.load(f)
    with open(os.path.join(DATA_DIR, "misgeneration_events.json")) as f:
        events = json.load(f)
    with open(os.path.join(RESULTS_DIR, "scope_narrowing_log.json")) as f:
        scope_logs = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading models ({device}) ...")
    retriever = SentenceTransformer(RETRIEVER_MODEL)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    gen_model.eval()

    results = []
    for event, scope_entry in zip(events, scope_logs):
        assert event["question"] == scope_entry["question"]
        assert event["attack_type"] == scope_entry["attack_type"]
        r = evaluate_event(
            scope_entry, event, corpus, event["injected_poison_texts"],
            retriever, gen_model, tok, device,
        )
        results.append(r)
        print(
            f"[{r['attack_type']:8s}] Q: {r['question'][:45]:45s} "
            f"| scope {len(r['scope'])} | DACC={r['dacc']:.2f} FPR={r['fpr']:.2f} FNR={r['fnr']:.2f} "
            f"(TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']})"
        )

    out_path = os.path.join(RESULTS_DIR, "rs_scores_and_threshold.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_dacc = sum(r["dacc"] for r in results) / len(results)
    avg_fpr = sum(r["fpr"] for r in results) / len(results)
    avg_fnr = sum(r["fnr"] for r in results) / len(results)
    print("")
    print(f"saved to {out_path}")
    print(f"평균 DACC={avg_dacc:.3f}  평균 FPR={avg_fpr:.3f}  평균 FNR={avg_fnr:.3f}")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print("")
    print("총 실행 시간: " + f"{time.time() - _script_t0:.1f}" + "초")
