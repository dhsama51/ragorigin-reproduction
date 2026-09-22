"""
Phase 2 - 실험 2-3: Self-Judgment Bias.

동일한 10개 misgeneration event에 대해, judgment LLM을 두 가지로 바꿔가며
scope narrowing + RS/threshold까지 전체 파이프라인을 재실행하고 비교:
  - self   : RAG 생성/proxy/judgment 전부 Qwen2.5-1.5B-Instruct (지금까지 사용한 baseline)
  - separate: judgment만 Llama-3.2-1B-Instruct로 교체 (RAG 생성/proxy는 Qwen 유지)

비교 지표: iteration 수, 최종 scope 크기, DACC/FPR/FNR 차이.

출력:
  results/self_judgment_bias/self_judgment_bias_result.json
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
RESULTS_DIR = os.path.expanduser("~/projects/ragorigin/results/self_judgment_bias")
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ALT_JUDGE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RETRIEVER_MODEL = "intfloat/e5-small-v2"
K = 5
MAX_ITERS = 200


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


def judge_match(judge_model, judge_tok, judge_device, response, reference_response):
    if response.strip().lower().startswith("i don't know") or \
       response.strip().lower().startswith("i do not know"):
        return False
    prompt = (
        "Does the RESPONSE convey the same meaning as the REFERENCE, even if worded "
        "differently? Answer with only one word: Yes or No.\n"
        f"RESPONSE: {response}\nREFERENCE: {reference_response}\nAnswer:"
    )
    out = generate(judge_model, judge_tok, judge_device, prompt, max_new_tokens=3)
    return out.strip().lower().startswith("yes")


def zscore(values):
    arr = np.asarray(values, dtype=np.float64)
    mu, sigma = arr.mean(), arr.std() + 1e-8
    return ((arr - mu) / sigma).tolist()


def avg_logprob_of_suffix(model, tok, device, prefix_text, full_text):
    prefix_ids = tok(prefix_text, return_tensors="pt").input_ids[0]
    full_ids = tok(full_text, return_tensors="pt").input_ids[0]
    b = prefix_ids.shape[0]
    if full_ids.shape[0] <= b:
        return float("-inf")
    input_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits[b - 1: full_ids.shape[0] - 1], dim=-1)
    target_ids = full_ids[b:].to(device)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean().item()


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


def narrow_scope(event, corpus, corpus_embeds, retriever, gen_model, gen_tok, device,
                  judge_model, judge_tok, judge_device):
    question = event["question"]
    incorrect_response = event["incorrect_response"]
    poison_texts = event["injected_poison_texts"]

    poison_embeds = retriever.encode(
        [f"passage: {t}" for t in poison_texts], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    combined_embeds = np.vstack([corpus_embeds, poison_embeds])
    combined_texts = [c["text"] for c in corpus] + poison_texts
    is_poisoned_flags = [False] * len(corpus) + [True] * len(poison_texts)

    q_embed = retriever.encode(
        [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]
    sims = combined_embeds @ q_embed
    ranked_idx = np.argsort(-sims)

    iterations = []
    cumulative_match = 0
    scope_ids = []
    i = 1
    while i <= MAX_ITERS:
        start, end = (i - 1) * K, i * K
        if start >= len(ranked_idx):
            break
        segment_idx = ranked_idx[start:end]
        segment_ids = [int(x) for x in segment_idx]

        if i == 1:
            response, match = incorrect_response, True
        else:
            segment_contexts = [combined_texts[idx] for idx in segment_idx]
            prompt = build_rag_prompt(segment_contexts, question)
            response = generate(gen_model, gen_tok, device, prompt)
            match = judge_match(judge_model, judge_tok, judge_device, response, incorrect_response)

        if match:
            cumulative_match += 1
        scope_ids.extend(segment_ids)
        terminate = (cumulative_match == i / 2)
        iterations.append({"i": i, "match": match, "terminate": terminate})
        if terminate:
            break
        i += 1

    return iterations, scope_ids, combined_texts, is_poisoned_flags


def measure_and_threshold(scope_ids, combined_texts, is_poisoned_flags, event,
                           retriever, gen_model, gen_tok, device):
    question = event["question"]
    incorrect_response = event["incorrect_response"]
    q_embed = retriever.encode(
        [f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]

    scope_texts = [combined_texts[i] for i in scope_ids]
    scope_flags = [is_poisoned_flags[i] for i in scope_ids]

    u_embeds = retriever.encode(
        [f"passage: {t}" for t in scope_texts], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    ES_raw = (u_embeds @ q_embed).tolist()

    SC_raw, GC_raw = [], []
    for u_text in scope_texts:
        SC_raw.append(avg_logprob_of_suffix(
            gen_model, gen_tok, device, build_prompt1_prefix(u_text), build_prompt1(u_text, question)))
        GC_raw.append(avg_logprob_of_suffix(
            gen_model, gen_tok, device, build_prompt2_prefix(u_text, question),
            build_prompt2(u_text, question, incorrect_response)))

    RS = [(e + s + g) / 3.0 for e, s, g in zip(zscore(ES_raw), zscore(SC_raw), zscore(GC_raw))]

    if len(set(np.round(RS, 6))) < 2:
        predicted = [False] * len(RS)
    else:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = km.fit_predict(np.array(RS).reshape(-1, 1))
        cluster_means = [np.mean([RS[i] for i in range(len(RS)) if labels[i] == c]) for c in [0, 1]]
        poisoned_cluster = int(np.argmax(cluster_means))
        predicted = [bool(labels[i] == poisoned_cluster) for i in range(len(RS))]

    tp = sum(1 for gt, p in zip(scope_flags, predicted) if gt and p)
    fp = sum(1 for gt, p in zip(scope_flags, predicted) if not gt and p)
    fn = sum(1 for gt, p in zip(scope_flags, predicted) if gt and not p)
    tn = sum(1 for gt, p in zip(scope_flags, predicted) if not gt and not p)
    n = len(scope_flags)
    return {
        "dacc": (tp + tn) / n if n else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
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
    gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    gen_model.eval()

    print(f"loading alt judge ({ALT_JUDGE_MODEL}) ...")
    alt_tok = AutoTokenizer.from_pretrained(ALT_JUDGE_MODEL)
    alt_model = AutoModelForCausalLM.from_pretrained(
        ALT_JUDGE_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    alt_model.eval()

    results = []
    for event in events:
        # ---- self: Qwen이 judge 겸용 ----
        iters_self, scope_self, ctexts, flags = narrow_scope(
            event, corpus, corpus_embeds, retriever, gen_model, gen_tok, device,
            gen_model, gen_tok, device,
        )
        metrics_self = measure_and_threshold(scope_self, ctexts, flags, event, retriever, gen_model, gen_tok, device)

        # ---- separate: Llama-3.2-1B가 judge ----
        iters_sep, scope_sep, ctexts2, flags2 = narrow_scope(
            event, corpus, corpus_embeds, retriever, gen_model, gen_tok, device,
            alt_model, alt_tok, device,
        )
        metrics_sep = measure_and_threshold(scope_sep, ctexts2, flags2, event, retriever, gen_model, gen_tok, device)

        print(
            f"Q: {event['question'][:35]:35s} ({event['attack_type']}) | "
            f"self: iter={len(iters_self)} scope={len(scope_self)} DACC={metrics_self['dacc']:.2f} | "
            f"separate: iter={len(iters_sep)} scope={len(scope_sep)} DACC={metrics_sep['dacc']:.2f}"
        )

        results.append({
            "question": event["question"], "attack_type": event["attack_type"],
            "self": {"n_iterations": len(iters_self), "scope_size": len(scope_self), **metrics_self},
            "separate": {"n_iterations": len(iters_sep), "scope_size": len(scope_sep), **metrics_sep},
        })

    out_path = os.path.join(RESULTS_DIR, "self_judgment_bias_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_iter_self = sum(r["self"]["n_iterations"] for r in results) / len(results)
    avg_iter_sep = sum(r["separate"]["n_iterations"] for r in results) / len(results)
    avg_dacc_self = sum(r["self"]["dacc"] for r in results) / len(results)
    avg_dacc_sep = sum(r["separate"]["dacc"] for r in results) / len(results)
    print("")
    print(f"saved to {out_path}")
    print(f"평균 iteration: self={avg_iter_self:.2f}  separate={avg_iter_sep:.2f}")
    print(f"평균 DACC:      self={avg_dacc_self:.3f}  separate={avg_dacc_sep:.3f}")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print("")
    print("총 실행 시간: " + f"{time.time() - _script_t0:.1f}" + "초")
