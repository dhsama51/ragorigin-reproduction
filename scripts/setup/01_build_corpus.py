"""
Phase 1 - Step 1: SQuAD corpus 구축 + target question 5개 선정 + retriever 임베딩.

절차:
  1. datasets 라이브러리로 SQuAD(train split) 로드
  2. context 중복 제거 후 시드 고정 셔플로 10,000개 샘플링 -> corpus
  3. corpus에 포함된 context를 gold context로 갖는 (question, answer) 중에서
     5개를 target question으로 선정 (benign 조건에서도 정답 context가
     실제로 corpus 안에 있어야 RAG가 정상적으로 답할 수 있으므로)
  4. e5-small-v2로 corpus 전체를 임베딩 (passage: 접두어 사용, E5 컨벤션)

출력:
  data/corpus.json          : [{"id": int, "text": str}, ...]
  data/passage_embeds.npy   : (N, dim) float32, corpus.json과 순서 일치, L2 정규화됨
  data/target_questions.json: [{"question": str, "gold_context_id": int,
                                 "correct_answer": str}, ...] 5개
"""
import time
import json
import os
import random

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

SEED = 42
CORPUS_SIZE = 10000
N_TARGET_QUESTIONS = 5
DATA_DIR = os.path.expanduser("~/projects/ragorigin/data")
RETRIEVER_MODEL = "intfloat/e5-small-v2"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("loading SQuAD (train split) ...")
    squad = load_dataset("rajpurkar/squad", split="train")
    print(f"  raw examples: {len(squad)}")

    # ---- 1) context 중복 제거, 각 고유 context에 대표 (question, answer) 하나 보관 ----
    context_to_qa = {}
    for ex in squad:
        ctx = ex["context"]
        if ctx not in context_to_qa:
            answers = ex["answers"]["text"]
            context_to_qa[ctx] = {
                "question": ex["question"],
                "answer": answers[0] if answers else None,
            }
    unique_contexts = list(context_to_qa.keys())
    print(f"  unique contexts: {len(unique_contexts)}")

    if len(unique_contexts) < CORPUS_SIZE:
        raise RuntimeError(
            f"고유 context({len(unique_contexts)})가 목표 corpus 크기({CORPUS_SIZE})보다 적음"
        )

    # ---- 2) corpus 샘플링 (시드 고정 셔플) ----
    random.shuffle(unique_contexts)
    corpus_texts = unique_contexts[:CORPUS_SIZE]
    corpus = [{"id": i, "text": t} for i, t in enumerate(corpus_texts)]
    text_to_id = {t: i for i, t in enumerate(corpus_texts)}

    # ---- 3) target question 5개 선정 (gold context가 corpus 안에 있는 것 중에서) ----
    candidates = [
        {
            "question": context_to_qa[ctx]["question"],
            "gold_context_id": text_to_id[ctx],
            "correct_answer": context_to_qa[ctx]["answer"],
        }
        for ctx in corpus_texts
        if context_to_qa[ctx]["answer"]
    ]
    random.shuffle(candidates)
    target_questions = candidates[:N_TARGET_QUESTIONS]
    if len(target_questions) < N_TARGET_QUESTIONS:
        raise RuntimeError("target question 후보가 부족함 - CORPUS_SIZE를 늘려야 함")

    # ---- 4) e5-small-v2로 corpus 임베딩 ----
    print(f"loading retriever ({RETRIEVER_MODEL}) ...")
    model = SentenceTransformer(RETRIEVER_MODEL)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"  device = {device}")

    passages = [f"passage: {c['text']}" for c in corpus]
    print(f"encoding {len(passages)} passages ...")
    embeds = model.encode(
        passages,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,  # 코사인 유사도 = 내적으로 계산 가능하도록 L2 정규화
        convert_to_numpy=True,
    ).astype(np.float32)

    # ---- 저장 ----
    with open(os.path.join(DATA_DIR, "corpus.json"), "w") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(DATA_DIR, "passage_embeds.npy"), embeds)
    with open(os.path.join(DATA_DIR, "target_questions.json"), "w") as f:
        json.dump(target_questions, f, ensure_ascii=False, indent=2)

    print(f"\ncorpus.json          : {len(corpus)}개")
    print(f"passage_embeds.npy    : {embeds.shape}")
    print(f"target_questions.json : {len(target_questions)}개")
    for tq in target_questions:
        print(f"  - Q: {tq['question']}  (A: {tq['correct_answer']})")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print(f"
총 실행 시간: {time.time() - _script_t0:.1f}초")
