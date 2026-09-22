"""
RAGOrigin 미니 재현 파이프라인에서 실제로 반복 호출되는 4가지 LLM 연산의
지연시간을 로컬 GPU(RTX 2070 Super 등)에서 실측하는 벤치마크.

측정 대상:
  1. scope_narrowing_generate : Algorithm 1의 각 iteration에서 K개 컨텍스트로
     RAG 응답을 생성하는 호출 (짧은 답변 생성, ~20 토큰)
  2. match_judgment_generate  : 생성된 응답 r이 오답 r과 의미적으로 일치하는지
     판정하는 호출 (Yes/No류 짧은 생성)
  3. sc_forward (Eq.6)        : Prompt 1(context+query) 구조에서 query 토큰들의
     평균 log-prob 계산 - 생성 없이 forward pass 1회
  4. gc_forward (Eq.7)        : Prompt 2(context+query+answer) 구조에서 answer
     토큰들의 평균 log-prob 계산 - 생성 없이 forward pass 1회

각 연산을 워밍업 3회 후 N회(기본 10) 반복 측정하여 평균/표준편차/최소/최대를 출력.
이 결과로 target question(=misgeneration event) 개수와 corpus 규모를
2시간 예산 안에서 얼마나 늘릴 수 있는지 역산하는 데 사용.

사용법:
  conda activate ragorigin
  python 00_benchmark_llm_calls.py --model Qwen/Qwen2.5-1.5B-Instruct --reps 10
"""
import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DUMMY_CONTEXTS = [
    "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France, "
    "completed in 1889 as the entrance arch for the World's Fair.",
    "Photosynthesis is the process by which green plants convert light energy, "
    "usually from the sun, into chemical energy stored in glucose molecules.",
    "The Amazon River is the largest river by discharge volume in the world, "
    "flowing through South America and emptying into the Atlantic Ocean.",
    "Python is a high-level, general-purpose programming language created by "
    "Guido van Rossum and first released in 1991.",
    "The mitochondrion is often referred to as the powerhouse of the cell "
    "because it generates most of the cell's supply of ATP.",
]
DUMMY_QUESTION = "Where is the Eiffel Tower located?"
DUMMY_INCORRECT_RESPONSE = "The Eiffel Tower is located in Berlin, Germany."


def build_prompt1(contexts, question):
    """SC(u) 계산용 Prompt 1 (논문 5.3절)."""
    ctx = "\n".join(contexts)
    return (
        "Below is a query from a user and a relevant context. "
        "Answer the question given the information in the context.\n"
        f"Context: {ctx}\nQuery: {question}"
    )


def build_prompt2(contexts, question, answer):
    """GC(u) 계산용 Prompt 2 (논문 5.3절)."""
    return build_prompt1(contexts, question) + f"\nAnswer: {answer}"


def time_calls(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def report(name, times):
    print(f"\n[{name}]  n={len(times)}")
    print(f"  mean = {statistics.mean(times):.3f}s")
    print(f"  std  = {statistics.pstdev(times):.3f}s")
    print(f"  min  = {min(times):.3f}s")
    print(f"  max  = {max(times):.3f}s")
    return statistics.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--k", type=int, default=5, help="scope narrowing 세그먼트 크기 K")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if device == "cuda":
        print(f"gpu = {torch.cuda.get_device_name(0)}")

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    contexts_k = DUMMY_CONTEXTS[: args.k] if args.k <= len(DUMMY_CONTEXTS) else (
        DUMMY_CONTEXTS * ((args.k // len(DUMMY_CONTEXTS)) + 1)
    )[: args.k]

    # ---- 1. scope_narrowing_generate ----
    gen_prompt = build_prompt1(contexts_k, DUMMY_QUESTION)
    gen_inputs = tok(gen_prompt, return_tensors="pt").to(device)

    def scope_narrowing_generate():
        with torch.no_grad():
            model.generate(
                **gen_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    t1 = time_calls(scope_narrowing_generate, args.reps)
    mean_gen = report("1) scope_narrowing_generate (RAG 응답 생성)", t1)

    # ---- 2. match_judgment_generate ----
    judge_prompt = (
        "Does the following response convey the same meaning as the target answer? "
        "Answer only Yes or No.\n"
        f"Response: {DUMMY_INCORRECT_RESPONSE}\nTarget answer: The Eiffel Tower is in Berlin."
    )
    judge_inputs = tok(judge_prompt, return_tensors="pt").to(device)

    def match_judgment_generate():
        with torch.no_grad():
            model.generate(
                **judge_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    t2 = time_calls(match_judgment_generate, args.reps)
    mean_judge = report("2) match_judgment_generate (Match 판정)", t2)

    # ---- 3. sc_forward (Eq.6) : 생성 없이 forward pass ----
    sc_prompt = build_prompt1([contexts_k[0]], DUMMY_QUESTION)
    sc_inputs = tok(sc_prompt, return_tensors="pt").to(device)

    def sc_forward():
        with torch.no_grad():
            model(**sc_inputs)

    t3 = time_calls(sc_forward, args.reps)
    mean_sc = report("3) sc_forward (SC(u) log-prob, Eq.6)", t3)

    # ---- 4. gc_forward (Eq.7) : 생성 없이 forward pass ----
    gc_prompt = build_prompt2([contexts_k[0]], DUMMY_QUESTION, DUMMY_INCORRECT_RESPONSE)
    gc_inputs = tok(gc_prompt, return_tensors="pt").to(device)

    def gc_forward():
        with torch.no_grad():
            model(**gc_inputs)

    t4 = time_calls(gc_forward, args.reps)
    mean_gc = report("4) gc_forward (GC(u) log-prob, Eq.7)", t4)

    # ---- 예산 역산 ----
    print("\n" + "=" * 60)
    print("예산 역산 (대략적인 추정치)")
    print("=" * 60)
    assumed_iters = 6       # 평균 조기종료 없이 6 iteration 정도 진행된다고 가정
    assumed_scope_size = assumed_iters * args.k
    per_event = (
        assumed_iters * (mean_gen + mean_judge)
        + assumed_scope_size * (mean_sc + mean_gc)
    )
    print(f"가정: iteration {assumed_iters}회, 최종 scope 크기 {assumed_scope_size}개 텍스트")
    print(f"misgeneration event 1개(baseline 1회 실행) 당 예상 시간: {per_event:.1f}초")

    for n_runs_per_event in [5, 10, 15, 20]:
        for n_events in [3, 5, 10, 15]:
            total_sec = per_event * n_runs_per_event * n_events
            print(
                f"  event당 실행 {n_runs_per_event:>2}회 x event {n_events:>2}개 "
                f"= 총 {total_sec/60:.1f}분"
            )
    print("\n주의: iteration 수·scope 크기는 가정치이며 실제 조기종료 여부에 따라 달라짐.")
    print("이 값은 상한선에 가까운 보수적 추정치로 참고할 것.")


if __name__ == "__main__":
    _script_t0 = time.time()
    main()
    print(f"
총 실행 시간: {time.time() - _script_t0:.1f}초")
