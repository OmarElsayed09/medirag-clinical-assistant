"""
Standalone CLI Benchmark & Evaluation Script for MediRAG
Covers all Evaluation Rubric Metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.evaluator import RAGEvaluator


def main():
    print("=========================================================================")
    print("MediRAG Comprehensive Evaluation & Metrics Suite")
    print("=========================================================================\n")
    print("Running automated clinical benchmark on RAG Triad & Safety metrics...\n")
    
    evaluator = RAGEvaluator()
    results = evaluator.run_benchmark()
    
    print("========================= EVALUATION SUMMARY =========================")
    print(f" Total Benchmark Questions       : {results['total_test_cases']}")
    print(f" Mean Faithfulness (Grounding)   : {results['mean_faithfulness_pct']}%")
    print(f" Mean Context Relevance         : {results['mean_context_relevance_pct']}%")
    print(f" Retrieval Hit Rate (Recall@K)  : {results['retrieval_hit_rate_pct']}%")
    print(f" Out-of-Scope Refusal Safety    : {results['safety_refusal_accuracy_pct']}%")
    print(f" Average Latency / Query        : {results['average_latency_sec']}s")
    print("=========================================================================\n")
    
    print("---------------------------- PER-QUERY AUDIT ----------------------------")
    for idx, d in enumerate(results["details"], 1):
        print(f"\n[{idx}] Category: {d['category']} | Lang: {d['language']}")
        print(f"    Question : {d['question']}")
        print(f"    Latency  : {d['latency_sec']}s | Faithfulness: {d['faithfulness_score']}% | Context Rel: {d['context_relevance_score']}%")
        print(f"    Safety Passed: {'YES' if d['safety_passed'] else 'NO'} | Citations: {d['citations_count']}")
        print(f"    Answer Snippet: {d['answer'][:120]}...\n")
    print("=========================================================================")
    print("All evaluation metrics successfully computed!")


if __name__ == "__main__":
    main()
