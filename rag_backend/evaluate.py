from collections import defaultdict
from rag_backend.rag_service import RAGService

# --- Test set built from notes.txt content ---
test_set = [
    # CATEGORY 1: Direct factual questions
    {"question": "What is AI?", "expected_keyword": "simulate human learning", "category": "direct"},
    {"question": "What is deep learning?", "expected_keyword": "subset of machine learning", "category": "direct"},
    {"question": "What is generative AI?", "expected_keyword": "create complex original content", "category": "direct"},
    {"question": "What is transfer learning?", "expected_keyword": "knowledge gained through one task", "category": "direct"},
    {"question": "What is reinforcement learning?", "expected_keyword": "trial-and-error", "category": "direct"},

    # CATEGORY 2: Indirect phrasing (no keyword overlap)
    {"question": "What technology lets a computer act on its own without a person controlling it?", "expected_keyword": "act independently", "category": "indirect"},
    {"question": "How does deep learning mimic the human brain?", "expected_keyword": "multilayered neural networks", "category": "indirect"},
    {"question": "What lets a model apply what it learned on one task to a different task?", "expected_keyword": "transfer learning", "category": "indirect"},

    # CATEGORY 3: Comparison / multi-concept (expect struggle - documents model limitation)
    {"question": "What's the difference between semi-supervised and self-supervised learning?", "expected_keyword": None, "category": "comparison"},
    {"question": "How is deep learning different from classic machine learning?", "expected_keyword": None, "category": "comparison"},

    # CATEGORY 4: Multi-part questions
    {"question": "What are VAEs and when were they introduced?", "expected_keyword": "2013", "category": "multi_part"},
    {"question": "Name the three phases of how generative AI works.", "expected_keyword": "Training", "category": "multi_part"},

    # CATEGORY 5: "Why" / inferential questions
    {"question": "Why does deep learning enable machine learning at a large scale?", "expected_keyword": "doesn't require human intervention", "category": "inferential"},

    # CATEGORY 6: Out-of-scope (tests hallucination resistance)
    {"question": "What is blockchain technology?", "expected_keyword": None, "category": "out_of_scope", "expect_idk": True},
    {"question": "What is quantum computing?", "expected_keyword": None, "category": "out_of_scope", "expect_idk": True},
    {"question": "Who invented the telephone?", "expected_keyword": None, "category": "out_of_scope", "expect_idk": True},

    # CATEGORY 7: Adversarial / tricky wording
    {"question": "Define diffusion models and mention the year they were first seen.", "expected_keyword": "2014", "category": "tricky"},
    {"question": "Which models are described as being at the core of tools like ChatGPT and BERT?", "expected_keyword": "Transformers", "category": "tricky"},
]


def run_evaluation(notes_path: str = "data/notes.txt"):
    with open(notes_path, "r", encoding="utf-8") as f:
        notes = f.read()

    rag = RAGService()
    rag.load_knowledge_base(notes)

    results = defaultdict(lambda: {"total": 0, "correct": 0})
    detailed_log = []

    for case in test_set:
        context = rag.store.retrieve(case["question"], k=3)
        answer = rag.generator.generate(context, case["question"])
        category = case["category"]
        results[category]["total"] += 1

        is_correct = False
        if case.get("expect_idk"):
            is_correct = "don't know" in answer.lower()
        elif case["expected_keyword"]:
            is_correct = case["expected_keyword"].lower() in context.lower()

        if is_correct:
            results[category]["correct"] += 1

        detailed_log.append({
            "category": category,
            "question": case["question"],
            "context_found": is_correct,
            "answer": answer
        })

        print(f"[{category}] Q: {case['question']}")
        print(f"   Expected match found: {is_correct}")
        print(f"   Answer: {answer}\n")

    print("=" * 50)
    print("SUMMARY BY CATEGORY")
    print("=" * 50)
    total_correct = 0
    total_cases = 0
    for cat, stats in results.items():
        print(f"{cat}: {stats['correct']}/{stats['total']}")
        total_correct += stats["correct"]
        total_cases += stats["total"]

    print("-" * 50)
    print(f"OVERALL: {total_correct}/{total_cases} ({100 * total_correct / total_cases:.1f}%)")

    return results, detailed_log


if __name__ == "__main__":
    run_evaluation()