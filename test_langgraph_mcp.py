import os
import sys
from app.pipeline.graph import rag_graph
from app.config import settings


def run_self_rag_verification():
    if not os.getenv("GROQ_API_KEY") and not settings.GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY is not set.")
        sys.exit(1)

    print("=" * 70)
    print("SELF-RAG VERIFICATION AUDIT: LANGGRAPH + MCP CONTEXT EXPANSION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test Case 1: Trigger Context Expansion (Self-RAG Loop Active)
    # ------------------------------------------------------------------
    print("\n[TEST 1] Testing Context Expansion Trigger...")
    expansion_query = "What are the specific clauses for high-voltage isolation in ISO standards?"

    state_input_1 = {
        "user_query": expansion_query,
        "top_k": 1,
        "similarity_threshold": 0.35,
        "metadata_filter": None,
        "effective_filter": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "context_retrieved": False,
        "needs_expansion": False,
        "expansion_target": None,
        "expansion_attempts": 0,
        "expanded_contexts": []
    }

    output_1 = rag_graph.invoke(state_input_1)

    print(f"-> Expansion Attempts:  {output_1.get('expansion_attempts')}")
    print(f"-> Expanded Sections:   {len(output_1.get('expanded_contexts', []))}")
    print(f"-> Needs Expansion:     {output_1.get('needs_expansion')}")
    print(f"-> Context Retrieved:   {output_1.get('context_retrieved')}")
    print(f"-> Answer Preview:\n   {output_1['answer'][:250]}...\n")

    # Assertions for Self-RAG behavior
    assert output_1["context_retrieved"] is True, "Test 1 failed: Context was not retrieved."
    assert output_1.get("expansion_attempts", 0) > 0, "Test 1 failed: Self-RAG grader never attempted expansion."
    assert len(output_1.get("expanded_contexts", [])) > 0, "Test 1 failed: Page expansion tool was not executed."
    print("[PASS] Test 1: Self-RAG detected clause triggers and widened document context via MCP.")

    # ------------------------------------------------------------------
    # Test Case 2: Verification of Fallback Circuit Breaker
    # ------------------------------------------------------------------
    print("\n[TEST 2] Testing Out-of-Domain Guardrail...")
    ood_query = "What is the boiling point of liquid helium on Mars?"

    state_input_2 = {
        "user_query": ood_query,
        "top_k": 2,
        "similarity_threshold": 0.75,
        "metadata_filter": None,
        "effective_filter": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "context_retrieved": False,
        "needs_expansion": False,
        "expansion_target": None,
        "expansion_attempts": 0,
        "expanded_contexts": []
    }

    output_2 = rag_graph.invoke(state_input_2)

    print(f"-> Expansion Attempts:  {output_2.get('expansion_attempts')}")
    print(f"-> Context Retrieved:   {output_2.get('context_retrieved')}")
    print(f"-> Fallback Answer:     {output_2['answer']}")

    assert output_2["context_retrieved"] is False, "Test 2 failed: Irrelevant query should have context_retrieved=False."
    assert output_2.get("expansion_attempts", 0) == 0, "Test 2 failed: Empty context should not trigger page expansion."
    assert len(output_2.get("expanded_contexts", [])) == 0, "Test 2 failed: No context should be expanded on out-of-domain queries."
    assert "cannot answer this question" in output_2["answer"], "Test 2 failed: Refusal message missing."
    print("[PASS] Test 2: Guardrail verified. Out-of-domain query did not waste expansion cycles.")

    print("\n" + "=" * 70)
    print("ALL SELF-RAG GRAPH TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_self_rag_verification()