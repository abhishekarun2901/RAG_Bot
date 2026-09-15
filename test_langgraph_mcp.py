import os
import sys
from app.pipeline.graph import rag_graph
from app.config import settings


def test_graph_execution():
    if not os.getenv("GROQ_API_KEY") and not settings.GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set.")
        sys.exit(1)

    print("\n--- Running LangGraph Execution Test ---")
    
    test_inputs = {
        "user_query": "What are the thermal safety requirements in ISO standards?",
        "top_k": 3,
        "similarity_threshold": 0.45,
        "metadata_filter": None,
        "effective_filter": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "context_retrieved": False
    }

    output = rag_graph.invoke(test_inputs)

    print(f"Query: {output['user_query']}")
    print(f"Context Retrieved: {output['context_retrieved']}")
    print(f"Sources Count: {len(output['sources'])}")
    print(f"Answer Output:\n{output['answer']}\n")

    assert "answer" in output
    assert isinstance(output["sources"], list)
    print("SUCCESS: LangGraph state machine executed correctly!")


if __name__ == "__main__":
    test_graph_execution()