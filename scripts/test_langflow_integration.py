"""
Test script for Langflow integration with KernelBench
Usage: uv run python scripts/test_langflow_integration.py
"""

from kernelbench.utils import create_inference_server_from_presets

def test_langflow_basic():
    """Test basic Langflow inference"""
    print("="*60)
    print("Testing Langflow Integration")
    print("="*60)

    # Create Langflow inference function
    print("\n1. Creating Langflow inference server...")
    inference_fn = create_inference_server_from_presets(
        server_type="langflow",
        verbose=True,
        time_generation=True
    )
    print("   ✓ Inference server created")

    # Test with a simple CUDA prompt
    print("\n2. Testing with CUDA kernel prompt...")
    prompt = """Write a CUDA kernel that performs element-wise addition of two vectors.

Requirements:
- Accept two input tensors A and B of size N
- Return output tensor C where C[i] = A[i] + B[i]
- Use efficient CUDA parallelization

Please provide the complete implementation."""

    print(f"\n   Prompt: {prompt[:100]}...")
    response = inference_fn(prompt)

    print("\n3. Response received:")
    print("-"*60)
    print(response[:500])  # Print first 500 chars
    if len(response) > 500:
        print(f"... (truncated, total length: {len(response)} chars)")
    print("-"*60)

    print("\n✓ Langflow integration test PASSED!")
    return response

def test_langflow_chat_format():
    """Test Langflow with chat message format"""
    print("\n" + "="*60)
    print("Testing Langflow with Chat Format")
    print("="*60)

    inference_fn = create_inference_server_from_presets(
        server_type="langflow",
        verbose=True
    )

    # Test with chat messages (list of dicts)
    messages = [
        {"role": "system", "content": "You are a CUDA programming expert."},
        {"role": "user", "content": "Explain what memory coalescing is in CUDA."}
    ]

    print("\n   Sending chat-formatted messages...")
    response = inference_fn(messages)

    print("\n   Response:")
    print("-"*60)
    print(response[:500])
    if len(response) > 500:
        print(f"... (truncated, total length: {len(response)} chars)")
    print("-"*60)

    print("\n✓ Chat format test PASSED!")

if __name__ == "__main__":
    try:
        # Run basic test
        test_langflow_basic()

        # Run chat format test
        test_langflow_chat_format()

        print("\n" + "="*60)
        print("All Tests PASSED! ✓")
        print("="*60)
        print("\nYou can now use Langflow with KernelBench:")
        print("  uv run python scripts/generate_and_eval_single_sample.py \\")
        print("    dataset_src=huggingface level=2 problem_id=40 \\")
        print("    server_type=langflow backend=cuda")

    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
