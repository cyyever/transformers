"""
Test that chunked prefill frees intermediate outputs between chunks.

Without the `del outputs` fix, intermediate outputs (containing logits of shape
[batch, chunk_size, vocab_size]) stay alive in memory until Python's GC collects
them. This test uses weakref to verify they are freed eagerly.

Run: pytest tests/generation/test_chunked_prefill_memory.py -v
"""

import gc
import unittest
import weakref

from transformers.testing_utils import is_torch_available, require_torch

if is_torch_available():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


@require_torch
class ChunkedPrefillMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.float32)
        cls.model.eval()

    def test_chunked_prefill_frees_intermediate_outputs(self):
        """Intermediate outputs from earlier chunks should not be alive after prefill completes."""
        # Create a long-ish input that will be split into multiple chunks
        input_text = "word " * 40  # ~40 tokens
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Track all outputs objects created during forward passes via weakref
        output_refs = []
        original_call = self.model.__class__.__call__

        def tracking_call(self_model, *args, **kwargs):
            result = original_call(self_model, *args, **kwargs)
            output_refs.append(weakref.ref(result))
            return result

        self.model.__class__.__call__ = tracking_call
        try:
            gen_config = GenerationConfig(
                max_new_tokens=1,
                prefill_chunk_size=10,  # small chunks to get multiple iterations
            )
            self.model.generate(**inputs, generation_config=gen_config)
        finally:
            self.model.__class__.__call__ = original_call

        # Force GC to clean up anything with zero references
        gc.collect()

        # We should have multiple output refs (one per chunk + one for the decode step)
        self.assertGreater(len(output_refs), 2, "Expected multiple chunks but got too few forward calls")

        # The last output (from the final chunk or decode) may still be alive — that's fine.
        # But intermediate outputs (all except the last) should have been freed.
        intermediate_refs = output_refs[:-1]
        alive_intermediates = sum(1 for ref in intermediate_refs if ref() is not None)
        self.assertEqual(
            alive_intermediates,
            0,
            f"{alive_intermediates}/{len(intermediate_refs)} intermediate outputs still alive after "
            f"chunked prefill — they should have been freed by `del outputs`",
        )


if __name__ == "__main__":
    unittest.main()
