import pytest
import torch
from torch.nn import functional as F

from velvet_tts._codec_attention import local_attention


@pytest.mark.parametrize("length", [1, 31, 32, 33, 113, 257])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_attention_matches_dense_mask(length, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(101)
    q, k, v = (torch.randn(2, 4, length, 32, device=device) for _ in range(3))
    delta = torch.arange(length, device=device)[:, None] - torch.arange(length, device=device)[None]
    expected = F.scaled_dot_product_attention(q, k, v, (delta >= 0) & (delta < 50))
    torch.testing.assert_close(local_attention(q, k, v, 50), expected, atol=2e-6, rtol=2e-5)
