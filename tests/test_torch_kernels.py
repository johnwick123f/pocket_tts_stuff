import copy
from types import SimpleNamespace

import pytest
import torch
from pocket_tts.modules.mlp import SimpleMLPAdaLN

from velvet_tts._torch_kernels import optimize_torch


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_sampler_matches_original(batch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(14)
    net = SimpleMLPAdaLN(32, 192, 32, 64, 4, num_time_conds=0).eval().to(device)
    original = copy.deepcopy(net)
    optimize_torch(SimpleNamespace(flow_lm=SimpleNamespace(flow_net=net)))
    condition, noise = torch.randn(batch, 64, device=device), torch.randn(batch, 32, device=device)
    with torch.no_grad():
        torch.testing.assert_close(
            net(condition, noise), original(condition, noise), atol=2e-6, rtol=2e-5
        )
