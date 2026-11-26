"""VelvetTTS synthesis on CPU and CUDA, including batch-one streaming."""

from ._version import __version__ as __version__
from .engine import VelvetTTS
from .results import GenerationResult, Profile

__all__ = ["GenerationResult", "Profile", "VelvetTTS"]
