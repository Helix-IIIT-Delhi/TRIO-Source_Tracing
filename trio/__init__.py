"""TRIO: Source Tracing of Synthetic Speech Systems"""

__version__ = "1.0.0"

from .models import TRIOModel
from .fusion import ParalinguisticFusion
from .attention import GatedAttention
from .correlation import CanonicalCorrelation

__all__ = ["TRIOModel", "ParalinguisticFusion", "GatedAttention", "CanonicalCorrelation"]
