"""omlx tenancy & scheduling for the shared 10.0.10.246:8000 endpoint.

Charter: docs/ARCHITECTURE_V3_DRAFT_0.5.md §7.7.

The omlx server hosts three models in v3.0:
    - Qwen3-Embedding-8B  (text embedding; production v2.13+)
    - ModernBERT          (reranker; production v2.12+)
    - ColPali / ColQwen2.5 (visual embedding; Phase C)

Without an explicit tenancy policy, ColPali's per-page embedding (~1s)
can block query-path requests for text embedding (~10ms) or rerank
(~50ms), violating Q5 latency budget.

Foundation-session status: scheduler is a priority-queue contract +
in-process FIFO with priority dispatch. The actual model dispatch is
delegated to the existing omlx client paths in
src/mmrag_v2/retrieval/* — this package only provides the tenancy
contract for Phase C's ColPali addition.
"""

from .scheduler import (  # noqa: F401
    OmlxScheduler,
    RequestPriority,
    ScheduledRequest,
    submit,
)
from .coresidency_monitor import (  # noqa: F401
    CoresidencyEvent,
    CoresidencyMonitor,
)
