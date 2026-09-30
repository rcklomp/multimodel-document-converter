"""Back-compat shim — the R3 code-indentation detector moved to src.

The single source of truth is now ``mmrag_v2.validators.code_quality`` so the
conversion-time specialist re-extraction (``mmrag_v3.processor``) and the
audit-time gate share ONE detector. Scripts and tests that ``import _code_quality``
keep working unchanged via this re-export. See ``docs/PLAN_R3_CODE_GATE_REDESIGN.md``
and ``docs/PLAN_EXTRACTION_FIDELITY_V1.md`` §5.4.
"""

from __future__ import annotations

from mmrag_v2.validators.code_quality import *  # noqa: F401,F403
