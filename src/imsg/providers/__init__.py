"""Model-provider wiring (SPEC §4.1, §6 `models.*`).

`imsg.providers.factory` is the one place that turns `config.yaml` into
the provider objects the pipeline stages and the MCP retrieval service
consume; `imsg.providers.manifest` owns `models/manifest.lock.yaml`
(the pinned repo/revision/license/runtime record) and its verifier.

Nothing here imports a real model runtime at module-import time — the
real provider classes are resolved lazily, by dotted path, inside the
`build_*` functions, so `imsg` stays importable (and every `fake`-backend
command stays runnable) without the `models` extra installed.
"""

from imsg.providers.factory import (
    backend_status_line,
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_text_provider,
)

__all__ = [
    "backend_status_line",
    "build_boundary_provider",
    "build_enrichment_providers",
    "build_multimodal_provider",
    "build_reranker",
    "build_text_provider",
]
