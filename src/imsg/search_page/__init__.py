"""The private local search page (D14): an always-on web page, on the
local network only, that returns every viable hit for a query, grouped
by conversation, with threads and attachments one click away.

It never loads a model. Semantic search asks the public MCP server's
already-loaded models through a loopback-only internal API
(`imsg.search_page.model_api_server`, hosted by `imsg mcp public`); when
that API is unavailable, full-text search still answers.

Modules:

- `config` — the `search_page:` config section.
- `secret_files` — 0600 files on the encrypted volume.
- `auth` — owner password, sessions, CSRF, login rate limiting, Host allowlist.
- `search` — every-hit search: full text, semantic above a threshold, not yet indexed.
- `highlight` — matched-term highlighting.
- `threads` — conversation windows for the thread view.
- `attachments` — serving files from the content-addressed cache, thumbnails.
- `labels` — relevance labels in the eval harness's format.
- `model_api_server` / `model_api_client` — the internal model API.
- `html` — server-rendered pages.
- `app` — the Starlette application and its security middleware.
- `server` — listeners and the process entry point.
- `cli` — `imsg search-page serve | set-password | init-model-secret | check`.

The KeepAlive launch agent is `imsg install-agents --only search-page`
(`imsg.agents.plists`): the shared supervisor waits for the volume, passes
the mount guard and reads `env:` secrets from 0600 files.
"""
