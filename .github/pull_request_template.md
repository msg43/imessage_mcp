## What this changes

<!-- Briefly describe the change and why it's needed. -->

## Checklist

- [ ] `uv run ruff check . && uv run mypy . && uv run pytest` passes
- [ ] Ran `scripts/check_public_safety.py` (or the equivalent pre-commit
      check) and reviewed the diff myself — no real names, phone
      numbers, emails, hostnames, project ids, bucket names, or business
      names anywhere, including commit messages. Fictional personas
      (Alice, Bob, Acme Construction) only.
- [ ] No credentials, tokens, or instance-specific config added
- [ ] If I touched a migration: I added a new migration rather than
      editing one that's already applied
- [ ] If I touched retrieval/segmentation: I ran the eval harness and
      can show a before/after diff
- [ ] Added a `CHANGELOG.md` entry if this is a notable change (schema,
      feature, fix batch, gate transition)

## Notes for reviewers

<!-- Anything that would help review go faster. -->
