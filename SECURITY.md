# Security Policy

## Reporting a vulnerability or a privacy problem

Report security issues and privacy problems **privately**, never in a
public GitHub issue. Use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Fill in the advisory form and submit.

This opens a private conversation with the maintainers that nobody else
can see. It is the only reporting channel for this project — do not open
a public issue, and do not paste message content, names, or phone
numbers into any public issue, discussion, or pull request, even to
illustrate a bug. If you're not sure whether something is sensitive,
treat it as sensitive and use private reporting.

A public [bug report](.github/ISSUE_TEMPLATE/bug_report.md) is fine for
ordinary crashes and behavior bugs that don't involve anyone's real data.

## What this project protects

In plain language, here is what stands between your message history and
the outside world:

- **Local-only data.** Your indexed messages and everything derived from
  them live on your own machine, under a data directory you control.
  Nothing is uploaded anywhere by default.
- **Encrypted volume.** The indexed data is expected to live on an
  encrypted disk, and the system refuses to start if that disk isn't
  mounted.
- **Subject and audience validation on the public-facing surface.** Every
  request to the part of the system that can be reached over a network
  is checked two ways: is this really the owner asking (subject), and
  was this request actually meant for this system (audience)? Both
  checks have to pass — either one alone can be bypassed by a replayed
  or misdirected request, so neither is optional and neither can be
  turned off through configuration.
- **Default-deny export.** Nothing leaves the local index and reaches a
  shared or external destination unless it's explicitly allowed. Group
  conversations require every participant to be allowlisted, and
  attachments are gated separately from message text.

## History rewrite (2026-09-24)

On 2026-09-24, this repository's git history was rewritten to remove
real contact information (names, phone numbers, and similar identifiers)
that had been used as test fixtures early in development. If you have an
older clone or fork, discard it and re-clone rather than merging old
history back in.

An automated check, `scripts/check_public_safety.py`, now scans changes
for likely real data before they land. It is a helpful backstop, not a
guarantee — it cannot reliably catch a single common first name used on
its own, for example. A clean run of the check is not proof that a
change is free of real personal data, so human review still matters:
if you're reviewing a pull request, look for this yourself rather than
trusting the check alone.
