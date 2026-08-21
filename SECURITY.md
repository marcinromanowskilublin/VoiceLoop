# Security policy

VoiceLoop controls a local Windows workstation. Treat reports about action
validation, token handling, privacy boundaries, recording, or unintended cloud
transmission as security-sensitive.

## Reporting

Do not open a public issue containing secrets, recordings, transcripts,
screenshots, medical data, or reproduction data from a real user environment.
Use GitHub's private vulnerability-reporting channel for this repository. If
that channel is unavailable, contact the repository owner privately before
sharing details.

Include:

- the affected commit and component;
- a minimal reproduction using synthetic data;
- the expected and observed behavior;
- whether any local or cloud data was exposed.

Never attach `listener/.env`, `data/`, `sources/`, `.screenpipe`, runtime logs,
tokens, or private corpus files.

## Security boundaries

- VoiceLoop is a loopback-only local application, not an internet-facing
  service.
- Models return typed plans; they do not receive arbitrary shell access.
- Executable actions are restricted to a local allowlist and validated
  arguments.
- Medium- and high-risk actions require local policy checks and, where
  configured, explicit confirmation.
- Screenpipe, meeting-audio upload, cloud LLMs, Hume, and n8n are optional.
  Broad context capture and meeting processing are disabled by default.
- The public tests and fixtures must contain only synthetic data.

## Supported scope

The verified target is Windows with Python 3.11. Linux can exercise the core
API and automated tests in degraded mode, without Windows automation. Optional
or experimental integrations are identified as such in `README.md` and are not
covered by a production-security claim.
