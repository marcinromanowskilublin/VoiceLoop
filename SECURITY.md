# Security policy

## Supported version

VoiceLoop is an experimental, pre-release project. Security fixes are applied
to the current `master` branch. Older commits and local forks are not maintained
as supported releases unless a release page explicitly says otherwise.

## Reporting a vulnerability

Do not disclose an unpatched vulnerability, credential, private transcript, or
personal or clinical data in a public issue.

Use GitHub's private vulnerability reporting for this repository:

<https://github.com/marcinromanowskilublin/VoiceLoop/security/advisories/new>

Include the affected revision, impact, reproducible steps, and the smallest
safe proof of concept. Redact secrets and personal data. If private reporting
is temporarily unavailable, contact the repository owner through the GitHub
profile without publishing exploit details.

The maintainer will acknowledge a complete report when practical, assess the
impact, and coordinate a fix or explain why the report is not accepted. No
specific response or remediation deadline is promised for this experimental
project.

## Scope

Particularly relevant areas include:

- authentication and the local session token;
- unintended network exposure of the loopback service;
- unsafe action execution or confirmation bypass;
- command, prompt, or content injection that reaches an executor;
- leakage of transcripts, memories, screenshots, recordings, or credentials;
- vulnerable or tampered dependencies and vendored assets.

VoiceLoop is not a medical device and has not been clinically validated.
Clinical-risk or product-claim concerns that are not software vulnerabilities
should be described without patient-identifying information.
