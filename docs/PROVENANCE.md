# Ownership, provenance, and licensing

This document records the current, reviewable basis for VoiceLoop's ownership
and licensing statements. It is an engineering record, not a substitute for a
contract or individualized legal advice.

## Project ownership

VoiceLoop is Marcin Romanowski's original project. Unless a file or the Git
history identifies another rightsholder, Marcin Romanowski owns the copyright
in the original project code, documentation, architecture descriptions, and
project-created diagrams. Public availability and an open-source license do not
transfer that ownership.

This statement deliberately does not claim ownership of third-party libraries,
service names, product marks, separately licensed material, or a contributor's
independent copyrightable contribution.

The same rule applies fairly to every contributor: if evidence establishes that
a person holds copyright in a specific, independently copyrightable
contribution, the project-owner statement does not claim or transfer that
contribution. GitHub access, commit metadata, or another technical account
status is not by itself proof of copyright ownership; any specific contribution
should be confirmed from the underlying work and applicable agreement.

The public repository does not waive, override, or replace any valid private
record or agreement concerning ownership, co-ownership, project continuity,
succession, or rights in a specific contribution. The existence and scope of
such rights are determined by the relevant evidence, agreement, license, and
applicable law, not by repository visibility alone.

## License chronology

- The repository history contains no `LICENSE`, `LICENCE`, `COPYING`, or
  `NOTICE` file immediately before commit
  `6c1de947ac05c93568e70b27c0423b4f8dcd0acc`.
- Commit `6c1de947ac05c93568e70b27c0423b4f8dcd0acc`, dated 2026-09-15, added the
  GNU Affero General Public License version 3 and the original `NOTICE.md`.
- Current covered VoiceLoop work is offered under `AGPL-3.0-only`. This record
  does not rewrite the historical licensing status of earlier snapshots and
  does not override a third party's separate license.

The repository license must not be changed, dual-licensed, or described as
retroactively relicensed without an explicit rights review and approval from
all relevant rightsholders.

## Contribution history

The Git history contains commits attributed to Marcin Romanowski and automated
or AI-assisted identities including Cursor Agent and `cursor[bot]`. Commit
metadata and `Co-authored-by` trailers are useful provenance signals, but they
do not alone decide legal authorship or ownership.

Future contributions follow [CONTRIBUTING.md](../CONTRIBUTING.md): no copyright
assignment, AGPL-3.0-only licensing for submitted contributions, DCO sign-off,
and disclosure of substantive AI assistance and third-party sources.

## Project-created visual assets

Project diagrams and portfolio illustrations under `docs/img/` are treated as
VoiceLoop project assets unless an individual file says otherwise. Their commit
history includes both maintainer-attributed and AI-tool-attributed revisions.
They must not be described as independently created by a named human when the
available record does not support that more specific statement.

The explanatory diagrams do not grant rights in third-party names, interfaces,
logos, screenshots, or underlying software that may be referenced.

## Vendored and installed dependencies

Vendored JavaScript has an inventory containing exact versions, upstream URLs,
hashes, and sizes in `vectorscope/static/vendor/MANIFEST.json`. Required
attribution and license text are collected in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Python packages are installed from the declared requirements. Their inclusion
in a lock file does not make them VoiceLoop property. A distributor that bundles
those packages must audit the exact artifact being shipped and include all
required notices.

## Source and data boundaries

The tracked `sources/` directories are intentionally empty except for guidance
and `.gitkeep` placeholders. Local notes, articles, books, recordings,
transcripts, screenshots, model files, databases, credentials, and runtime
artifacts are excluded by `.gitignore` and must not be committed without a
specific provenance, privacy, and license review.

Synthetic test fixtures are not evidence that private user or clinical data may
be published. Contributors must not substitute real personal data for synthetic
fixtures.

## Review procedure

Before publishing a new asset, imported code, dataset, model, or substantial
generated artifact, record:

1. the file or directory;
2. its creator or rightsholder, if known;
3. the source URL or acquisition record;
4. the license or written permission;
5. whether it was modified or generated with an AI tool;
6. required attribution and redistribution conditions;
7. the reviewer and review date.

If any item is unknown, do not represent the material as exclusively owned by
VoiceLoop. Keep it out of a public release until the uncertainty is resolved.
