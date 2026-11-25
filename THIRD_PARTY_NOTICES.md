# Third-party notices

VoiceLoop includes or interoperates with third-party software. Copyright in
those components belongs to their respective rightsholders. Nothing in the
VoiceLoop copyright notice claims ownership of third-party material.

## Vendored JavaScript

The files below are stored in `vectorscope/static/vendor/` so that the local
diagnostic panel can run without an internet connection. Versions, download
locations, file sizes, and SHA-256 checksums are recorded in
`vectorscope/static/vendor/MANIFEST.json`.

| File | Package | Version | Rightsholder notice | License |
|---|---|---:|---|---|
| `cytoscape.min.js` | Cytoscape.js | 3.31.0 | Copyright (c) 2016-2025, The Cytoscape Consortium | MIT |
| `cytoscape-fcose.js` | cytoscape-fcose | 2.2.0 | Copyright (c) 2018-present, iVis-at-Bilkent | MIT |
| `cose-base.js` | cose-base | 2.2.0 | Copyright (c) 2019-present, iVis@Bilkent | MIT |
| `layout-base.js` | layout-base | 2.0.1 | Copyright (c) 2019, iVis@Bilkent | MIT |
| `plotly-cartesian.min.js` | plotly.js-cartesian-dist-min | 3.0.1 | Copyright 2012-2025, Plotly, Inc. | MIT |

Upstream sources:

- <https://github.com/cytoscape/cytoscape.js>
- <https://github.com/iVis-at-Bilkent/cytoscape.js-fcose>
- <https://github.com/iVis-at-Bilkent/cose-base>
- <https://github.com/iVis-at-Bilkent/layout-base>
- <https://github.com/plotly/plotly.js>

### MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Python dependencies

Runtime and development dependencies are declared in `listener/requirements.in`,
`listener/requirements.lock`, and `listener/requirements-dev.txt`. They are not
relicensed as VoiceLoop. Each package remains governed by the license supplied
by its publisher. A lock file records versions; it is not a copyright or license
transfer. Anyone distributing a bundled application, installer, container, or
virtual environment must review and reproduce the notices required by the exact
dependency set included in that distribution.

## External services and applications

References to Windows, VoiceAttack, UI.Vision, n8n, Qdrant, Screenpipe,
Deepgram, Azure Speech, Hume, Gemini, OpenAI-compatible APIs, LM Studio, or other
products describe optional interoperability. Their names and marks belong to
their respective owners. Inclusion here does not imply sponsorship, endorsement,
or transfer of rights.

## Corrections

If an attribution or license entry is incomplete, please report it through a
private security advisory when confidentiality is needed, or through a GitHub
issue when the information is already public. Preserve existing notices while
the matter is reviewed.
