# The served chat template's MIT sources

Test inputs of `tests/cpu/test_licences.py`, which rebuilds the served chat template
(`launch/profiles/*/chat_template_mm.jinja`, sha256 `7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb`)
from them, byte for byte:

| file | what | sha256 | licence |
|---|---|---|---|
| `zai-org-GLM-5.3-Flash-690b705-chat_template.jinja` | `chat_template.jinja` of zai-org/GLM-5.3-Flash at revision `690b705278a3a58e538fcb37c2ca8b5f9511213c` | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` | MIT, Copyright (c) 2026 Z.AI Co., Ltd (`licenses/MIT-zai-org-GLM-5.3-Flash.txt`) |
| `MiaAI-Lab-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-eaf90d0-chat_template.jinja` | `files/chat_template.jinja` of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at `eaf90d06bc` (2026-08-27): Raymond Lucke's thinking gate | `96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a` | MIT, Copyright (c) 2026 Mia's AI Lab (`licenses/MIT-MiaAI-Lab-GLM-5.3-Flash-EXL3-2x-DGX-Sparks.txt`) |
| `thinking-gate-eaf90d0.diff` | the thinking-gate hunk: `diff` from that project's first file (`1286b8ddfd`, zai-org's template at `04c4e9e9`) to `eaf90d06bc` | `98e431892a72418322d0d7c52bfc7faa94b381a242e25ff2f98477d35cbbbe81` | MIT, as the file above |

`NOTICE` (MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) has the history of the file.
