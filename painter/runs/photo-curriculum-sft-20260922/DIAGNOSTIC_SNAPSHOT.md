# Matched evaluation diagnostic snapshot

Measured at **2026-09-22T05:44:33Z** from the live Linux node. This is a bounded operational snapshot while the finite evaluation was still running; it is not a quality judgment and does not change the evaluation denominator.

The frozen baseline had 28 completed traces. The student system prompt matched the baseline system prompt byte-for-byte for every sampled midpoint and final trace.

| checkpoint | traces observed | valid turns | invalid turns |
| --- | ---: | ---: | ---: |
| step 512 | 27 | 19 | 8 |
| step 1024 | 28 | 18 | 10 |

The structured renderer errors and their categories were:

| checkpoint | task ID | structured error | category |
| --- | --- | --- | --- |
| step 512 | `coco-capability-20260922-000000000133` | `SyntaxError: Unexpected end` | JavaScript syntax |
| step 512 | `coco-capability-20260922-000000000382` | `Cannot redefine property: arc` | property redefinition |
| step 512 | `coco-capability-20260922-000000000542` | `SyntaxError: Unexpected token (` | JavaScript syntax |
| step 512 | `coco-capability-20260922-000000000030` | `p5.Brush not constructor` | constructor/API mismatch |
| step 512 | `coco-capability-20260922-000000000641` | `No complete JavaScript` | incomplete JavaScript |
| step 512 | `coco-capability-20260922-000000000636` | `B is not defined` | undefined symbol |
| step 512 | `coco-capability-20260922-000000000201` | `SyntaxError: Unexpected }` | JavaScript syntax |
| step 512 | `coco-capability-20260922-000000000081` | `b.map is not a function` | argument shape/type |
| step 1024 | `coco-capability-20260922-000000000009` | `number 418 not iterable` | argument shape/type |
| step 1024 | `coco-capability-20260922-000000000599` | `a not iterable` | argument shape/type |
| step 1024 | `coco-capability-20260922-000000000089` | `number 205 not iterable` | argument shape/type |
| step 1024 | `coco-capability-20260922-000000000208` | `gaussian not defined` | undefined symbol |
| step 1024 | `coco-capability-20260922-000000000030` | `rnd not defined` | undefined symbol |
| step 1024 | `coco-capability-20260922-000000000077` | `Invalid array length` | invalid array length |
| step 1024 | `coco-capability-20260922-000000000201` | `SyntaxError: Unexpected (` | JavaScript syntax |
| step 1024 | `coco-capability-20260922-000000000572` | `y not defined` | undefined symbol |
| step 1024 | `coco-capability-20260922-000000000584` | `number 347.5 not iterable` | argument shape/type |
| step 1024 | `coco-capability-20260922-000000000431` | structured error present; exact text retained in private trace | other structured renderer error |

No invalid case in this snapshot was classified as a renderer-deadline failure. Raw prompts, responses, endpoint fields and credentials remain in the private trace archive.

The earlier draft of this file incorrectly classified failures by searching generated reply text and listed API member names as if they were renderer diagnoses. This corrected version uses the structured `info.turns[].error` field and removes that unsupported classification.
