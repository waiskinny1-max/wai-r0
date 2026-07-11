# WAI-R0 Conversation CSV Schema

## Canonical columns

| Column | Required | Semantics |
|---|---:|---|
| `id` | recommended | Stable example identifier; duplicate IDs are rejected. |
| `split` | optional | `train`, `val`, or `test`; used only when declared-split mode is enabled. |
| `task_family` | optional | Family used for stratified analysis and provenance. |
| `difficulty` | optional | Dataset-authored difficulty label; never treated as ground truth capability. |
| `system` | optional | System context. |
| `user` | required for chat schema | User content. |
| `assistant` | required for chat schema | Supervised assistant target. |
| `answer_format` | optional | Expected answer representation. |
| `eval_type` | optional | Intended evaluation mode. |
| `metadata_json` | optional | JSON object encoded as a CSV field. |

A legacy single-text schema is accepted by the compatibility trainer, but the native v0.5 path is designed for explicit conversation rows.

## Validation

The audit checks:

- readable UTF-8/CSV structure;
- required fields;
- valid metadata JSON;
- maximum field sizes;
- duplicate IDs;
- exact normalized content duplicates;
- split assignment;
- cross-split duplicate content;
- token/character-length summaries.

Rejected rows are counted and sampled. Native v0.5 training refuses an audit containing rejected rows.

## Split policy

Hash splitting is the default even when a `split` column exists. Enable declared splits only when they are independently trustworthy and contain usable validation/test rows. Split assignment includes the split seed and canonical row content.

Related examples should be grouped before splitting when the dataset contains paraphrases or multiple turns derived from the same source. Exact deduplication cannot detect every semantic near-duplicate.

## Content hash

The dataset manifest records the source SHA-256 and the resolved split policy. Modifying the file invalidates exact stream resume and changes run identity.
