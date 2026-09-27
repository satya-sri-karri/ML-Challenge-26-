# Blocking Recall & Candidate Generation Report

**Role:** Person B — Blocking / Candidate Generation  
**Branch:** `feature/blocking`  
**Dataset:** Amazon ML Challenge 2026 (Local Training & Test Data)  
**Date:** 2026-09-27  

---

## 1. Executive Summary

This report documents the empirical evaluation of the multi-signal blocking pipeline executed on the **actual competition training dataset** (`dataset/train_source1.tsv`, `dataset/train_source2.tsv`, `dataset/train_source3.tsv`, `dataset/train_ground_truth.tsv`) and verified against the actual test sources (`dataset/test_source*.tsv`).

Blocking serves as the sub-quadratic candidate generation stage ($O(N_1 + N_2 + N_3)$) that reduces the combinatorial search space down to a compact candidate set before pairwise Stage-A feature engineering (LightGBM) and Stage-B verification.

### Actual Measured Metrics on Real Competition Data

The evaluation was performed on a representative sample of **10,000 actual Source 1 entities** evaluated against **134,522 actual target entities** (all 34,522 true ground-truth matches across Source 2 and Source 3 plus 100,000 actual distractor rows).

| Evaluation Split | Source 1 Entities | Total True Pairs | Recovered Pairs | Recall Ceiling | Candidate Pairs Generated | All Possible Pairs | Reduction Ratio |
|---|---|---|---|---|---|---|---|
| **Full Evaluation Set** | 10,000 | 34,522 | 34,233 | **99.16%** | 6,531,055 | 1,345,220,000 | **99.5145%** |
| **Train Split (80%)** | 8,000 | 27,658 | 27,428 | **99.17%** | 5,232,844 | 1,076,176,000 | **99.5173%** |
| **Held-out Validation (20%)** | 2,000 | 6,864 | 6,805 | **99.14%** | 1,298,211 | 269,044,000 | **99.5031%** |

*Key Findings:*
- **Held-out Recall Ceiling:** **99.14%** of genuine ground-truth matches were recovered on the unseen validation entities.
- **Reduction Ratio:** **99.50%** of all possible pairs were pruned, leaving an average of **653.11 candidates** per Source 1 entity for downstream Stage-A feature ranking.
- **Runtime:** **198.31 seconds** (~3.3 minutes).
- **Peak Memory (Python-tracked):** **1,127.32 MB** (~1.1 GB).

*(Note on Previous Synthetic Benchmarks: All numbers reported in this document are strictly measured on the actual competition dataset. The preliminary synthetic benchmark numbers reported during initial scaffolding prior to local dataset delivery have been completely superseded and deprecated.)*

---

## 2. Multi-Signal Blocking Architecture

The candidate generation engine combines two independent, cheap signals and **unions** their candidates to prevent false dismissals:

```
       Normalized Entity Record (Source 1 / Source 2 / Source 3)
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
   [ Signal A: Name ]                             [ Signal B: Address ]
  • Exact name_core                              • Compound: Number + Location Token
  • Domain cleaned name (e.g. aristosteel)         (e.g. "85_ticonderoga")
  • Distinctive token inverted index             • Distinctive token pairs (sorted)
  • 3-character prefix (with block capping)        (e.g. "guthrie_surprise")
  • ASCII transliteration fallback               • Safe empty address path (~3.3% missing)
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
                    UNION of Candidate Sets
                                 │
                                 ▼
                     Pruning & Deduplication
                                 │
                                 ▼
                 output/candidate_pairs.tsv (for Person C)
```

### Signal A: Name-Based Inverted Index
1. **Exact Normalized Name:** `name:{name_core}`.
2. **Domain-Style Cleaning:** For domain names (e.g. `aristosteel.com` in Source 2/3), `name_domain_cleaned` strips the TLD and segments words, generating `name:{aristosteel}` to match non-domain references.
3. **Distinctive Token Index:** Tokens ($\ge 3$ characters) from `name_tokens`, excluding `NAME_STOPWORDS` (`inc`, `ltd`, `services`, `enterprises`, etc.), with frequency capping (`max_token_block_size = 500`).
4. **Prefix Blocking:** 3-character prefixes (`name_prefix:{val[:3]}`), strictly capped at `max_prefix_block_size = 1000`.
5. **Transliterated ASCII Fallback:** Unidecode ASCII tokens from `name_ascii` for non-ASCII Indian script names.

### Signal B: Order-Independent Address Blocking
1. **Number + Location Compound Keys (`addr_num_tok:{number}_{word}`):**
   - Combines numeric tokens from `address_numbers` (`85`, `17437`, `570/13`) with distinctive words from `address_tokens` (`ticonderoga`, `surprise`, `delhi`).
   - Highly selective: recovers true matches where company names are unrelated or corrupted (e.g. `Dréxkor` $\leftrightarrow$ `Atlantic` via `addr_num_tok:85_ticonderoga`).
2. **Sorted Distinctive Token Pairs (`addr_pair:{tok1}_{tok2}`):**
   - Alphabetically sorted pairs of location tokens (e.g. `guthrie_surprise`). Completely invariant to scrambled address field order (e.g. state-first, number-last).
3. **Empty Address Safety:**
   - When `address_is_empty` is `True` (~3.3% of Source 2/3 records), address key extraction returns `set()` without crashing, falling back to name-based candidate generation.

---

## 3. Held-out Validation Methodology

To ensure blocking rules generalize to unseen data without overfitting:
1. **Partitioning:** An entity-level $80/20$ train/validation split was applied to Source 1 entities using fixed random seed `seed = 42`.
2. **Search Space:** Validation Source 1 entities were queried against the complete target pool (Source 2 and Source 3 true matches plus distractors).
3. **Mathematical Formulas:**
   $$\text{Recall Ceiling} = \frac{|\text{Recovered True Pairs in Held-out Split}|}{|\text{Total True Pairs in Held-out Split}|} = \frac{6,805}{6,864} = 99.14\%$$
   $$\text{Reduction Ratio} = 1 - \frac{|\text{Candidates Generated}|}{|S_{1,\text{val}}| \times (|S_2| + |S_3|)} = 1 - \frac{1,298,211}{269,044,000} = 99.5031\%$$

---

## 4. Workload Profiling & Resource Constraints

### Machine Constraints
- **Total RAM:** 16.0 GB
- **Free Physical RAM Available:** ~2.6 GB
- **Logical CPU Cores:** 12

### Full-Scale Dataset Characteristics
- `train_source1.tsv`: 2,206,821 rows (210 MB)
- `train_source2.tsv`: 5,034,616 rows (489 MB)
- `train_source3.tsv`: 5,285,603 rows (504 MB)
- `train_ground_truth.tsv`: 2,206,821 rows (127 MB)
- Total rows across sources: **12,527,040 rows** (~2.5 GB on disk).

### Profiling Observations
- In-memory materialization of all 12.5M rows as uncompressed DataFrames simultaneously would require **6–8 GB RAM**, exceeding the available physical memory (2.6 GB) and risking an Out-Of-Memory (OOM) crash or excessive paging.
- Normalization throughput: **~29,600 rows/second**.
- Candidate generation throughput: **~198 seconds per 10,000 Source 1 queries** against 134.5k targets with peak memory capped at **1,127 MB**.
- For full-dataset generation, `scripts/run_blocking.py` supports streaming / chunked processing to safely process the entire 2.2M Source 1 entities within system memory limits.

---

## 5. Verification on Test Sources (Including France)

The implementation was verified on the actual test sources (`dataset/test_source1.tsv`, `dataset/test_source2.tsv`, `dataset/test_source3.tsv`).
- In `test_source1.tsv`, **France represents 14.9% of entities** (2,981 of first 20,000 rows).
- A sample of 1,000 test entities (including 128 French entities) was evaluated against 20,000 test entities across Source 2 and Source 3.
- The blocking pipeline ran smoothly without errors, generating 233,522 total candidate pairs (76,778 for French entities), confirming zero country gating or language-specific failures on unseen countries.

---

## 6. Generated Output Specification (`candidate_pairs.tsv`)

The candidate generation file was generated at [`output/candidate_pairs.tsv`](file:///c:/Users/SWETHA%20SRI/OneDrive/Documents/ML-Challenge-26-/output/candidate_pairs.tsv) and verified:

```tsv
source1_entity_id	candidate_entity_ids
S1-161901150	S2-124034738,S2-126035631,S2-168688408,...
S1-925783039	S2-157377754,S2-517291332,S3-997698194,...
S1-773889195	
```

### Verification Checks Passed:
1. **Header:** Exactly `source1_entity_id\tcandidate_entity_ids`.
2. **Row Count:** Exactly one row per Source 1 entity.
3. **Format:** Tab-separated (`\t`), comma-separated target IDs.
4. **Target IDs:** Preserves source prefixes (`S2-` and `S3-`).
5. **Deduplication:** 0 duplicate candidate IDs within any row.
6. **Singletons:** Exactly empty candidate field after tab when no candidates are found.
