# Blocking module - how to use it

Owned by Person B. Branch: `feature/blocking`.

## What it does

src/blocking/ generates candidate entity pairs between Source 1 and one target source (Source 2 or Source 3) before downstream matching.

The goal is to avoid comparing every Source 1 entity with every target entity. Candidate pairs are generated using exact-name and prefix blocking.

The implementation currently uses the normalized name_core column provided by src/normalization/.

## Quick start

from src.blocking.blocker import generate_candidates

candidates = generate_candidates(source1_normalized, source2_normalized)

The same function can be used for Source 3:

candidates = generate_candidates(source1_normalized, source3_normalized)

The input DataFrames must contain:

- entity_id
- name_core

name_core should come from the normalization module.

## Candidate output

The returned DataFrame contains:

- source1_index
- source2_index
- source1_entity_id
- source2_entity_id

Each row represents one possible Source 1 to target-source entity pair.

Duplicate candidate pairs are removed.

## Blocking methods

### Exact-name blocking

Records with the same non-empty name_core are placed in the same block.

For example:

Source 1: microsoft
Source 2: microsoft

This produces a candidate pair.

### Prefix blocking

Records with a name_core length of at least three characters are grouped using the first three characters.

For example:

microsoft -> mic
micromax -> mic

This produces a candidate pair.

Very large prefix blocks are skipped to prevent candidate explosion.

The default maximum Source 2 prefix block size is 1000.

Exact-name candidates are still generated independently of this limit.

## Measuring candidate volume

Use measure_candidate_volume() to measure the size and distribution of generated candidate pairs.

It returns:

- total_candidate_pairs
- unique_source1_entities
- unique_source2_entities
- avg_candidates_per_source1

## Measuring blocking recall

Use measure_blocking_recall() when training ground truth is available.

The ground-truth DataFrame must contain:

- source1_entity_id
- matched_entity_ids

The candidates DataFrame must contain:

- source1_entity_id
- source2_entity_id

The returned metrics are:

- total_true_pairs
- recovered_true_pairs
- blocking_recall

Blocking recall is calculated as:

recovered true pairs / total true pairs

This measurement should be performed using the training ground truth before changing the blocking rules.

## Measuring runtime and memory

Use measure_blocking_performance() to measure blocking runtime, peak Python-tracked memory, and the number of generated candidate pairs.

It returns:

- runtime_seconds
- peak_memory_mb
- candidate_pairs

Memory is measured using Python tracemalloc. This represents Python-tracked peak memory and does not represent total process memory usage.


## Assumptions

- Inputs have already been normalized by src/normalization/.
- name_core is the primary blocking field.
- Empty name_core values do not generate candidates.
- Source 1 is compared with one target source at a time.
- Source 2 and Source 3 can therefore be processed separately.
- country_norm is not used as a blocking gate.
- Blocking generates candidates only; downstream components perform similarity and matching.

## Limitations

- Current blocking uses exact-name and three-character prefix keys only.
- Address-based blocking is not currently implemented.
- Phonetic, token, n-gram, or embedding blocking is not currently implemented.
- Prefix blocks larger than max_prefix_block_size are skipped.
- Skipping large prefix blocks can reduce blocking recall.
- Actual blocking recall and large-dataset runtime should be measured against the training dataset before changing blocking rules.
- Dataset paths are supplied by the caller; no local dataset path is hard-coded in the blocking module.

## Testing

The blocking module currently has eight unit tests covering:

- exact-name candidate generation
- prefix blocking
- candidate-volume metrics
- runtime and memory measurement
- blocking recall
- missing required columns
- empty names
- large-prefix-block protection

The complete repository test suite currently passes all tests.
