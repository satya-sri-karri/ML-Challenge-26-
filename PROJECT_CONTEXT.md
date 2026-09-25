# Amazon ML Challenge 2026 — Project Context

## Problem

Business Entity Resolution across multiple data sources.

The project matches entities from Source 1 against corresponding entities in Source 2 and Source 3.

Source 1 is the reference/deduplicated source.

---

## Dataset Structure

Each team member keeps the dataset locally.

The dataset must NOT be uploaded to GitHub.

```text
dataset/
├── train/
│   ├── train_ground_truth.tsv
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_source3.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv



Proposed Architecture

Raw Source 1 / Source 2 / Source 3
|
v
Normalization
|
v
Blocking / Candidate Generation
|
v
Candidate Pairs
|
v
Stage-A Pairwise ML Model
|
v
Stage-B Borderline Verification
|
v
Graph Consistency
|
v
Threshold Calibration
|
v
Final Matching Results



Repository Structure
ML-Challenge-26-/
│
├── src/
│   ├── normalization/
│   ├── blocking/
│   ├── features/
│   ├── matching/
│   │   ├── stage_a/
│   │   ├── stage_b/
│   │   └── consistency/
│   └── pipeline.py
│
├── notebooks/
├── utils/
├── models/
├── output/
│
├── README.md
├── PROJECT_CONTEXT.md
├── Documentation_template.md
├── requirements.txt
└── .gitignore
The dataset/ directory exists only locally and is excluded by .gitignore.

Team Responsibilities
Person A — Normalization + EDA

Branch:

feature/normalization

Owns:

src/normalization/

Responsibilities:

inspect source schemas
perform dataset profiling
normalize names
normalize addresses
handle Unicode and punctuation
handle missing values
create reusable normalized fields
define the shared normalized-data interface
document assumptions and limitations

Person A must NOT implement blocking, Stage-A ML, Stage-B verification, or graph consistency.

Person B — Blocking / Candidate Generation

Branch:

feature/blocking

Owns:

src/blocking/

Responsibilities:

candidate generation
scalable blocking
candidate pair creation
blocking recall measurement
candidate volume measurement
runtime and memory measurement

Person B should consume the normalization interface created by Person A.

Person C — Features + Stage-A ML

Branch:

feature/stage-a

Owns:

src/features/
src/matching/stage_a/

Responsibilities:

pairwise similarity features
candidate-pair training data
model training
Stage-A GBDT model
evaluation
error analysis
inference interface

Person C should consume candidate pairs produced by Person B and normalized data produced by Person A.

Person D — Stage-B + Consistency + Integration

Branch:

feature/stage-b

Owns:

src/matching/stage_b/
src/matching/consistency/
src/pipeline.py

Responsibilities:

borderline candidate verification
Stage-B model/LLM verification where appropriate
graph consistency
threshold calibration
end-to-end pipeline integration
submission validation

Person D should consume the outputs of Persons A, B, and C rather than rebuilding their components.

Important Constraints
Do not upload the dataset to GitHub.
Do not use external business/entity lookup.
Do not hard-code local dataset paths.
The pipeline must scale to millions of records.
Avoid O(n²) all-pairs comparisons.
Do not overwrite another team member's component.
Keep interfaces between components documented.
Test code locally before pushing.
Do not commit generated large outputs.
Do not commit trained model files unless explicitly required.
Keep the final pipeline reproducible.
Do not tune against the hidden test set.
Shared Component Principle

Every component must document:

input format
output format
column names
data types
file locations
expected behavior
assumptions
limitations

Downstream components must consume the interfaces created by upstream components instead of independently rebuilding them.

Git Workflow

Each team member works on a separate branch:

feature/normalization
feature/blocking
feature/stage-a
feature/stage-b

The main branch contains the integrated project.

Workflow:

1. Pull latest main
2. Work on personal feature branch
3. Test locally
4. Commit changes
5. Push feature branch
6. Create Pull Request
7. Review
8. Merge into main

Do not directly modify another person's branch.

Claude Collaboration Rules

Every Claude account must:

Read this file before implementing anything.
Inspect the existing repository before creating files.
Respect the assigned component ownership.
Reuse existing interfaces instead of duplicating functionality.
Avoid unnecessary dependencies.
Avoid rewriting working code without evidence.
Explain important architectural decisions.
Provide exact files changed.
Provide commands required to run and test the implementation.
Report limitations and assumptions.