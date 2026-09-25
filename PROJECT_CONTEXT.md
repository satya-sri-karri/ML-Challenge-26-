\# Amazon ML Challenge 2026 — Project Context



\## Problem



Business Entity Resolution across multiple data sources.



The project matches entities from Source 1 against corresponding entities

in Source 2 and Source 3.



Source 1 is the reference/deduplicated source.



\---



\## Dataset Structure



Each team member keeps the dataset locally.



```text

dataset/

├── train/

│   ├── train\_ground\_truth.tsv

│   ├── train\_source1.tsv

│   ├── train\_source2.tsv

│   └── train\_source3.tsv

│

└── test/

&#x20;   ├── test\_source1.tsv

&#x20;   ├── test\_source2.tsv

&#x20;   └── test\_source3.tsv

