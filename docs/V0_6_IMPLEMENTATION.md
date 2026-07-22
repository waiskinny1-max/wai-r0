# WAI-R0 v0.6 — Ground Truth

v0.6 turns the v0.5 evidence harness into a complete small-model learning lineage:

1. train and hash a deterministic tokenizer;
2. audit and compile data into verified memory-mapped shards;
3. train with exact sampler/checkpoint lineage;
4. resume without changing data order or optimizer state;
5. evaluate language, context, generation, and systems behavior;
6. generate through the exact training chat template;
7. register artifacts and plan bounded experiments;
8. verify the repository and package before release.

The release deliberately does not claim a generally capable model. Its executed CPU baseline learned narrow copy/sort templates while failing unseen arithmetic and classification cases. That is useful evidence: the stack can now distinguish successful training mechanics from unsupported capability claims.

The next evidence boundary is real target-GPU execution, larger provenance-reviewed data, tokenizer-independent language metrics, and matched trained-model comparisons for MLA-lite, recurrence, and MoE.
