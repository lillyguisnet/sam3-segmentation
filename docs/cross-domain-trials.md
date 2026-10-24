# Five non-corn trials

Read the [full report, configurations and limitations](../skills/sam3-segmentation/references/cross-domain-trials.md).
It lives inside the self-contained skill so installed copies retain the lessons.

The same SAM 3 implementation was run on a dog, jars, pinned insects, mushroom
caps and cattle, using explicit image-specific settings. Useful masks appeared,
but semantic false positives and parent/part duplicates remained. More masks did
not establish better annotation. None of these outputs is accepted ground truth.

[Scalar evidence without photographs or masks](../tests/evidence/cross-domain-trials.json)
