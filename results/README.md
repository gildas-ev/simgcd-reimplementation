# Results

Per-seed metrics (`metrics.jsonl`) and training logs (`train.log`, that also includes the git commit, the dirty flag and the full config).

Traceability:
- seed 42 logs commit `2d37790` is no longer in the published history (rewritten afterwards)
- seeds 42 and 56 were resumed with a dirty working tree. The training code is identical
- the pretrained checkpoints of release v1.1 were evaluated with the code at tag `v1.1`, which reproduces the README table
