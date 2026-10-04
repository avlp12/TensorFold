# Spark TP4 branch

I carry the GLM-5.3-Flash TP4 loader and communicator changes on TensorFold v0.6.5 (609ca419). I keep the full EXL3 checkpoint unchanged. Rank-local tensor shapes, capacity accounting, metadata gathers and request doorbells use the actual world size. A TP2 pre-split checkpoint is rejected for TP4.

Validation: 7 host split/gather/doorbell tests pass. CLI topology and worker-follow regressions also passed (67 passed, 4 skipped). Four-rank all-gather/local-reduction hardware checks and the complete model matrix passed: 38 cells, 24 measured cells, 82 requests, six basic answers. Startup was73.36s. See the attached validation JSON for measured medians. I do not claim general quality or batch invariance. This branch retains the serial GLM CUDA request path; it does not claim batched GLM serving. I am evaluating [upstream PR243](https://github.com/ashhart/TensorFold/pull/243) separately because it is a large draft change.
