# Future experiments

Ideas worth running later, with what we measured and how to test them.

## Decode camera images after the training-batch prefetch

**Status:** not started (deferred 2026-11-15).

**Why.** Camera TRAIN jobs on `donut_camera` sit at about 17.9 GB resident in
sim-controller (WSL2 limit 23.5 GB) once training starts, and the figure stays
flat while the online buffer fills. The replay tables are not the cause.
Reverb compresses chunks, so measured on real demo frames a stored step costs:

| Image storage | Measured per step | Raw tensor size |
|---|---|---|
| float32 | ~23 KB | 84,672 B |
| uint8 (`store_images_uint8`) | ~17 KB | 21,168 B |

100k demos plus a full 75k online buffer is therefore about 3 GB. The likely
remainder is the training input pipeline: `replay.make_local_replay` calls
`.prefetch(50)` on both the online and the demo dataset, and with uint8
storage `_Uint8ImageReplayBuffer._as_dataset` decodes to float32 *before* that
prefetch. Each queued batch is 256 x 2 steps x 84,672 B, about 43 MB, so 50 of
them per dataset is about 2.2 GB, about 4.3 GB for both. This split is
estimated, not profiled.

**Change.** Keep the prefetched batches in uint8 and decode afterwards (prefetch,
then map `decode_image`), or cut the prefetch depth from 50. The AWAC demo
iterator and the critic warm-up dataset call `demo_replay.as_dataset` directly
and must still receive float32 images.

**Expected effect.** Roughly 3 GB less resident memory with uint8 storage.
Learning should be unchanged: the agent sees the same float32 [0, 1] images,
only the queue holds them more compactly.

**How to test.**
1. Run one camera TRAIN job with the change and log trainer RSS (largest
   `python -u /python_ws/src/robotaxi.py` process) every minute for the first
   ~2,000 steps. Compare with the ~17.9 GB baseline from job
   `6afa2875832027d3aaf0d807`.
2. Check the per-step learner time in the `TRAIN end:` lines
   (`learner_sec=`) did not rise, i.e. the GPU is not starved by a later decode.
3. If memory drops as expected, profile what is left (`tracemalloc` or a heap
   snapshot) before raising buffer sizes further.
