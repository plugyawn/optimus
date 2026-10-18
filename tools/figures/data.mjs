// Measurements plotted in docs/results.md.
//
// Raw run artifacts live under the ignored results/ tree, so every number here
// is copied from the measurement tables in KERNEL.md (or the experiments/ logs
// they summarize). Each block names the table it came from.

// KERNEL.md, "Current A100 p128 bridge evidence" -> "Bridge p128 timing".
// A100-SXM4-80GB, p128, rank 64, q/v targets, vLLM LoRA-kernel bridge.
export const bridgeTiming = {
  totalSeconds: 97.71,
  lazyDeltaSeconds: 49.34,
  parts: [
    { label: "Model rollout, scheduling, scoring", seconds: 97.71 - 49.34, lazy: false },
    { label: "vLLM LoRA kernels (shrink + expand)", seconds: 20.56, lazy: true },
    { label: "Other hook work (dispatch, delta add)", seconds: 49.34 - 20.56 - 10.14 - 8.22, lazy: true },
    { label: "Factor-stack construction", seconds: 10.14, lazy: true },
    { label: "Row-candidate metadata", seconds: 8.22, lazy: true },
  ],
};

// KERNEL.md, "Current L40S stateless-counter expand evidence" (A/B rows).
// Triton, L40S-48GB, 16 candidates. Latency in milliseconds.
export const expandAB = [
  { rows: 64, rank: 64, out: 1024, storedMs: 0.0465, counterMs: 0.05 },
  { rows: 256, rank: 64, out: 1024, storedMs: 0.185, counterMs: 0.0935 },
  { rows: 256, rank: 128, out: 4096, storedMs: 1.281, counterMs: 0.4008 },
  { rows: 512, rank: 128, out: 4096, storedMs: 2.7291, counterMs: 0.7207 },
];

// KERNEL.md, "Current L40S end-to-end triton-counter evidence" and
// "Current L40S end-to-end triton-counter-inplace evidence".
// L40S-48GB, p128, 8 prompts, rank 64 q/v, per-phase sync timer.
export const p128Throughput = [
  { label: "vLLM LoRA-kernel bridge", detail: "candidate batch 16", candSec: 6.11, ours: false },
  { label: "Triton counter expand", detail: "candidate batch 32", candSec: 12.422, ours: true },
  { label: "Triton counter, in-place add", detail: "candidate batch 64", candSec: 13.924, ours: true },
];
// Same in-place configuration measured with the host timer
// (KERNEL.md, "Timing-mode correction for the same p128 cbs64 shape").
export const p128HostTimerCandSec = 16.576;
