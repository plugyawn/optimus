# Prime GPU Ledger

Use this ledger for every Prime GPU allocation used by Optimus.

## Active Pods

No active Prime pods. Last checked after terminating `ff41f93eff0d4029819d4788a4d6ab45` at
`2026-05-23T23:25Z`; `prime pods list --plain` reported `Compute Pods (Total: 0)`.

Recent 4x create attempt at `2026-05-23T22:37:36Z`:

- pod_id: ff41f93eff0d4029819d4788a4d6ab45
  name: oc-main-optimus-l40sx4-20260523-2237
  owner: current-agent
  purpose: Optimus P1024/P4096 GPU suite rerun on at least 4 GPUs, using tensor parallel size 4 with full throughput/scaling report if provisioning succeeds.
  gpu: 4x L40S 48GB PCIe
  provider_region: crusoecloud US
  availability_id: 104d20
  image: ubuntu_22_cuda_12
  disk_gb: 128
  price_per_hour: $4.00
  branch_commit: main@d4b6009 with dirty worktree from active Optimus refactor
  created_at: 2026-05-23T22:39Z
  expected_stop: terminate immediately if no IP/SSH after provisioning timeout, or after suite completion/failure.
  planned_command: `prime pods create --id 104d20 --name oc-main-optimus-l40sx4-20260523-2237 --image ubuntu_22_cuda_12 --disk-size 128 --yes --plain`
  status: completed; SSH verified at `ubuntu@160.211.45.215 -p 22`; GPU check showed 4x NVIDIA L40S 46GB; Optimus smoke and P1024/P4096 TP=4 suite completed; artifacts fetched to `results/prime_runs/l40sx4_20260523_2237/results`; strict local validation passed.
  ssh: ubuntu@160.211.45.215 -p 22
  termination_policy: current-agent owns this attempt; terminate any successfully created pod on failed setup, completed run, or idle state.
  terminated_at: 2026-05-23T23:25Z
  cleanup_check: `prime pods terminate ff41f93eff0d4029819d4788a4d6ab45 --yes --plain`; `prime pods list --plain` reported 0 active pods after termination.

Recent 8x create attempt at `2026-05-23T22:34:59Z`:

- pod_id: bd4c21e28ae24d90b91b2ba379fde160
  name: oc-main-optimus-a100x8-20260523-2234
  owner: current-agent
  purpose: Optimus 8xA100 P1024/P4096 GPU suite rerun with tensor parallel size 8, full throughput plots, and release-grade scaling evidence if provisioning succeeds.
  gpu: 8x A100 40GB PCIe
  provider_region: lambdalabs US
  availability_id: 30a1c3
  image: ubuntu_22_cuda_12
  price_per_hour: $15.92
  branch_commit: main@d4b6009 with dirty worktree from active Optimus refactor
  created_at: 2026-05-23T22:35Z
  expected_stop: terminate immediately if no IP/SSH after provisioning timeout, or after suite completion/failure.
  planned_command: `prime pods create --id 30a1c3 --name oc-main-optimus-a100x8-20260523-2234 --image ubuntu_22_cuda_12 --yes --plain`
  status: terminated after provisioning stayed without IP/SSH; installation status was `FINISHED`, but pod remained `PROVISIONING` with `IP N/A` and `SSH N/A`.
  termination_policy: current-agent owns this attempt; terminate any successfully created pod on failed setup, completed run, or idle state.
  terminated_at: 2026-05-23T22:37Z
  cleanup_check: `prime pods list --plain` reported 0 active pods after termination.

## Run Log

| date | pod id | gpu type/count | command | status | shutdown evidence |
| --- | --- | --- | --- | --- | --- |
| 2026-05-23T22:39Z | `ff41f93eff0d4029819d4788a4d6ab45` | 4x L40S 48GB PCIe | `TENSOR_PARALLEL_SIZE=4 POPULATIONS="1024 4096" BENCH_ADAPTERS=8 RUN_HALVING=0 bash scripts/run_optimus_gpu_suite.sh` | completed Optimus smoke plus P1024/P4096 TP=4 GPU suite; artifacts fetched to `results/prime_runs/l40sx4_20260523_2237/results`; strict validation passed with P1024/P4096 and systems report | `prime pods terminate ff41f93eff0d4029819d4788a4d6ab45 --yes --plain`; `prime pods list --plain` reported 0 pods |
| 2026-05-23T22:35Z | `bd4c21e28ae24d90b91b2ba379fde160` | 8x A100 40GB PCIe | `prime pods create --id 30a1c3 --name oc-main-optimus-a100x8-20260523-2234 --image ubuntu_22_cuda_12 --yes --plain` | terminated after provisioning stayed at `PROVISIONING` with installation `FINISHED` but no IP/SSH | `prime pods terminate ... --yes`; `prime pods list --plain` reported 0 pods |
| 2026-05-23T21:34Z | `d93eaae2f80b4246b0eb3754bc2f0181` | 2x L40S 48GB PCIe | `TENSOR_PARALLEL_SIZE=2 POPULATIONS="1024 4096" BENCH_ADAPTERS=8 RUN_HALVING=0 bash scripts/run_optimus_gpu_suite.sh` | completed P1024/P4096 Optimus GPU suite with vLLM 0.9.2/Torch 2.7.0; artifacts fetched to `results/prime_runs/l40sx2_20260523_2134/results`; strict local validation passed for populations 1024,4096 and bench adapters 8 | `prime pods terminate d93eaae2f80b4246b0eb3754bc2f0181 --yes`; `prime pods list --plain` reported 0 pods |
| 2026-05-23T20:19Z | `d063ba9ea2774bcba3ec6c47a8898713` | 8x A100 40GB PCIe | `prime pods create --id 961209 --name oc-main-optimus-a100x8-20260523-2019 --image ubuntu_22_cuda_12 --yes --plain` | terminated after >6 min stuck provisioning with no IP/SSH; installation was `FINISHED` | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |
| 2026-05-23T20:12Z | `2339259f1d7849b793dd233b2138f02e` | 8x A100 40GB PCIe | `prime pods create --id 30a1c3 --name oc-main-optimus-a100x8-20260523-2012 --image cuda_12_4_pytorch_2_6 --yes --plain` | terminated after >6 min stuck provisioning with no IP/SSH and installation `PENDING` | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |
| 2026-05-23T19:59Z | n/a | 8x A100 80GB SXM4 | `prime pods create --id da617d --image ubuntu_22_cuda_12 ...` | create failed: Vultr HTTP 400 `Unable to complete the request. Please try again later.` | no pod created |
| 2026-05-23T19:59Z | n/a | 8x L40 48GB PCIe | `prime pods create --id ba3d6f --image ubuntu_22_cuda_12 ...` | create failed: Vultr HTTP 400 `Unable to complete the request. Please try again later.` | no pod created |
| 2026-05-23T19:58Z | n/a | 8x L40 48GB PCIe | `prime pods create --id ba3d6f --image cuda_12_4_pytorch_2_6 ...` | create failed: Vultr does not support image `CUDA_12_4_PYTORCH_2_6` | no pod created |
| 2026-05-23T19:54Z | n/a | 8x A100 40GB PCIe | `prime pods create --id 9d1b02 --image cuda_12_4_pytorch_2_6 ...` | create failed: no valid GPU configuration found | no pod created |
| 2026-05-23T19:53Z | n/a | 1x GH200 96GB SXM5 | `prime pods create --id 0f1fb9 --image cuda_12_4_pytorch_2_6 ...` | create failed: no valid GPU configuration found | no pod created |
| 2026-05-23T19:52Z | n/a | 8x A100 80GB SXM4 | `prime pods create --id da617d --image cuda_12_4_pytorch_2_6 ...` | create failed: no valid GPU configuration found | no pod created; prior `prime pods list` reported 0 pods |
| 2026-05-23T19:44Z | `10a49888ffaf48b9922ec2d2aa664cd4` | 8x A100 40GB PCIe | `prime pods create --id 961209 --image cuda_12_4_pytorch_2_6 ...` | terminated after >5 min stuck provisioning with no IP/SSH | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |
| 2026-05-23T19:43Z | `12e41f2e11e44389a76d5726a93e392a` | 1x L40S 48GB PCIe | remote Optimus bootstrap smoke and `scripts/remote/optimus_prime_smoke.sh` | passed after installing `g++-12` and `libcurand-dev-13-0`; fetched `results/prime_l40s_smoke_20260523_1924/results/prime_smoke` | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |
| 2026-05-23T19:21Z | n/a | 1x A100 40GB | `prime pods create --id b203f6 ...` | create failed: no valid GPU configuration found | no pod created; `prime pods list` reported 0 pods |
| 2026-05-23T19:20Z | `31150a266b5548c0b78b5bbedffabfce` | 8x A100 40GB | `prime pods create --id 9d1b02 ...` | terminated after >6 min stuck provisioning with no IP/SSH | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |
| 2026-05-23T19:13Z | n/a | 8x A100 40GB | `prime pods create --id 961209 ...` | create failed: HTTP 503 stale Lambda availability | no pod created |
| 2026-05-23T19:02Z | `14f1dcd8bb7e478fae01350b716a8d77` | 1x L40S_48GB | planned Optimus smoke bootstrap | terminated before workload while public API/docs were being refactored | `prime pods terminate ... --yes`; `prime pods list` reported 0 pods |

## Availability Preflight

Checked at `2026-05-23T22:34:59Z`; no active pods were listed. Current 8-GPU
availability returned three 8xA100 40GB Lambda listings and one 8xA100 80GB
Vultr listing. Selected the cheapest adequate 8xA100 40GB US listing with
`ubuntu_22_cuda_12`, because the same US availability id was previously tested
only with the PyTorch image while the current remote bootstrap now pins its own
runtime stack.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | selected current attempt with `ubuntu_22_cuda_12` |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck with no IP/SSH |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously timed out with no IP/SSH |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | larger fallback, previously failed create on Vultr |

Checked at `2026-05-23T22:37:36Z` after the 8xA100 attempt was terminated;
no 4xA100 listings were available, but one 4xL40S listing was available and
selected as the user's allowed 4x fallback.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `104d20` | 4x L40S 48GB PCIe | crusoecloud | US | `$4.00` | selected 4x fallback |

Checked at `2026-05-23T19:12Z`; no active pods were listed. Selected `961209`
for the Optimus 8xA100 smoke and P1024/P4096 GPU suite if bootstrap passes.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | selected current 8xA100 option |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | same price, alternate region |

Checked at `2026-05-23T19:21Z`; 8xA100 create/provisioning failed, no active
pods remained, and only 1xA100 was available for a cheap bootstrap smoke.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `b203f6` | 1x A100 40GB SXM4 | lambdalabs | US | `$1.99` | selected bootstrap smoke fallback |

Checked at `2026-05-23T19:22Z`; A100 fallback also failed at create time, so
selected the cheapest L40S for bootstrap-only validation.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `2f3fcb` | 1x L40S 48GB PCIe | massedcompute | US | `$0.82` | selected remote bootstrap smoke only |

Checked at `2026-05-23T20:12Z`; no active pods were listed. Selected the new
Lambda US 8xA100 listing for the Optimus P1024/P4096 GPU suite attempt.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | selected current 8xA100 option |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | alternate region |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | alternate region |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | larger fallback, previously stale |

Checked at `2026-05-23T20:19Z`; no active pods were listed after terminating
the stale US A100 pod. Selected the Lambda DE 8xA100 listing with
`ubuntu_22_cuda_12` to test whether the PyTorch image was the provisioning
blocker.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | selected alternate image attempt |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | alternate region |
| `730864` | 8x B200 180GB SXM6 | lambdalabs | US | `$53.52` | available but too expensive for first retry |

Checked at `2026-05-23T20:30Z`; no active pods were listed. Available 8-GPU
inventory was still limited to previously stale A100/L40/A10080 paths plus an
8xB200 option at `$53.52/hr`. No new pod was launched without explicit approval
for the higher-cost B200 tier.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `ba3d6f` | 8x L40 48GB PCIe | vultr | US | `$13.37` | previously failed create on Vultr |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously stuck/failed |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |
| `730864` | 8x B200 180GB SXM6 | lambdalabs | US | `$53.52` | fresh but high-cost option |

Checked again at `2026-05-23T20:33Z`; no active pods were listed and the same
8-GPU inventory remained. No pod was launched.

Checked again at `2026-05-23T20:36Z`; no active pods were listed and the same
8-GPU inventory remained. No pod was launched.

Checked at `2026-05-23T18:51:42Z`; no pods were launched.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | cheapest adequate 8xA100 listed |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | same price, alternate region |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | larger memory/headroom |
| `0f1fb9` | 1x GH200 96GB SXM5 | lambdalabs | US | `$2.29` | cheap smoke-test option |

Checked at `2026-05-23T20:40:54Z`; no active pods were listed. The available
8-GPU inventory was unchanged except the high-cost B200 option was not returned
by this filtered availability call. No pod was launched.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `ba3d6f` | 8x L40 48GB PCIe | vultr | US | `$13.37` | previously failed create on Vultr |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously stuck/failed |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |

- pod_id: ce7872097f39466b81eb68d7f34f17e9
  name: oc-optimus-p4096-jp-a100-20260523-2045
  owner: current-agent
  purpose: Optimus P1024/P4096 GPU suite on 8xA100; start with provisioning/SSH smoke, then run `scripts/run_optimus_gpu_suite.sh` if usable.
  gpu: 8x A100 40GB PCIe
  provider_region: lambdalabs JP
  availability_id: 9d1b02
  image: ubuntu_22_cuda_12
  price_per_hour: $15.92
  created_at: 2026-05-23T20:45:41Z
  expected_stop: terminate immediately if no IP/SSH after provisioning timeout, or after suite completion/failure.
  status: terminated after provisioning timeout; no IP/SSH assigned after 12 polls from 20:46:10Z to 20:52:08Z
  termination_policy: current-agent owns this pod; terminate on failed setup, completed run, or idle state.
  terminated_at: 2026-05-23T20:52:44Z
  cleanup_check: `prime pods list` reported 0 active pods after termination.

Checked at `2026-05-23T20:58:40Z`; no active pods were listed. Availability
again showed the same stale 8-GPU pool, with the US Lambda A100 listing returned
to the list. No pod was launched after the JP timeout because the remaining
options are the same US/DE/JP Lambda A100 paths that have already failed to
reach IP/SSH or the Vultr paths that previously failed create.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `ba3d6f` | 8x L40 48GB PCIe | vultr | US | `$13.37` | previously failed create on Vultr |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | previously stuck provisioning |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | just timed out with no IP/SSH |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |

Checked at `2026-05-23T21:09:33Z`; no active pods were listed. Availability
again returned only previously failed 8xA100/Vultr options for the requested
8-GPU class. No pod was launched because all adequate options matched provider
paths that had already failed create or failed to reach IP/SSH during this
Optimus run series.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | previously stuck provisioning |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously timed out with no IP/SSH |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |

Checked at `2026-05-23T21:15:29Z`; no active pods were listed. The same
8xA100 Lambda and 8xA10080 Vultr availability IDs were still returned. No pod
was launched because this was the same stale provider inventory already
exercised during the current Optimus GPU-suite attempts.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | previously stuck provisioning |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously timed out with no IP/SSH |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |

Checked at `2026-05-23T21:18:58Z`; no active pods were listed. The 8xA100
availability call again returned only the same Lambda US/DE/JP listings that
previously failed provisioning or IP/SSH assignment. No pod was launched.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | previously stuck provisioning |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously timed out with no IP/SSH |

Checked at `2026-05-23T21:26:01Z`; no active pods were listed. The 8xA100
availability calls returned only the same Lambda 40GB listings and Vultr 80GB
listing already exercised in this run series. No pod was launched.

| availability id | gpu | provider | region | price/hour | note |
| --- | --- | --- | --- | ---: | --- |
| `30a1c3` | 8x A100 40GB PCIe | lambdalabs | US | `$15.92` | previously stuck provisioning |
| `961209` | 8x A100 40GB PCIe | lambdalabs | DE | `$15.92` | previously stuck provisioning |
| `9d1b02` | 8x A100 40GB PCIe | lambdalabs | JP | `$15.92` | previously timed out with no IP/SSH |
| `da617d` | 8x A100 80GB SXM4 | vultr | US | `$22.40` | previously failed create on Vultr |

| 2026-05-23T21:28Z | n/a | 2x A100 80GB PCIe | `prime pods create --id 70eb73 --name oc-main-optimus-a100x2-20260523-2128 --image cuda_12_4_pytorch_2_6 --yes --plain` | create failed: MassedCompute does not support image `CUDA_12_4_PYTORCH_2_6` | no pod created |

- pod_id: f22d8019bee24ebc87ae5f8e6f87119e
  name: oc-main-optimus-a100x2-20260523-2129
  owner: current-agent
  purpose: Optimus P1024/P4096 GPU suite fallback on at least 2 GPUs; start with provisioning/SSH smoke, then run the suite with tensor parallel size 2 if usable.
  gpu: 2x A100 80GB PCIe
  provider_region: massedcompute US
  availability_id: 70eb73
  image: ubuntu_22_cuda_12
  price_per_hour: $2.40
  created_at: 2026-05-23T21:29Z
  expected_stop: terminate immediately if no IP/SSH after provisioning timeout, or after suite completion/failure.
  status: terminated after provisioning timeout; no IP/SSH assigned from 2026-05-23T21:29:06Z through 2026-05-23T21:33:23Z while installation status was `FINISHED`.
  termination_policy: current-agent owns this pod; terminate on failed setup, completed run, or idle state.
  terminated_at: 2026-05-23T21:33Z
  cleanup_check: `prime pods list` reported 0 active pods after termination.

- pod_id: d93eaae2f80b4246b0eb3754bc2f0181
  name: oc-main-optimus-l40sx2-20260523-2134
  owner: current-agent
  purpose: Optimus P1024/P4096 GPU suite fallback on at least 2 GPUs; start with provisioning/SSH smoke, then run the suite with tensor parallel size 2 if usable.
  gpu: 2x L40S 48GB PCIe
  provider_region: crusoecloud US
  availability_id: 2126e0
  image: ubuntu_22_cuda_12
  disk_gb: 512
  price_per_hour: $2.00
  created_at: 2026-05-23T21:34Z
  expected_stop: terminate immediately if no IP/SSH after provisioning timeout, or after suite completion/failure.
  status: completed and terminated; SSH verified; Prime installation `FINISHED`; GPU check showed 2x NVIDIA L40S; P1024 and P4096 Optimus GPU suite completed with `TENSOR_PARALLEL_SIZE=2`, `POPULATIONS="1024 4096"`, `BENCH_ADAPTERS=8`, `RUN_HALVING=0`, and vLLM 0.9.2/Torch 2.7.0 CUDA 12.6. Artifacts fetched to `results/prime_runs/l40sx2_20260523_2134/results`; strict local validation passed; report PNGs were valid.
  termination_policy: current-agent owns this pod; terminate on failed setup, completed run, or idle state.
  terminated_at: 2026-05-23T22:16Z
  cleanup_check: `prime pods list --plain` reported 0 active pods after termination.

## Required Entry Fields

- Launch timestamp.
- Prime pod id and region/provider metadata.
- GPU type and count.
- Exact branch/commit and dirty-worktree note.
- Exact command and output root.
- Expected stop condition.
- Shutdown timestamp or explicit blocker.
