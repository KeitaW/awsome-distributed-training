---
name: Add SLURM CI Support
about: Track the work needed to add CI coverage for SLURM-based test cases
title: "Add SLURM CI support for uncovered test cases and benchmarks"
labels: enhancement, ci/cd, slurm
---

## Summary

The current SLURM CI pipeline only covers **2 out of 14+ test cases** with dedicated workflows. Multiple test cases ship SLURM sbatch scripts and training configurations but have **no dedicated CI validation**, relying solely on a generic PR review fallback that does not perform framework-specific regression testing.

## Current State

### Existing SLURM CI Workflows
| Workflow | Scope | Clusters | Details |
|---|---|---|---|
| `fsdp-regression-test-container.yml` | FSDP (container) | p5, p5-smhp | Docker build + enroot conversion |
| `fsdp-regression-test-venv.yml` | FSDP (venv) | p5, p5-smhp | 5 models (llama2 7B/13B/70B, llama3.1 8B/70B), 375 min timeout |
| `megatron-ci-slurm.yaml` | Megatron-LM | p5 | Build + enroot conversion only (no training run) |
| `pr-review-and-slurm-test.yml` | Generic fallback | SLURM (8 nodes) | Detects pytest/run.sh/main.py, not framework-specific |

### Test Cases with SLURM Content but NO Dedicated CI

#### PyTorch Test Cases

1. **PyTorch DDP** - `3.test_cases/pytorch/ddp/slurm/`
   - `1.venv-train.sbatch`, `3.container-train.sbatch`
   - Foundational distributed training pattern; baseline for all other frameworks
   - No dedicated CI

2. **DeepSpeed** - `3.test_cases/pytorch/deepspeed/`
   - `1.build-image.sbatch` + multi-stage Llama fine-tuning pipeline
   - `finetune_hf_llama/1.convert-weights-to-hf.sbatch`
   - `finetune_hf_llama/scripts/finetune_llama.sbatch`
   - `finetune_hf_llama/2.convert-weights-to-mega-ds.sh` (submits sbatch)
   - `finetune_hf_llama/3.finetune-llama.sh` (submits sbatch)
   - Production-grade Llama fine-tuning with weight conversion pipeline
   - No dedicated CI

3. **MosaicML Composer** - `3.test_cases/pytorch/mosaicml-composer/`
   - `mpt/1.c4-preprocess.sbatch`, `mpt/2.train-mpt-manual-distributed.sbatch`
   - `stable-diffusion/multi-node/2.train.sbatch`
   - MPT training + Stable Diffusion multi-node training
   - No dedicated CI

4. **TorchTitan** - `3.test_cases/pytorch/torchtitan/slurm/`
   - `0.create_conda_env.sh`, `1.llama_3_8b_torchtitan.sh`
   - Llama 3.1 8B with float8 precision, torch.compile, FSDP float8 all-gather
   - Documented 15.92% throughput improvement (MFU 39.73% -> 46.06%)
   - No dedicated CI

5. **Optimum-Neuron (Llama3)** - `3.test_cases/pytorch/optimum-neuron/llama3/slurm/`
   - 7-stage fine-tuning pipeline: create_env -> download_model -> compile -> finetune -> consolidate -> merge_lora -> inference
   - AWS Trainium / Neuron optimization
   - No dedicated CI

6. **NeuronX-Distributed (Llama3)** - `3.test_cases/pytorch/neuronx-distributed/llama3/slurm/`
   - 3D parallelism (tensor + pipeline + data) on 16x trn1.32xlarge
   - Llama3-70B continual pretraining with auto-resume on HyperPod
   - 256-checkpoint sharding for 70B model
   - No dedicated CI

7. **TRL GRPO** - `3.test_cases/pytorch/trl/grpo/`
   - `train.sbatch`
   - Reinforcement learning with Group Relative Policy Optimization
   - No dedicated CI

8. **Picotron SmolLM** - `3.test_cases/pytorch/picotron/SmolLM-1.7B/slurm/`
   - `train.sbatch`
   - Small language model training
   - No dedicated CI

9. **VeRL** - `3.test_cases/pytorch/verl/`
   - SLURM-based RL training scripts
   - No dedicated CI

10. **SMHP ESM2** - `3.test_cases/23.SMHP-esm2/`
    - `2.train_ddp.sh`, `3.train_fsdp.sh`, `4.train_docker_dpp.sh`
    - Structural biology ESM2 model with DDP/FSDP/Docker variants
    - No dedicated CI

#### Megatron Test Cases

11. **NeMo** - `3.test_cases/megatron/nemo/slurm/`
    - `venv.sh`, `run.py` (Python SLURM launcher), `env_vars.json`
    - Supports GPT3 (126M-175B) and Llama (7B-70B) via configuration
    - No dedicated CI

12. **BioNeMo** - `3.test_cases/megatron/bionemo/`
    - `bionemo_2.5/train-esm.sbatch`
    - ESM protein model training
    - No dedicated CI

#### Other Frameworks

13. **JAX** - `3.test_cases/jax/`
    - `jax.sbatch` (384 processes, 8 GPU/node)
    - Only non-PyTorch framework in the repo
    - No dedicated CI

### Micro-Benchmarks (All Uncovered)

14. **NCCL Tests** - `micro-benchmarks/nccl-tests/slurm/`
    - `nccl-tests-ami.sbatch`, `nccl-tests-container.sbatch`
    - Topology-aware NCCL performance validation

15. **Expert Parallelism** - `micro-benchmarks/expert-parallelism/`
    - `pplx-garden.sbatch`, `uccl-pplx-garden.sbatch`, `pplx-kernels.sbatch`
    - Expert parallelism benchmarking

16. **NCCOM Tests** - `micro-benchmarks/nccom-tests/slurm/`
    - `nccom-tests.sbatch`
    - Collective communication benchmarking

17. **NVShmem** - `micro-benchmarks/nvshmem/slurm/`
    - `alltoall_latency.sbatch`, `shmem_put_bw_internode.sbatch`, `shmem_put_bw_intranode.sbatch`
    - NVShmem performance benchmarking

### Validation & Observability (All Uncovered)

18. **PyTorch Env Validation** - `4.validation_and_observability/1.pytorch-env-validation/`
    - `1.torch-screen.sbatch`
    - PyTorch environment validation on SLURM clusters

19. **GPU Cluster Healthcheck** - `4.validation_and_observability/2.gpu-cluster-healthcheck/slurm/`
    - `sbatch-intensive.sh` (intensive GPU healthcheck)
    - `sbatch-lightweight.sh` (minimal impact healthcheck)
    - `sbatch-quarantine-workflow.sh` (identifies and quarantines bad nodes)
    - NCCL allreduce tests, topology checks, quarantine automation

20. **Nsight Profiling** - `4.validation_and_observability/5.nsight/`
    - `fsdp-llama2/1.distributed-training.sbatch`
    - `nccl/0.nsight_nccl.sbatch`
    - `nemotron/1.nemotron.sbatch`
    - NVIDIA Nsight performance profiling for distributed training

## Coverage Summary

| Category | Total | With Dedicated CI | Coverage |
|---|---|---|---|
| Training test cases | 14 | 2 (FSDP, Megatron-LM) | 14% |
| Micro-benchmarks | 4 | 0 | 0% |
| Validation & observability | 3 | 0 | 0% |
| **Overall** | **21** | **2** | **10%** |

## Proposed Work Items

### Phase 1: High-Impact Training Frameworks
Extend the existing `fsdp-regression-test-container.yml` pattern to critical frameworks:

- [ ] **PyTorch DDP regression workflow** - foundational distributed training; baseline for all other frameworks
- [ ] **DeepSpeed regression workflow** - production Llama fine-tuning pipeline (multi-stage: convert -> train)
- [ ] **TorchTitan regression workflow** - latest PyTorch distributed patterns with float8 optimization
- [ ] **Complete Megatron-LM CI** - extend `megatron-ci-slurm.yaml` beyond build-only to include actual training runs

### Phase 2: AWS Trainium and NeMo
- [ ] **Optimum-Neuron regression workflow** - AWS Trainium fine-tuning pipeline (7-stage)
- [ ] **NeuronX-Distributed regression workflow** - 3D parallelism on trn1 clusters
- [ ] **NeMo SLURM regression workflow** - NeMo launcher-based training with GPT3/Llama configs

### Phase 3: Remaining Frameworks
- [ ] **MosaicML Composer regression workflow** - MPT + Stable Diffusion training
- [ ] **TRL GRPO regression workflow** - reinforcement learning training
- [ ] **BioNeMo regression workflow** - ESM protein model training
- [ ] **JAX regression workflow** - non-PyTorch distributed training
- [ ] **VeRL SLURM regression workflow** - RL training on SLURM
- [ ] **Picotron/SmolLM regression workflow** - small model training
- [ ] **SMHP ESM2 regression workflow** - structural biology DDP/FSDP variants

### Phase 4: Benchmarks and Validation
- [ ] **NCCL benchmark CI** - scheduled NCCL topology-aware performance tests
- [ ] **Expert parallelism benchmark CI** - validate MoE parallelism performance
- [ ] **GPU healthcheck CI** - automated cluster health validation before regression runs
- [ ] **Nsight profiling CI** - performance regression detection via profiling
- [ ] **PyTorch env validation CI** - pre-flight environment checks

### Phase 5: Static Validation (PR-level)
Add SLURM-specific validation to `pr-review-and-slurm-test.yml`:

- [ ] **SBATCH syntax validation** - validate `#SBATCH` directives (node counts, partitions, time limits)
- [ ] **Shell script linting** - `shellcheck` for all `.sh` and `.sbatch` scripts under `slurm/` directories
- [ ] **Environment variable consistency** - verify EFA, NCCL, and CUDA environment variables match documented minimums
- [ ] **Dockerfile validation** - lint Dockerfiles referenced by container-based sbatch scripts

## Implementation Notes

### Workflow Template Pattern
The existing SLURM CI follows a consistent pattern that new workflows should replicate:
1. SSH into SLURM cluster host (`p5en.smml.aiml.aws.dev`)
2. Transfer code via SCP
3. Build container (Docker -> enroot) or set up venv
4. Submit sbatch job and capture job ID
5. Monitor `squeue` for job completion (with timeout)
6. Collect logs via SCP
7. Upload logs as GitHub artifacts
8. Cleanup remote resources

### Path-Based Triggers
Each workflow should be scoped to its respective test case directory:
```yaml
on:
  push:
    branches: [ "main" ]
    paths:
      - '3.test_cases/<framework>/<test_case>/**'
  pull_request:
    paths:
      - '3.test_cases/<framework>/<test_case>/**'
```

### Cluster Requirements
- **GPU clusters**: p5 (H100), p5-smhp (SageMaker HyperPod) - for PyTorch/Megatron test cases
- **Neuron clusters**: trn1 - for Optimum-Neuron and NeuronX-Distributed test cases
- **SSH access**: GitHub Actions OIDC -> AWS IAM role -> SSH key to SLURM host
- **Shared filesystem**: FSX at `/fsx/agents/pr-reviews/`
- **Container runtime**: Enroot for container-based tests

### Multi-Stage Pipeline Considerations
Some test cases (DeepSpeed, Optimum-Neuron) have multi-stage pipelines where each stage depends on the previous one. These workflows need:
- Sequential sbatch job submission with dependency tracking (`--dependency=afterok:$JOB_ID`)
- Intermediate artifact validation between stages
- Longer timeouts to accommodate full pipeline execution

## Priority

**Phase 1** is the highest priority because:
- DDP is the foundational pattern all other frameworks build on
- DeepSpeed Llama fine-tuning is a production workflow
- TorchTitan represents the cutting-edge PyTorch distributed training direction
- Megatron-LM CI currently only builds containers but never runs training

**Phase 5 (Static Validation)** should be implemented in parallel with Phase 1 as it requires no GPU infrastructure and catches errors early in the PR review cycle.
