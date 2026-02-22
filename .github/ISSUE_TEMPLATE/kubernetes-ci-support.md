---
name: Add Kubernetes CI Support
about: Track the work needed to add CI coverage for Kubernetes-based test cases
title: "Add Kubernetes CI support for uncovered test cases and manifests"
labels: enhancement, ci/cd, kubernetes
---

## Summary

The current CI pipeline has limited Kubernetes/EKS coverage. Only **PyTorch FSDP on EKS** (`fsdp-eks-regression.yml`) has a dedicated CI workflow. Multiple other test cases that include Kubernetes manifests and deployment configurations have **no CI validation at all**.

## Current State

### Existing CI Workflows
| Workflow | Platform | Scope |
|---|---|---|
| `fsdp-regression-test-container.yml` | SLURM | FSDP (container) - 5 models x 2 clusters |
| `fsdp-regression-test-venv.yml` | SLURM | FSDP (venv) - 5 models x 2 clusters |
| `fsdp-eks-regression.yml` | **EKS** | FSDP only - 5 models x 1 cluster (p5-eks) |
| `megatron-ci-slurm.yaml` | SLURM | Megatron-LM (build only) |
| `pr-review-and-slurm-test.yml` | Ubuntu + SLURM | Code review, security, version checks, SLURM tests |

### Kubernetes Content WITHOUT CI Coverage

The following test cases have Kubernetes manifests/configurations but **no CI pipeline**:

1. **Megatron NeMo on Kubernetes** - `3.test_cases/megatron/nemo/kubernetes/`
   - Has data processing pod templates, NeMo 2.0 with SkyPilot orchestration
   - No CI workflow

2. **Megatron-LM on Kubernetes** - `3.test_cases/megatron/megatron-lm/kubernetes/`
   - Has documentation and K8s deployment configs
   - Only SLURM CI exists (`megatron-ci-slurm.yaml`), no K8s CI

3. **PyTorch DDP on Kubernetes** - `3.test_cases/pytorch/ddp/kubernetes/`
   - Has K8s manifests for distributed data parallel training
   - No CI workflow

4. **PyTorch Distillation on Kubernetes** - `3.test_cases/pytorch/distillation/kubernetes/`
   - Has K8s deployment configurations
   - No CI workflow

5. **PyTorch NeuronX-Distributed on Kubernetes** - `3.test_cases/pytorch/neuronx-distributed/llama3/kubernetes/`
   - Has K8s manifests for Neuron-based training
   - No CI workflow

6. **PyTorch Optimum-Neuron on Kubernetes** - `3.test_cases/pytorch/optimum-neuron/llama3/kubernetes/`
   - Has K8s deployment for fine-tuning
   - No CI workflow

7. **VeRL (Reinforcement Learning) on Kubernetes** - `3.test_cases/pytorch/verl/kubernetes/`
   - Has RayCluster YAML, KubeRay setup, GRPO/DAPO training configs
   - No CI workflow

8. **VeRL on HyperPod EKS** - `3.test_cases/pytorch/verl/hyperpod-eks/`
   - Has HyperPod EKS-specific deployment configs
   - No CI workflow

9. **MosaicML Stable Diffusion on EKS** - `3.test_cases/pytorch/mosaicml-composer/stable-diffusion/multi-node/`
   - Has `3.stable-diffusion-eks.yaml-template` and `4.etcd.yaml`
   - No CI workflow

### Additional Gaps

- **No Kubernetes manifest validation** in `pr-review-and-slurm-test.yml` - K8s YAMLs are not validated for schema correctness during PR review
- **No Helm chart testing** for `4.validation_and_observability/3.efa-node-exporter/EKS/` Helm chart
- **No Terraform/CloudFormation validation** for infrastructure configs in `1.architectures/7.sagemaker-hyperpod-eks/`
- **Existing EKS workflow only tests on p5-eks** - no coverage for p5en-eks or p6-eks clusters (though env vars are pre-configured for them)

## Proposed Work Items

### Phase 1: Static Validation (PR-level checks)
Add Kubernetes-specific validation to the existing `pr-review-and-slurm-test.yml` workflow:

- [ ] **Kubernetes manifest validation** using `kubeconform` or `kubeval` for all `*.yaml` files under `kubernetes/` directories
- [ ] **Helm chart linting** using `helm lint` for the EFA node exporter chart
- [ ] **Template variable validation** - ensure all `envsubst` variables (`$IMAGE_URI`, `$NUM_NODES`, etc.) are properly referenced in templates
- [ ] **Terraform validation** using `terraform validate` for HyperPod EKS modules
- [ ] **CloudFormation linting** using `cfn-lint` for CFN templates

### Phase 2: Expand EKS Regression Testing
Extend the existing `fsdp-eks-regression.yml` pattern to other frameworks:

- [ ] **Megatron-LM EKS regression workflow** - similar to FSDP EKS workflow but targeting Megatron manifests
- [ ] **Megatron NeMo EKS regression workflow** - test NeMo 2.0 training on EKS
- [ ] **PyTorch DDP EKS regression workflow** - test DDP training on EKS
- [ ] **VeRL EKS regression workflow** - test KubeRay-based RL training on EKS

### Phase 3: Multi-Cluster and Neuron Support
- [ ] **Expand EKS cluster matrix** to include p5en-eks and p6-eks clusters in FSDP workflow
- [ ] **NeuronX-Distributed EKS workflow** - test on Neuron-based EKS clusters (trn1/inf2)
- [ ] **Optimum-Neuron EKS workflow** - test fine-tuning on Neuron EKS
- [ ] **HyperPod EKS integration tests** - validate HyperPod-specific configurations

### Phase 4: Observability and Infrastructure
- [ ] **Helm chart CI** - lint, template, and dry-run test the EFA node exporter Helm chart
- [ ] **Observability stack validation** - verify Prometheus/Grafana configs deploy correctly
- [ ] **Infrastructure-as-code CI** - Terraform plan and CloudFormation validate for architecture configs

## Implementation Notes

### Self-Hosted Runner Requirements
The existing EKS workflow uses `self-hosted` runners tagged with cluster names (e.g., `p5-eks`). New K8s workflows will need:
- Self-hosted runners deployed on each target EKS cluster
- `kubectl` access configured on runners
- ECR push permissions for container image builds
- Kubeflow Training Operator (for PyTorchJob resources)
- KubeRay Operator (for VeRL/RayCluster resources)

### Workflow Template Pattern
The existing `fsdp-eks-regression.yml` provides a good template pattern:
1. Checkout code
2. Build Docker image
3. Push to ECR
4. Apply K8s manifest with `envsubst` variable substitution
5. Monitor pod status and PyTorchJob conditions
6. Collect logs as artifacts
7. Cleanup resources

New workflows should follow this pattern for consistency.

### Path-Based Triggers
Each workflow should be scoped to its respective test case directory to avoid unnecessary CI runs:
```yaml
on:
  pull_request:
    paths:
      - '3.test_cases/<framework>/<test_case>/kubernetes/**'
      - '3.test_cases/<framework>/<test_case>/Dockerfile'
```

## Priority

**Phase 1 (Static Validation)** is the highest priority as it provides immediate value with minimal infrastructure requirements - it can run on standard `ubuntu-latest` runners and catches manifest errors before they reach EKS clusters.
