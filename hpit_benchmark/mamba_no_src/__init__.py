# Source: Mamba-NO — Mamba Neural Operator
# Based on: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
#            https://arxiv.org/abs/2312.00752
# Architecture reference: state-spaces/mamba, https://github.com/state-spaces/mamba
# LaMO operator design: M3RG-IITD/LaMO, https://github.com/M3RG-IITD/LaMO
# PINNacle benchmark: Hao et al. NeurIPS 2024, https://arxiv.org/abs/2306.08827
#
# Implementation: pure-PyTorch selective scan (no compiled CUDA kernels).
# Runs on CPU for dry-runs; uses mamba_ssm fast path on GPU if installed.
