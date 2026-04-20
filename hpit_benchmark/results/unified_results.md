# HPIT PDE Benchmark — Unified Results

L2 relative error: `||u_pred - u_true||_2 / ||u_true||_2`
Lower is better.  `nan` = no checkpoint (random weights).
`see_paper` = value reported in the source paper.
`pending` = benchmark run not yet complete.
`-` = not tested.

| Model | Burgers1D | Burgers2D | HeatComplexGeometry | KuramotoSivashinsky | NavierStokes2D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 04.18-07.13.52-vanilla_pinn | 0.0203 | 0.5221 | - | - | - |
| 04.18-10.10.05-pinn_lra | 0.0840 | 0.4465 | - | - | - |
| 04.18-21.38.01-rar | 0.0393 | 0.5216 | - | - | - |
| DeepONet | 0.0661 | 0.2718 | 0.3725 | 0.9021 | 0.8658 |
| FNO | 0.0034 | 0.1980 | 0.0379 | 0.1705 | 0.8208 |
| GNOT | 0.0403 | 0.2693 | 0.1166 | 1.1228 | 0.6799 |
| PINO | 0.2022 | 0.2572 | 0.0659 | 1.0000 | 0.9033 |

## Sources
- PINNacle: Hao et al., NeurIPS 2024 — https://arxiv.org/abs/2306.08827
- PDEBench: Takamoto et al., NeurIPS 2022 — https://arxiv.org/abs/2210.07182
- FNO: Li et al., ICLR 2021 — https://arxiv.org/abs/2010.08895
- GNOT: Hao et al., ICML 2023 — https://arxiv.org/abs/2302.14376
- DeepONet: Lu et al., Nature Machine Intelligence, 2021 — https://doi.org/10.1038/s42256-021-00302-5
- PINO: Li et al., 2021 — https://arxiv.org/abs/2111.08907
- Mamba-NO: Gu & Dao, ICLR 2024 (Mamba SSM) — https://arxiv.org/abs/2312.00752
  Operator wrapper inspired by LaMO (M3RG-IITD) — https://github.com/M3RG-IITD/LaMO
  Pure-PyTorch selective scan; upgrades to mamba_ssm fast path on GPU.