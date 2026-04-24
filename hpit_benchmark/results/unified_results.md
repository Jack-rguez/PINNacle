# HPIT PDE Benchmark — Unified Results

L2 relative error: `||u_pred - u_true||_2 / ||u_true||_2`
Lower is better.  `nan` = no checkpoint (random weights).
`see_paper` = value reported in the source paper.
`pending` = benchmark run not yet complete.
`-` = not tested.

| Model | Burgers1D | Burgers2D | HeatComplexGeometry | KuramotoSivashinsky | NavierStokes2D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 04.23-07.05.41-vanilla_pinn | 0.0201 | 0.5178 | - | - | - |
| 04.23-10.01.15-pinn_lra | 0.0350 | 0.4795 | - | - | - |
| 04.23-21.37.23-rar | 0.0433 | 0.5082 | - | - | - |
| DeepONet | 0.0708 | 0.2708 | 0.4511 | 0.9784 | 0.7508 |
| FNO | 0.0040 | 0.1854 | 0.0405 | 0.1679 | 0.8210 |
| GNOT | 0.0437 | 0.2695 | 0.0558 | 1.0735 | 0.6361 |
| HPIT | 2.1176 | 0.3195 | 0.9256 | 0.8854 | 0.9201 |
| HPIT (no physics) | 0.0118 | 0.2496 | 0.0108 | 0.0502 | 0.6943 |
| Mamba-NO | 0.0050 | 0.1572 | 0.0209 | 0.0203 | 0.9256 |
| PINO | 0.2023 | 0.2625 | 0.0637 | 0.9916 | 0.9034 |

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