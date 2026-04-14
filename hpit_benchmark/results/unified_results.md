# HPIT PDE Benchmark — Unified Results

L2 relative error: `||u_pred - u_true||_2 / ||u_true||_2`
Lower is better.  `nan` = no checkpoint (random weights).
`see_paper` = value reported in the source paper.
`pending` = benchmark run not yet complete.
`-` = not tested.

| Model | Burgers1D | Burgers2D | HeatComplexGeometry | KuramotoSivashinsky | NavierStokes2D |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepONet | 1.0084 | 1.0025 | 1.0089 | 1.0035 | 1.0091 |
| FNO | 1.0021 | 1.0044 | 1.0049 | 0.9931 | 1.0107 |
| GNOT | 1.0110 | 0.9995 | 1.0009 | 1.0149 | 1.0102 |
| HPIT | 0.9905 | 1.0227 | 1.0000 | 1.0016 | 1.0623 |
| Mamba-NO | 1.0413 | 1.0755 | 1.2633 | 1.1248 | 1.0508 |
| PINO | nan | 1.0052 | 1.0020 | 1.0332 | 1.0104 |

## Sources
- PINNacle: Hao et al., NeurIPS 2024 — https://arxiv.org/abs/2306.08827
- PDEBench: Takamoto et al., NeurIPS 2022 — https://arxiv.org/abs/2210.07182
- FNO: Li et al., ICLR 2021 — https://arxiv.org/abs/2010.08895
- GNOT: Hao et al., ICML 2023 — https://arxiv.org/abs/2302.14376
- DeepONet: Lu et al., Nature Machine Intelligence, 2021 — https://doi.org/10.1038/s42256-021-00302-5
- PINO: Li et al., 2021 — https://arxiv.org/abs/2111.08907
- Mamba-NO: Cheng et al., 2024 — https://arxiv.org/abs/2410.02113 (no public code for our PDEs)