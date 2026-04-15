 #!/usr/bin/env bash
  set -euo pipefail
  cd "$(dirname "$0")"   # always run from repo root

  python3 benchmark.py --name rar --method rar --iter 20000
  python3 hpit_benchmark/fno_benchmark.py
  python3 hpit_benchmark/gnot_benchmark.py
  python3 hpit_benchmark/deeponet_benchmark.py
  python3 hpit_benchmark/pino_benchmark.py
  python3 hpit_benchmark/mamba_no_benchmark.py
  python3 hpit_benchmark/hpit_pde_benchmark.py
  python3 hpit_benchmark/collect_results.py