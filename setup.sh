#!/usr/bin/env bash
# Sets up the authors' code next to our scripts. Run once from the repo root:  bash setup.sh
set -e
if [ ! -d LTSF-Linear ]; then
  git clone https://github.com/cure-lab/LTSF-Linear.git LTSF-Linear
fi
# Fix 1 of 2: np.Inf was removed in NumPy 2.0
sed -i.bak 's/np\.Inf/np.inf/g' LTSF-Linear/utils/tools.py && rm -f LTSF-Linear/utils/tools.py.bak
echo "Patched np.Inf -> np.inf in LTSF-Linear/utils/tools.py"
echo "Fix 2 of 2 is a command-line flag: always pass --num_workers 0 (avoids a macOS multiprocessing crash)."
echo
echo "Next: put the authors' benchmark CSVs (ETTh1.csv, weather.csv, national_illness.csv, ...) in LTSF-Linear/dataset/"
echo "(download link is in the authors' README), then see replication/run_replication.sh"
