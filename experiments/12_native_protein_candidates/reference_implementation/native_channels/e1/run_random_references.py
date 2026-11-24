"""Nine predeclared complete official-architecture initialization references."""
import sys
from pathlib import Path
import scan

for seed in range(88001,88010):
    result_path=scan.ROOT/f'random_{seed}'/'results.json'
    if result_path.exists():
        print('ALREADY_COMPLETED',seed,flush=True)
        continue
    print('START_RANDOM',seed,flush=True)
    sys.argv=['scan.py','--random-seed',str(seed)]
    scan.main()
print('ALL_NINE_RANDOM_REFERENCES_COMPLETED',flush=True)
