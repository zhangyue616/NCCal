"""Run this experiment from an installed package or source checkout."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'src'))
from nccal.molecular.reproduce import main
if __name__=='__main__':main(experiment=Path(__file__).resolve().parent.name)
