"""Simple smoke test used by CI to ensure dependencies import correctly."""
import sys
import pandas as pd
import numpy as np


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"pandas: {pd.__version__}")
    print(f"numpy: {np.__version__}")


if __name__ == '__main__':
    main()
