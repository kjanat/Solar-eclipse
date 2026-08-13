import sys

from solar_eclipse_analysis.cli import main

if __name__ == "__main__":
    main([*sys.argv[1:], "analyze"])
