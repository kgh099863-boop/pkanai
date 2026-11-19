from pathlib import Path
import runpy

if __name__ == "__main__":
    test_file = Path(__file__).resolve().parent / "tests" / "test_suite.py"
    runpy.run_path(str(test_file), run_name="__main__")
