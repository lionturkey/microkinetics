import filecmp
import os

def test_basic():
    os.system("python pid_mir.py")
    assert filecmp.cmp("test_results.png", "pid_results.png")
