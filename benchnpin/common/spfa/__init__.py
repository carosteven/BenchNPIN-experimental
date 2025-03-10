import importlib.util
import sys
import os

module_path = os.path.join(os.path.dirname(__file__), "spfa.cpython-310-x86_64-linux-gnu.so")
spec = importlib.util.spec_from_file_location("spfa", module_path)
spfa = importlib.util.module_from_spec(spec)
sys.modules["spfa"] = spfa
spec.loader.exec_module(spfa)