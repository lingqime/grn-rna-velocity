# ============================================
# Cell 12. Check VeloCycle dependencies
# ============================================

import importlib.util
import sys

packages = [
    "torch",
    "pyro",
    "velocycle",
]

print("Python:", sys.version)
print()

for pkg in packages:
    spec = importlib.util.find_spec(pkg)

    if spec is None:
        print(f"{pkg:10s}: NOT INSTALLED")
    else:
        module = __import__(pkg)

        version = getattr(module, "__version__", "version unavailable")

        print(f"{pkg:10s}: installed")
        print(f"             version = {version}")
        print(f"             path    = {spec.origin}")