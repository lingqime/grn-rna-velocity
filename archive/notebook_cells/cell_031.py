# ============================================
# Cell 32. Inspect AngularSpeed parameterization
# ============================================

from pathlib import Path

angular_source = (
    velocycle_legacy
    / "velocycle"
    / "angularspeed.py"
)

velocity_source = (
    velocycle_legacy
    / "velocycle"
    / "velocity_inference_model.py"
)

print("========== angularspeed.py ==========")

src = angular_source.read_text()

start = src.find("class AngularSpeed")
end = src.find("\nclass ", start + 1)

if end == -1:
    end = len(src)

print(src[start:end])

print("\n\n========== occurrences of 'speed_pyro' ==========")

vsrc = velocity_source.read_text()

for i, line in enumerate(vsrc.splitlines(), start=1):
    if (
        "speed_pyro" in line
        or "nu0" in line
        or "ν" in line
        or "omega" in line.lower()
    ):
        print(f"{i:4d}: {line}")