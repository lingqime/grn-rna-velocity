# ============================================
# Cell 28. Understand official phase representation
# ============================================

from pathlib import Path

phases_source = (
    velocycle_legacy
    / "velocycle"
    / "phases.py"
)

print("--- Relevant Phases source ---")

source = phases_source.read_text()

# Print the Phases class definition
start = source.find("class Phases")
end = source.find("\nclass ", start + 1)

if end == -1:
    end = len(source)

print(source[start:end])