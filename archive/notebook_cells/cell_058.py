# ============================================
# Cell 56. Inspect compact checkpoint contents
# and currently available VeloCycle objects
# ============================================

print("Compact checkpoint keys:")
print(vc.files)

print("\nRelevant variables currently in memory:")

names_to_check = [
    "cycle_pyro",
    "cycle_pyro_ref",
    "velocity_fit",
    "phase_fit",
    "data_to_fit",
]

for name in names_to_check:
    if name in globals():
        obj = globals()[name]
        print(
            f"{name:20s}",
            type(obj),
            getattr(obj, "shape", "")
        )
    else:
        print(
            f"{name:20s}",
            "NOT IN MEMORY"
        )

# Inspect any cycle-like arrays in compact checkpoint
print("\nCheckpoint arrays:")

for key in vc.files:
    arr = vc[key]
    print(
        f"{key:20s}",
        "shape =", arr.shape,
        "dtype =", arr.dtype
    )