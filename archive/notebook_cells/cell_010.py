# ============================================
# Cell 11. Locate official 986-condition
# VeloCycle result pickle
# ============================================

pickle_dir = base_dir / "pickle_result_outputs"

pickle_files = sorted(
    pickle_dir.glob("*986conditions*.pkl.gz")
)

print("986-condition pickle files found:", len(pickle_files))

for p in pickle_files:
    size_gb = p.stat().st_size / 1024**3
    print(f"\n{p.name}")
    print(f"  compressed size: {size_gb:.2f} GB")

print(
    "\nAvailable RAM (GB):",
    round(psutil.virtual_memory().available / 1024**3, 2)
)

print(
    "Free disk (GB):",
    round(psutil.disk_usage(base_dir).free / 1024**3, 2)
)