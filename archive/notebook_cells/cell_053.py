# ============================================
# Cell 52b. Inspect interval_rows structure
# ============================================

print("type(interval_rows):", type(interval_rows))
print("number of intervals:", len(interval_rows))

print("\nFirst interval:")
print(interval_rows[0])

print("\nType of first interval:")
print(type(interval_rows[0]))

# If dict-like, show keys
if isinstance(interval_rows[0], dict):
    print("\nKeys:")
    print(interval_rows[0].keys())

# If tuple/list-like, show length
if isinstance(interval_rows[0], (tuple, list)):
    print("\nLength of first interval:")
    print(len(interval_rows[0]))