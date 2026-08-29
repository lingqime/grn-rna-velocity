# ============================================
# Cell 8. Load official VeloCycle phase metadata
# ============================================

phase_df = pd.read_csv(phase_metadata)

print("phase_df shape:", phase_df.shape)
print("\nColumns:")
print(list(phase_df.columns))

print("\nFirst 5 rows:")
print(phase_df.head())

print("\nUnique cell IDs:")
print(phase_df["Unnamed: 0"].nunique())

print("\ncell_cycle_phi summary:")
print(phase_df["cell_cycle_phi"].describe())