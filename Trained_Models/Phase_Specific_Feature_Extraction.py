# Assuming 'phase' exists and is relevant
phase_features = []

for seq_id, group in df.groupby('sequence_id'):
    phase_group = group.groupby('phase')
    phase_stats = {}
    for phase, sub in phase_group:
        for col in sensor_cols:
            if col not in sub.columns or not np.issubdtype(sub[col].dtype, np.number):
                continue
            phase_stats[f"{col}_phase{phase}_mean"] = sub[col].mean()
            phase_stats[f"{col}_phase{phase}_std"] = sub[col].std()
    phase_stats['sequence_id'] = seq_id
    phase_stats['gesture'] = group['gesture'].iloc[0]
    phase_features.append(phase_stats)

phase_df = pd.DataFrame(phase_features)
print(f"✅ Phase-specific Data Shape: {phase_df.shape}")
