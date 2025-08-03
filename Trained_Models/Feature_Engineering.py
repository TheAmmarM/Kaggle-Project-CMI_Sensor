# Add these features BEFORE training
# Eyebrow vs Eyelash differentiation
df['eyebrow_heat_dominance'] = (df['thm_1_max'] - df['thm_2_max']) / (df['thm_1_max'] + 1e-6)
df['eyelash_vertical_dispersion'] = df.filter(regex='tof_1_v[4-7]').std(axis=1)

# Neck pinch vs scratch
df['neck_pinch_stability'] = df['tof_5_v0_mean'] - df['tof_5_v63_mean']  # Corner difference
df['scratch_motion_profile'] = np.sqrt(df['acc_z_std']**2 + df['rot_x_std']**2)

# Temporal dynamics
df['gesture_intensity'] = df['acc_z_max'] * df['sequence_counter_max']