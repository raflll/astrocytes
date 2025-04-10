import matplotlib.pyplot as plt
import numpy as np

# spine density
# sham_data = [0.216666667,
#              0.45,
#              0.433333333,
#              0.583333333,
#              0.433333333,
#              ]
# dpi7_data = [0.1875,
#              0.166666667,
#              0.325,
#              0.283333333
#              ]

# mushroom density
sham_data = [0.183333333,
            0.45,
            0.191666667,
            0.375,
            0.316666667
             ]
dpi7_data = [0.1125,
            0.1,
            0.233333333,
            0.191666667,
             ]

# thin density
# sham_data = [0.033333333,
#             0.015,
#             0.216666667,
#             0.175,
#             0.1
#              ]
# dpi7_data = [0.0625,
#             0.066666667,
#             0.091666667,
#             0.091666667
#              ]

# Calculate statistics
sham_mean = np.mean(sham_data)
dpi7_mean = np.mean(dpi7_data)
sham_sem = np.std(sham_data, ddof=1) / np.sqrt(len(sham_data))
dpi7_sem = np.std(dpi7_data, ddof=1) / np.sqrt(len(dpi7_data))

# Create figure
fig, ax = plt.subplots(figsize=(3, 4))

# Set positions for the two groups
positions = [0.3, 0.8]
width = 0.2

# Add jitter to the data points to avoid overlap
jitter_amount = 0.08  # Controls the amount of horizontal jitter
np.random.seed(42)  # For reproducibility

# Create jittered x positions - UPDATED to match new positions
jittered_sham_x = np.ones(len(sham_data)) * positions[0] + np.random.uniform(-jitter_amount, jitter_amount, len(sham_data))
jittered_dpi7_x = np.ones(len(dpi7_data)) * positions[1] + np.random.uniform(-jitter_amount, jitter_amount, len(dpi7_data))

# Plot individual data points with jitter
ax.scatter(jittered_sham_x, sham_data, color='white', edgecolor='black', s=50, zorder=3)
ax.scatter(jittered_dpi7_x, dpi7_data, color='black', s=50, zorder=3)

# Plot mean lines
ax.plot([positions[0] - width/2, positions[0] + width/2], [sham_mean, sham_mean],
         color='blue', linewidth=2, zorder=2)
ax.plot([positions[1] - width/2, positions[1] + width/2], [dpi7_mean, dpi7_mean],
         color='blue', linewidth=2, zorder=2)

# Plot error bars
ax.errorbar(positions[0], sham_mean, yerr=sham_sem, color='blue', linewidth=2,
             capsize=4, capthick=2, zorder=1)
ax.errorbar(positions[1], dpi7_mean, yerr=dpi7_sem, color='blue', linewidth=2,
             capsize=4, capthick=2, zorder=1)

# Add significance star - UPDATED to match new positions
y_max = max(max(sham_data), max(dpi7_data))
y_pos = y_max + 0.1
ax.plot([positions[0], positions[1]], [y_pos, y_pos], color='black', linewidth=1)
ax.text((positions[0] + positions[1])/2, y_pos + 0.05, '*', ha='center', va='center', fontsize=14)

# Set y-axis breaks and limits
ax.set_ylim(0, 0.75)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Create a y-axis break to match the original plot
ax.set_yticks([0, 0.25, 0.50, 0.75])
ax.set_yticklabels(['0', '0.25', '0.50', '0.75'])

# Set x-axis parameters
ax.set_xticks(positions)
ax.set_xticklabels(['sham', 'CCI'], rotation=45)
ax.set_xlim(0, 1.1)

# Set axis labels
ax.set_ylabel('Mushroom density (spine/μm)')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()