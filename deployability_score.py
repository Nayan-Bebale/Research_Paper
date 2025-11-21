import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data (from your provided table; includes Energy column)
data = {
    'model_name': ['fasterrcnn', 'fasterrcnn_mobilenet', 'fasterrcnn_v2', 'ssd', 'ssdlite',
                   'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                   'yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x'],
    'accuracy': [0.830, 0.863, 0.832, 0.833, 0.757, 0.559, 0.612, 0.593, 0.618, 0.640, 0.610, 0.632, 0.654, 0.609, 0.601],
    'latency_time': [3.526, 1.554, 3.162, 1.070, 1.295, 3.592, 3.360, 4.442, 5.917, 6.116, 4.185, 3.824, 4.327, 6.229, 6.833],
    'energy_j': [178.502, 82.551, 162.396, 69.256, 77.335, 259.305, 241.995, 323.971, 435.225, 452.492,
                 303.232, 278.607, 319.104, 465.671, 511.817]
}
df = pd.DataFrame(data)

# Deployability params (0-5)
w1, w2, w3 = 0.60, 0.25, 0.15           # weights (sum=1)
latency_ref = 1.0                       # seconds (reference)
energy_ref = 400.0                      # joules (reference)

# Normalization
lat_norm = np.minimum(1.0, df['latency_time'] / latency_ref)
eng_norm = np.minimum(1.0, df['energy_j'] / energy_ref)

# Compute deployability on 0-5 scale
df['deployability_0_5'] = 5 * (w1 * df['accuracy'] + w2 * (1 - lat_norm) + w3 * (1 - eng_norm))

# Round for presentation
df['deployability_0_5'] = df['deployability_0_5'].round(3)

# Short labels
labels = {
    'fasterrcnn': 'FR', 'fasterrcnn_mobilenet': 'FR-Mobile', 'fasterrcnn_v2': 'FR-v2',
    'ssd': 'SSD', 'ssdlite': 'SSD-Lite',
    'yolov8n': 'Y8n', 'yolov8s': 'Y8s', 'yolov8m': 'Y8m', 'yolov8l': 'Y8l', 'yolov8x': 'Y8x',
    'yolov10n': 'Y10n', 'yolov10s': 'Y10s', 'yolov10m': 'Y10m', 'yolov10l': 'Y10l', 'yolov10x': 'Y10x'
}
df['label'] = df['model_name'].map(labels)

# Save CSV for Appendix / reproducibility
df.to_csv('deployability_table_0_5.csv', index=False)

# Plot (0-5)
plt.figure(figsize=(12,8))
max_area = 2200
sizes = (df['deployability_0_5'] / 5.0) * max_area + 60   # scale bubble sizes
sc = plt.scatter(df['latency_time'], df['accuracy'], s=sizes,
                 c=df['deployability_0_5'], cmap='coolwarm', vmin=0, vmax=5,
                 alpha=0.85, edgecolors='k', linewidths=0.4)

# Annotate
for i, row in df.iterrows():
    plt.annotate(row['label'], (row['latency_time'], row['accuracy']), xytext=(4, 3),
                 textcoords='offset points', fontsize=9)

cbar = plt.colorbar(sc)
cbar.set_label('Deployability Score (0–5)')
cbar.set_ticks([0,1,2,3,4,5])

plt.xlabel('Latency (s)')
plt.ylabel('Accuracy (mAP)')
plt.title('Accuracy vs. Latency for Selected Models\n(Bubble size & color ∝ Deployability Score (0–5))')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('fig3_deploy_0_5.png', dpi=300, bbox_inches='tight')
plt.show()
