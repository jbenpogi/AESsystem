import pandas as pd
import os

# Path to your file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
input_path = os.path.join(BASE_DIR, "data", "training_set_rel3.tsv")
output_path = os.path.join(BASE_DIR, "data", "asap_clean.csv")

# Load TSV
df = pd.read_csv(input_path, sep='\t', encoding='latin-1')

# Keep only important columns
df = df[['essay_set', 'essay', 'domain1_score']]

# Max scores per essay_set
max_scores = {
    1: 12,
    2: 6,
    3: 3,
    4: 3,
    5: 4,
    6: 4,
    7: 30,
    8: 60
}

# Normalize scores to 0–100
df['normalized_score'] = df.apply(
    lambda row: (row['domain1_score'] / max_scores[row['essay_set']]) * 100,
    axis=1
)

# Save as CSV
df.to_csv(output_path, index=False)

print("✅ Dataset prepared:", output_path)