# build_gloss_dict.py
import csv


def build_gloss_dict(csv_files):
    """Build a gloss dictionary from multiple CSV files"""
    gloss_set = set()

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                gloss = row[2].strip()  # Gloss is in the 3rd column
                gloss_set.add(gloss)

    # Convert to sorted list and create dictionary
    gloss_list = sorted(list(gloss_set))
    gloss_dict = {gloss: idx for idx, gloss in enumerate(gloss_list)}

    return gloss_dict


# Usage
csv_files = ["aslcitizen_training_set.csv"]
gloss_dict = build_gloss_dict(csv_files)

print("Gloss Dictionary:")
for gloss, idx in gloss_dict.items():
    print(f"'{gloss}': {idx},")

print(f"\nTotal unique signs: {len(gloss_dict)}")
