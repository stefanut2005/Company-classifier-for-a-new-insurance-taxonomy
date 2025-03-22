from transformers import pipeline
import pandas as pd

taxonomy_data = pd.read_csv('insurance_taxonomy - insurance_taxonomy.csv')
df = pd.read_csv("ml_insurance_challenge.csv")
all_labels = taxonomy_data['label'].tolist()

classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=0)  # For more accurate but slower classification
# classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", device=0)  # For faster, but less accurate 

batch_size = 8
df["insurance_label"] = ""

for i in range(0, len(df), batch_size):
    batch_texts = [
        (row["description"] + " " + row["niche"]) if pd.isna(row["business_tags"]) or row["business_tags"].strip() == "" 
        else (row["business_tags"] + " " + row["niche"])
        for _, row in df.iloc[i:i+batch_size].iterrows()
    ]

    results = classifier(batch_texts, candidate_labels=all_labels, multi_label=True, batch_size=batch_size)

    if isinstance(results, dict):
        results = [results]

    print(f"First {i + batch_size} companies done.....")
    batch_top_labels = [", ".join(res['labels'][:10]) for res in results]

    df.loc[i:i+len(batch_top_labels)-1, "insurance_label"] = batch_top_labels
    df.to_csv("classified_companies.csv", index=False)

print("Classification saved in classified_companies.csv")
