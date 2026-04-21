from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Load model (uses default DistilBERT fine-tuned model)
sentiment_pipeline = pipeline("sentiment-analysis")

# Larger and more realistic dataset
texts = [
    "I absolutely love this product, it's fantastic!",
    "Worst experience ever, totally disappointed.",
    "The service was okay, nothing special.",
    "Amazing quality and great support!",
    "I hate how bad this turned out.",
    "Not good, could be better.",
    "Pretty decent overall, I liked it.",
    "Terrible, I will never buy this again.",
    "Excellent work, very impressed!",
    "It's fine, not too bad."
]

# Run sentiment analysis
results = sentiment_pipeline(texts)

# Convert to DataFrame for better analysis
df = pd.DataFrame(results)
df['text'] = texts

# Count sentiments
sentiment_counts = df['label'].value_counts()

# ---- BAR PLOT ----
plt.figure()
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Count Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()

# ---- CONFIDENCE SCORE HISTOGRAM ----
plt.figure()
plt.hist(df['score'])
plt.title("Confidence Score Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.show()

# ---- PRINT RESULTS ----
print("\nDetailed Results:\n")
for i, row in df.iterrows():
    print(f"Text: {row['text']}")
    print(f"Sentiment: {row['label']}, Confidence: {row['score']:.4f}\n")

# ---- BASIC METRICS ----
pos = (df['label'] == 'POSITIVE').sum()
neg = (df['label'] == 'NEGATIVE').sum()

print("Total Samples:", len(df))
print("Positive:", pos)
print("Negative:", neg)
