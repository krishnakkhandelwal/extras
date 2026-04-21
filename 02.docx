import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist
from collections import Counter

# required data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Sample corpus (better than 1 line text)
text = """
Natural Language Processing is a fascinating field of Artificial Intelligence.
It helps computers understand human language.
Machine learning and deep learning are widely used in NLP tasks.
"""

# ---------------- TOKENIZATION ----------------
sentences = sent_tokenize(text)
words = word_tokenize(text.lower())

print("Sentences:", sentences)
print("Words:", words)

# ---------------- STOPWORD REMOVAL ----------------
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

print("\nFiltered Words:", filtered_words)

# ---------------- POS TAGGING ----------------
pos_tags = pos_tag(filtered_words)
print("\nPOS Tags:", pos_tags)

# ---------------- STEMMING ----------------
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]

print("\nStemmed Words:", stemmed_words)

# ---------------- LEMMATIZATION ----------------
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

print("\nLemmatized Words:", lemmatized_words)

# ---------------- PLOT 1: Word Frequency ----------------
freq_dist = FreqDist(filtered_words)
common_words = freq_dist.most_common(10)

words_plot = [w[0] for w in common_words]
counts_plot = [w[1] for w in common_words]

plt.figure()
plt.bar(words_plot, counts_plot)
plt.title("Top Word Frequencies (After Preprocessing)")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# ---------------- PLOT 2: POS Tag Distribution ----------------
pos_counts = Counter(tag for word, tag in pos_tags)

labels = list(pos_counts.keys())
values = list(pos_counts.values())

plt.figure()
plt.bar(labels, values)
plt.title("POS Tag Distribution")
plt.xlabel("POS Tags")
plt.ylabel("Count")
plt.show()

# ---------------- PLOT 3: Before vs After ----------------
original_count = len(words)
filtered_count = len(filtered_words)

plt.figure()
plt.bar(["Original", "After Cleaning"], [original_count, filtered_count])
plt.title("Word Count Reduction After Preprocessing")
plt.ylabel("Number of Words")
plt.show()
