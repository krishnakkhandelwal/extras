from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "Machine learning is fun and machine learning is powerful"

# Tokenization
tokens = word_tokenize(text)

# Generate bigrams
bigrams = list(ngrams(tokens, 2))
print("Bigrams:", bigrams)

# Generate trigrams
trigrams = list(ngrams(tokens, 3))
print("Trigrams:", trigrams)

# Bag of Words model
corpus = [
    "machine learning is fun",
    "learning is powerful",
    "machine is useful"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())

plt.figure()
plt.bar(features, counts)
plt.xticks(rotation=45)
plt.title("Bag of Words Frequency")
plt.ylabel("Count")
plt.show()

print("BoW Matrix:\n", X.toarray())
