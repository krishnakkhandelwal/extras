import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('gutenberg')

text = gutenberg.raw('austen-emma.txt')
words = word_tokenize(text)

# Frequency distribution
fdist = nltk.FreqDist(words)

# Top 20 words
top_words = fdist.most_common(20)
words_list = [w[0] for w in top_words]
counts = [w[1] for w in top_words]

# Plot
plt.figure()
plt.bar(words_list, counts)
plt.xticks(rotation=90)
plt.title("Top 20 Word Frequency")
plt.xlabel("Words")
plt.ylabel("Count")
plt.show()

print("Total Words:", len(words))
