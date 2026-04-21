import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Sample text
text = """
Natural Language Processing is a field of AI that focuses on interaction between humans and computers.
It involves tasks like translation, summarization, and sentiment analysis.
Machine learning models are widely used in NLP applications.
These models improve automatically through experience.
NLP is widely used in chatbots, search engines, and voice assistants.
"""

# Step 1: Sentence Tokenization
sentences = sent_tokenize(text)

# Step 2: Word Tokenization
tokenized = [word_tokenize(sent.lower()) for sent in sentences]

# Step 3: Create vocabulary
all_words = list(set([word for sent in tokenized for word in sent]))

# Step 4: Create sentence vectors
vectors = []
for sent in tokenized:
    vec = [sent.count(word) for word in all_words]
    vectors.append(vec)

# Step 5: Similarity Matrix
sim_matrix = cosine_similarity(vectors)

# Step 6: Graph + TextRank
nx_graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(nx_graph)

# Step 7: Rank sentences
ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

# Step 8: Select top 2 sentences
summary = " ".join([ranked[i][1] for i in range(2)])

print("Extractive Summary:\n", summary)




import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Sample text
text = """
Natural Language Processing is a field of AI that focuses on interaction between humans and computers.
It involves tasks like translation, summarization, and sentiment analysis.
Machine learning models are widely used in NLP applications.
These models improve automatically through experience.
NLP is widely used in chatbots, search engines, and voice assistants.
"""

# Step 1: Sentence Tokenization
sentences = sent_tokenize(text)

# Step 2: Word Tokenization
tokenized = [word_tokenize(sent.lower()) for sent in sentences]

# Step 3: Create vocabulary
all_words = list(set([word for sent in tokenized for word in sent]))

# Step 4: Create sentence vectors
vectors = []
for sent in tokenized:
    vec = [sent.count(word) for word in all_words]
    vectors.append(vec)

# Step 5: Similarity Matrix
sim_matrix = cosine_similarity(vectors)

# Step 6: Graph + TextRank
nx_graph = nx.from_numpy_array(sim_matrix)
scores = nx.pagerank(nx_graph)

# Step 7: Rank sentences
ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

# Step 8: Select top 2 sentences
summary = " ".join([ranked[i][1] for i in range(2)])

print("Extractive Summary:\n", summary)
