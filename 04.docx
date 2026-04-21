import spacy
import matplotlib.pyplot as plt
from collections import Counter

nlp = spacy.load("en_core_web_sm")

text = "Google was founded in the United States and has offices in India and London."

doc = nlp(text)

entities = [ent.label_ for ent in doc.ents]
counter = Counter(entities)

labels = list(counter.keys())
values = list(counter.values())

# Plot
plt.figure()
plt.bar(labels, values)
plt.title("Entity Type Distribution")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.show()

for ent in doc.ents:
    print(ent.text, ent.label_)


# Print entities
for ent in doc.ents:
    print(ent.text, ent.label_)

# Show tokens + entity tags
for token in doc:
    print(token.text, token.ent_type_)

# Visualize (works in Jupyter)
from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)

# Additional stats
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities:", entities)
