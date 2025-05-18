import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name.lower())
    return list(synonyms)

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random.shuffle(words)
    num_replaced = 0

    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            index = new_words.index(word)
            new_words[index] = synonym
            num_replaced += 1
        if num_replaced >= n:
            break

    return " ".join(new_words)
