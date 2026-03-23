# Importing Packages
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


# Download Stopwords only if it is not found
try:
	nltk.data.find('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

# Defining Stopwords
STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(text):
	"""
	Removes stopwords from the text and return set of unique words
	"""
	words = re.findall(r"\b\w+\b", text.lower())
	return {w for w in words if w not in STOPWORDS}


def preprocess_string(text):
	"""
	Returns cleaned list of words => remove punctuations around the text
	"""
	words = text.split()
	cleaned_words = [word.lower().strip(".,!?") for word in words]
	return cleaned_words


def build_vocab(docs):
	"""
	Builds vocabulary by assigning unique index to each unigram and bigram
	Creates master vocab by indexing unique words across all the provided docs
	"""
	unique_words = set()  # To assure no duplicate words are added

	for doc in docs:
		words = preprocess_string(doc)
		unique_words.update(set(words))  # Gathering unique unigrams across all the provided docs

		for i in range(len(words)):
			if i < len(words) - 1:
				bigram = words[i] + " " + words[i + 1]
				unique_words.add(bigram)  # Gathering unique bigrams across all the provided docs

	vocab = {word: idx for idx, word in enumerate(sorted(unique_words))}  # Indexing each word
	return vocab


def create_bow_vector(doc, vocab):
	"""
	Creates corresponding vector representation for given input text
	Applies BagOfWords [Frequency Count] on the text using vocabulary for vectorization
	Takes both unigram and bigrams into account
	"""

	# Create a zero vector with the same length as the vocabulary
	vector = np.zeros(len(vocab), dtype=int)
	# Preprocess the input string
	words = preprocess_string(text=doc)

	for i in range(len(words)):
		if words[i] in vocab:
			# Increment the vector slot corresponding to the unigrams and bigraqms, if found in the vocab
			# vocab format => {'word': idx}
			vector[vocab[words[i]]] += 1
			if i < len(words) - 1:
				bigram = words[i] + " " + words[i + 1]
				if bigram in vocab:
					vector[vocab[bigram]] += 1

	return vector


def build_bm25_index(chunks):
	"""
	Build a BM25Okapi index from the chunk texts for lexical-based retrieval.
	"""
	corpus = [c["content"].lower().split() for c in chunks]
	return BM25Okapi(corpus)
