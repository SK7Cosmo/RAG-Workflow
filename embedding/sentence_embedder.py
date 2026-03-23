# Importing Packages
import numpy as np

from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


def compute_cosine_similarity(vec_a, vec_b):
	"""
	Compute cosine similarity between two vectors:
	1 means identical direction, 0 means orthogonal.
	"""
	return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


def create_sentence_embeddings(sentences):
	"""
	Creates sentence embeddings for corresponding sentence
	Shape (sentence_count, 384)
	"""
	# Load a pre-trained embedding model from Sentence Transformers.
	model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
	embeddings = model.encode(sentences, show_progress_bar=False)  # Sentence => Embedding Vector
	return embeddings
