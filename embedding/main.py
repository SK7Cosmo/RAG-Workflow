import json
import numpy as np

from vectorizer import build_vocab, create_bow_vector
from ingestion.chunking import load_and_chunk_dataset
from ingestion.chroma_store import build_chroma_collection
from tsne_visualizer import compute_tsne_embeddings, plot_embeddings
from sentence_embedder import create_sentence_embeddings, compute_cosine_similarity


# Define Knowledge Base for text representation - query matching experiment
with open("../data/sk7_knowledge_base2.json", "r") as file_obj:
	KNOWLEDGE_BASE = json.load(file_obj)

# Define metadata for the t-SNE plot experiment
with open("../data/tsne_plot_metadata.json", "r") as file_obj:
	tsne_plot_metadata = json.load(file_obj)


def experiment_text_representations(choice, query):
	# Rank the knowledge base sentences based on relevance to the query using BagOfWords approach
	docs = KNOWLEDGE_BASE['sentences']

	if choice == 1:
		vocab = build_vocab(docs=docs)

		query_vec = create_bow_vector(query, vocab)
		bow_match_scores = []

		for idx, doc in enumerate(docs):
			doc_vec = create_bow_vector(doc, vocab)
			bow_match_score = np.dot(query_vec, doc_vec)
			bow_match_scores.append((idx, bow_match_score))

		# Sort documents so that higher lexical overlap is first
		bow_match_scores.sort(key=lambda x: x[1], reverse=True)
		return bow_match_scores

	# Use Sentence embedding and Cosine Similarity, find the most similar sentence to the query
	elif choice == 2:
		kbase_embeddings = create_sentence_embeddings(sentences=docs)
		query_embedding = create_sentence_embeddings(sentences=query)

		sentence_embedding_scores = []

		for idx in range(kbase_embeddings.shape[0]):
			sim_score = compute_cosine_similarity(kbase_embeddings[idx], query_embedding)
			sim_score = float(f"{sim_score * 100:.2f}")
			sentence_embedding_scores.append((idx, sim_score))

		# Sort documents so that higher cosine similarity is first
		sentence_embedding_scores.sort(key=lambda x: x[1], reverse=True)
		return sentence_embedding_scores

	else:
		print("\nInvalid choice")


# Construct t-SNE and generate a plot to visualize
def visualize_tsne():
	sentences = []
	categories = []

	# Retrieve category name and sentences for each topic
	for topic in tsne_plot_metadata.values():
		category = topic["category"]
		for sentence in topic['content']:
			sentences.append(sentence)
			categories.append(category) 	# To label each of the sentences separately

	# Compute t-DNE embeddings for each of the sentences
	reduced_embeddings = compute_tsne_embeddings(sentences=sentences)
	plot_embeddings(
		reduced_embeddings=reduced_embeddings,
		sentences=sentences,
		categories=categories)

	print("""\nPlot successfully generated and saved in the "plots" folder """)


if __name__ == "__main__":

	experiment_choice = int(input("""Choose relevant option based on type of experiment to be performed: 
	1. Ranking knowledge base sentences for given query using different Text Representations
	2. Visualize knowledge base sentences using t-SNE 
	\nChoice: """))

	if experiment_choice == 1:
		text_repr_choice = int(input("""Choose relevant option based on method of matching query vs knowledge base: 
		1. BagOfWords - Unigram and Bigram match
		2. Sentence Embedding - Cosine Similarity
		\nChoice: """))

		# Example Query
		# How does a system combine external data with language generation to improve responses?
		query = input("\nEnter the query to be matched with the preconfigured knowledge base: ")

		results = experiment_text_representations(choice=text_repr_choice, query=query)

		print("\nSearch Results:")
		for idx, score in results:
			print(f" Doc {idx} | Score: {score} | Text: {KNOWLEDGE_BASE['sentences'][idx]}")

	elif experiment_choice == 2:
		visualize_tsne()

	else:
		print("\nInvalid choice")
