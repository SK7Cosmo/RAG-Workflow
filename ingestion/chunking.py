# Importing Packages
import re
import json


def perform_chunk(text, chunk_size=30):
	"""
	Splits the given text into chunks of size 'chunk_size' [word count], preserving sentence boundaries.
	Returns a list of chunk strings.
	"""

	# Split text into semantic sentences
	sentences = re.split(r'[.!?]+', text)

	chunks = []
	current_chunk = ""

	for sentence in sentences:
		if not sentence:
			continue

		# If adding sentence to the chunk exceeds chunk size, save existing chunk without adding the ROI sentence
		if len((current_chunk + " " + sentence).split()) > chunk_size:
			chunks.append(current_chunk.strip())
			current_chunk = sentence  # current sentence marks the beginning of new chunk
		else:
			current_chunk += " " + sentence  # Keep adding the sentence to the chunk

	# Add last chunk to the chunk list
	if current_chunk:
		chunks.append(current_chunk.strip())

	return chunks


def load_and_chunk_dataset(data, chunk_size=30):
	"""
	Splits each document of the master dataset into smaller chunks.
	Metadata such as 'doc_id', 'chunk_id' is included with each chunk for easy retrieval.
	"""
	all_chunks = []
	for doc in data:
		doc_id = doc["id"]
		doc_category = doc["category"]
		doc_text = doc["content"]
		doc_chunks = perform_chunk(doc_text, chunk_size)
		for chunk_id, chunk_str in enumerate(doc_chunks):
			all_chunks.append({
				"doc_id": doc_id,
				"category": doc_category,
				"chunk_id": chunk_id,
				"content": chunk_str
			})
	return all_chunks


if __name__ == "__main__":
	# Define Knowledge Base for document chunking experiment
	with open("data/sk7_knowledge_base3.json", "r") as file_obj:
		dataset = json.load(file_obj)

	master_chunks = load_and_chunk_dataset(data=dataset)
	print("\nLoaded the dataset and created ", len(master_chunks), " chunks from dataset.\n")