# Importing Packages
from embedding.vectorizer import remove_stopwords


def retrieve_top_result_by_keyword_overlap(query, documents):
	"""
	Retrieves the document that has the most overlapping words with the input query
	"""

	# Split the query into lowercase words and store them in a set
	clean_query_words = remove_stopwords(query)
	best_doc_id = None
	best_overlap_score = 0

	for doc_id, doc in documents.items():
		# Remove stopwords from doc content nd thr user query
		# Concat title and content to get efficient match
		# Compare the query words with the document's content words
		# Calculate number of overlapping words of query vs reference doc(s)

		clean_doc_words = remove_stopwords(" ".join(doc["content"]))
		clean_title_words = remove_stopwords(doc['title'])
		doc_master = clean_doc_words.union(clean_title_words)
		overlap_score = len(clean_query_words.intersection(doc_master))

		if overlap_score > best_overlap_score:
			best_overlap_score = overlap_score
			best_doc_id = doc_id

	# Return the best document, or None if nothing matched
	return documents.get(best_doc_id)


def retrieve_top_results_by_distance(query, collection, category=None, top_k=3, similarity_threshold=0.60):
	"""
		Retrieves the top_k chunks from Chroma 'collection' that are most relevant to the given query.
		Returns a list of retrieved chunks, each containing 'chunk' text, 'doc_id', 'distance' and 'similarity.
		Filters the chunk only if similarity is greater than the set threshold
		If 'category' filter is provided, will be used as a filter
	"""
	retrieved_chunks = []
	fallback = False

	if category[0]:
		# If user opted to query with category filter
		where_clause = {"category": {"$in": category}}

		# Perform the initial search with category filter (if provided)
		results = collection.query(
			query_texts=[query],
			n_results=top_k,
			where=where_clause
		)

		# Fallback in case no results are found with category filter - no filter query
		if not results['documents'] or not results['documents'][0]:
			results = collection.query(
				query_texts=[query],
				n_results=top_k,
			)

			# Return empty list if no results are found even without filter
			if not results['documents'] or not results['documents'][0]:
				return retrieved_chunks, fallback
			else:
				# Chunks retrieved by fallback mechanism - no filter query
				fallback = True

	else:
		# If user opted to query without category filter
		results = collection.query(
				query_texts=[query],
				n_results=top_k,
			)

		# Return empty list if no results are found
		if not results['documents'] or not results['documents'][0]:
			return retrieved_chunks, fallback

	# Gather each retrieved chunk if results are found in the query, along with its distance score
	for i in range(len(results['documents'][0])):
		distance = results['distances'][0][i]
		similarity = 1/(1+distance)

		if similarity < similarity_threshold:
			continue 	# Not matching enough with the query

		retrieved_chunks.append({
			"content": results['documents'][0][i],
			"doc_chunk_id": results['ids'][0][i],
			"category": results["metadatas"][0][i]["category"],
			"distance": distance,
			"similarity": similarity
		})

	return retrieved_chunks, fallback


def perform_hybrid_retrieval(query, chunks, bm25, collection, top_k=3, alpha=0.5):
	"""
	Merge BM25 and embedding-based results.

	1. BM25 scores are computed for all the chunks
	2. Similarity scores for top 30 (3*10) embedding chunks are calculated
	3. Normalize both BM25 and similarity to [0,1]
	4. Combine with weighting:
	5. Sort by final score in descending order.

	'alpha' controls how much weight lexical vs. embedding-based similarity gets.
	1.0 => only BM25 ; 0 => only embeddings ; 0.5 => balanced
	"""
	retrieved_chunks = []

	# Map chunk string id [chroma collection] to numerical index [unique enum idx on master chunk]
	id_to_index = {
		f"chunk_{chunk['doc_id']}_{chunk['chunk_id']}": idx
		for idx, chunk in enumerate(chunks)
	}

	# BM25 scores are computed for all the chunks
	tokenized_query = query.lower().split()
	bm25_scores = bm25.get_scores(tokenized_query)
	# For normalizing BM25 scores
	bm25_min, bm25_max = (min(bm25_scores), max(bm25_scores)) if bm25_scores.size > 0 else (0, 1)

	# Similarity scores for top 30 (3*10) embedding chunks are calculated
	embed_results = collection.query(query_texts=[query], n_results=min(top_k*10, len(chunks)))
	embed_scores_dict = {}
	for i in range(len(embed_results['documents'][0])):
		chunk_id = embed_results['ids'][0][i]
		distance = embed_results['distances'][0][i]
		similarity = 1 / (1 + distance)

		# Map chunk string id [chroma collection] to numerical index [unique enum idx on master chunk]
		chunk_num_id = id_to_index.get(chunk_id)
		if chunk_num_id is not None:
			embed_scores_dict[chunk_num_id] = similarity

	# For normalizing embedding similarity scores
	sims = list(embed_scores_dict.values())
	sim_min, sim_max = min(sims), max(sims)

	merged = []
	for i, chunk in enumerate(chunks):
		bm25_raw = bm25_scores[i]
		# BM25 Score normalized for the chunk
		if bm25_max != bm25_min:
			bm25_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
		else:
			bm25_norm = 0.0

		# Embedding Similarity score normalized for the chunk
		embed_sim = embed_scores_dict.get(i, 0.0)
		embed_sim = (embed_sim - sim_min) / (sim_max - sim_min)

		# Final merged score for the chunk
		final_score = alpha * bm25_norm + (1 - alpha) * embed_sim
		merged.append((i, final_score))

	# Filter out low-score chunks (less than 0.2)
	merged = [item for item in merged if item[1] >= 0.2]

	# Sorting and getting top 3 results
	merged.sort(key=lambda x: x[1], reverse=True)
	top_results = merged[:top_k]

	for i, score in top_results:
		chunk_data = chunks[i]
		retrieved_chunks.append({
			"content": chunk_data['content'],
			"doc_id": chunk_data['doc_id'],
			"chunk_id": chunk_data['chunk_id'],
			"category": chunk_data.get("category", "unknown"),
			"embedding_similarity_score": embed_scores_dict.get(i, 0.0),
			"bm25_score_normalized": bm25_scores[i],
			"final_score": score
		})

	return retrieved_chunks
