# Importing Packages
from embedding.bow_vectorizer import remove_stopwords


def retrieve_top_result_by_keyword_overlap(query, documents):
	"""
	Retrieves the document that closely matches with the input query
	Naive way of keyword match technique is used to build a less complex pipeline
	Only the best matching document is returned
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


def retrieve_top_results_by_distance(query, collection, categories=None, top_k=3, distance_threshold=1.0):
	"""
		Retrieves the top_k chunks from Chroma 'collection' that are most relevant to the given query.
		Returns a list of retrieved chunks, each containing 'chunk' text, 'doc_id', and 'distance'.
		Filters the chunk only if distance metric is lesser than the threshold
		If 'categories' are provided, will be used as a filter
	"""
	retrieved_chunks = []

	if categories[0]:
		# If user opted to query with category filter
		where_clause = {"category": {"$in": categories}}

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
			return retrieved_chunks

	else:
		# If user opted to query without category filter
		results = collection.query(
				query_texts=[query],
				n_results=top_k,
			)

		# Return empty list if no results are found
		if not results['documents'] or not results['documents'][0]:
			return retrieved_chunks

	# Gather each retrieved chunk if results are found in the query, along with its distance score
	for i in range(len(results['documents'][0])):
		if results['distances'][0][i] > distance_threshold:
			continue 	# Not matching enough with the query
		retrieved_chunks.append({
			"content": results['documents'][0][i],
			"doc_id": results['ids'][0][i],
			"category": results["metadatas"][0][i]["category"],
			"distance": results['distances'][0][i]
		})

	return retrieved_chunks
