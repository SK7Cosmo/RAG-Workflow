# Importing packages
import json

from rag.llm import get_llm_response
from ingestion.chunking import load_and_chunk_dataset
from ingestion.chroma_store import build_chroma_collection
from rag.retrieval import retrieve_top_result_by_keyword_overlap, retrieve_top_results_by_distance
from rag.llm import generate_naive_response, generate_rag_response

import warnings
warnings.filterwarnings('ignore')

from transformers.utils import logging
logging.set_verbosity_error()

# Define Knowledge Base for RAG retrieval by keyword overlap
with open("data/sk7_knowledge_base1.json", "r") as file_obj:
	KNOWLEDGE_BASE = json.load(file_obj)

# Define Knowledge Base for RAG retrieval by distance
with open("data/sk7_knowledge_base3.json", "r") as file_obj:
	dataset = json.load(file_obj)


if __name__ == "__main__":
	# Used for fallback intimation [filter by category failed]
	additional_prompt = None

	master_chunks = load_and_chunk_dataset(data=dataset)
	print("\nLoaded the dataset and created", len(master_chunks), "chunk(s) from dataset.\n")

	collection = build_chroma_collection(chunks=master_chunks, collection_name="rag_collection")
	total_chunk_docs = collection.count()
	print("\nChromaDB collection created with", total_chunk_docs, "chunk document(s).")

	agent_choice = int(input("""\nChoose relevant option based on type of agent to be tested: 
			1. Basic Agent
			2. Custom RAG Agent - Keyword Overlap based [JSON Source]
			3. Custom RAG Agent - Distance based [ChromaDB Source]
			\nChoice: """))
	query = input("\nEnter the Prompt: ")

	if agent_choice == 1:
		# Sample Queries
		"""
		The capital of India is
		Currency followed in New York is
		What day comes after Saturday?
		"""

		print("\nNaive Agent's Response:\n\n", generate_naive_response(query))

	elif agent_choice == 2:
		# Sample Queries
		"""
		Give an overview on Agentic AI
		What is the workflow for an AI Agentic system
		Name the Primary Components of Agentic AI
		"""

		retrieved_doc = retrieve_top_result_by_keyword_overlap(query, KNOWLEDGE_BASE)
		if retrieved_doc:
			rag_content = retrieved_doc["content"]
		else:
			rag_content = None
		print("\nRAG Agent's Response [Keyword Overlap based]:\n\n", generate_rag_response(
																							query=query,
																							rag_content=rag_content,
																							additional_prompt=additional_prompt))

	elif agent_choice == 3:
		# Sample Queries
		"""
		What are some recent technological breakthroughs? ; Filter => Education
		"""

		filter_choice = input("\nDo you want to filter by category (y/n)?: ")
		if filter_choice.lower() == 'y':
			category = input("\nEnter the Category filter: ").lower()
		elif filter_choice.lower() == 'n':
			category = None
		else:
			print("\nInvalid choice")
			quit()

		rag_content = []

		# retrieves top 3 chunks tha match the query (and optional category filter0
		retrieved_chunks, fallback = retrieve_top_results_by_distance(
																	query=query,
																	collection=collection,
																	category=[category],
																	top_k=3)

		for chunk in retrieved_chunks:
			rag_content.append(chunk['content'])

		if fallback:
			additional_prompt = """Mention that response could not be filtered by provided category
								Hence, used only query to generate response - in new line\n"""

		print("\nRAG Agent's Response [Distance based]:\n\n", generate_rag_response(
																					query=query,
																					rag_content=rag_content,
																					additional_prompt=additional_prompt))

	else:
		print("\nInvalid choice")
