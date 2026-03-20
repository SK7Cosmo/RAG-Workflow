# Importing Packages
import os
import configparser

from openai import OpenAI

# Reading the config file
creds_config = configparser.ConfigParser()
creds_config.read('config.ini')

# Setting up the OpenRouter Key
keys = dict(creds_config.items('keys'))
openrouter_api_key = keys['openrouter_api_key']
os.environ["OPENROUTER_API_KEY"] = openrouter_api_key


# Initialize OpenRouter client
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=os.getenv("OPENROUTER_API_KEY"), )


def get_llm_response(user_prompt):
	"""
	Sends a prompt to the OpenAI model through OpenRouter API and returns the response.
	"""

	system_prompt = """You are an helpful AI assistant. You always answer to the user's queries."""

	try:
		response = client.chat.completions.create(
			model="openai/gpt-4o-mini",
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt}
			],
			temperature=0.0,  # Factual
			max_tokens=500,  # Maximum length of the generated response.
			top_p=1.0,
			frequency_penalty=0.0,  # To avoid phrase repetitions
			presence_penalty=0.0,  # To avoid topic repetition
		)
		# Extract and return the assistant's message from the response
		return response.choices[0].message.content

	except Exception as e:
		return f"An error occurred: {e}"


def generate_naive_response(query):
	"""
	Using LLM's pretrained knowledge base to respond to the query
	"""
	prompt = f"Answer directly the following query: {query}"
	return get_llm_response(prompt)


def generate_rag_response(query, rag_content, additional_prompt=None):
	"""
	Using the custom knowledge base to enrich the user prompt
	and customize the LLM's response

	If no info relevant to the query found in the knowledge base,
	avoids hallucination and politely refuses to answer

	If fallback is true, user intimated that response could not be filtered by provided category
	Hence, used only query to generate response
	"""
	if rag_content:
		prompt = f"Question: {query}\nAnswer using only the following context:\n"
		for fact in rag_content:
			prompt += f"- {fact}\n"

		if additional_prompt:
			prompt += additional_prompt

		prompt += "List the lines you used as evidence with 'Cited lines:' in new line.\n"
		prompt += "Specify that you have made use of preconfigured Knowledge Base in new line"
		prompt += "\nAnswer: "

	else:
		prompt = f"""
		Question: {query}\n
		No relevant information was retrieved for the above question.
		Politely refuse to answer. 
		"""

	return get_llm_response(prompt)
