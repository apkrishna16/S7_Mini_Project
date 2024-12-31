# Import necessary libraries and modules
import os  
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate  
from sentence_transformers import SentenceTransformer
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import chromadb 
from chromadb.config import Settings
import textwrap 
import pickle


# File paths for storing retriever and chunked documents
BM25_RETRIEVER_FILE = "main/data/bm25_retriever.pkl"
CHUNKED_DOCS_FILE = "main/data/chunked_documents.pkl"
TXT_DIR = 'main/data/Judgement_txt'  # Directory containing judgment text files

COLLECTION_NAME = "legal_judgments"  # ChromaDB collection name
model_name = "llama3.2:3b"  # Evaluation model

def generate_query_variations(query, llm_model, n=3):
    """
    Generates alternative query variations for retrieving documents from a vector database.

    Parameters:
    - query: The original query string.
    - llm_model: The language model used for generating variations.
    - n: Number of variations to generate (default is 3).

    Output:
    - Returns a list containing the original query and its variations.
    """
    
    response_schemas = [
        ResponseSchema(name=f"variation_{i+1}", description=f"Variation {i+1} of the query")
        for i in range(n)
    ] 
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
 
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            f"You are an AI assistant. Your task is to generate {n} alternative versions "
            "of the following question to retrieve relevant documents from a vector database. "
            f"Format the output strictly as JSON with fields: {', '.join([f'variation_{i+1}' for i in range(n)])}."
            "You must always return valid JSON fenced by a markdown code block. Do not return any additional text."
            "\nOriginal question: {question}"
            "\n{format_instructions}"
        ),
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
 
    chain = prompt | llm_model 
    response = chain.invoke({"question": query})
    parsed_response = output_parser.parse(response)
 
    variations = [query]
    variations.extend([
        parsed_response[f"variation_{i+1}"]
        for i in range(n)
    ])
 
    return variations

def init_chromadb(collection_name):
    """
    Initializes ChromaDB with persistence settings.

    Parameters:
    - collection_name: Name of the collection to initialize in ChromaDB.

    Output:
    - Returns an initialized collection from ChromaDB.
    """
    
    client = chromadb.Client(Settings(is_persistent=True, persist_directory= 'main/data/chormadb'))
    collection = client.get_or_create_collection(collection_name)
    print(f"ChromaDB initialized and collection '{collection_name}' ready")
    return collection

def vector_search(query, collection, top_n=5):
    """
    Performs vector-based search to retrieve top `n` document chunks.

    Parameters:
    - query: The query string for which relevant documents are to be retrieved.
    - collection: The collection from which documents are to be searched.
    - top_n: Number of top results to retrieve (default is 5).

    Output:
    - Returns a list of dictionaries containing document details such as ID, content, parent document name, and rank.
    """

    encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    query_embedding = encoder_model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_n) 
     
    vector_results = [
        {
            "id": doc_id,
            "document": doc_content,
            "parent": doc_id.split("_chunk_")[0],   
            "rank": rank + 1  
        }
        for rank, (doc_id, doc_content) in enumerate(zip(results["ids"][0], results["documents"][0]))
    ] 
    return vector_results

def keyword_search(query, top_n=5):
    """
    Performs keyword-based search using BM25 to retrieve top `n` document chunks.

    Parameters:
    - query: The query string for which relevant documents are to be retrieved.
    - top_n: Number of top results to retrieve (default is 5).

    Output:
    - Returns a list of dictionaries containing document details such as ID, content, parent document name, and rank.
    """
    
    with open(BM25_RETRIEVER_FILE, "rb") as f:
        retriever = pickle.load(f)

    with open(CHUNKED_DOCS_FILE, "rb") as f:
        corpus = pickle.load(f)
    
    processed_query = query.split() 
    results = retriever.get_top_n(processed_query, corpus, top_n) 
 
    keyword_results = [
        {
            "id": f'{result.metadata.get("source", "unknown")}_doc_{i}',
            "document": result.page_content,
            "parent": result.metadata.get("source", "unknown"),   
            "rank": i + 1  
        }
        for i, result in enumerate(results)
    ] 
    
    return keyword_results

def reciprocal_reranking(vector_results, keyword_results, k=60):
    """
    Reranks results from vector and keyword search using Reciprocal Rank Fusion (RRF) 
    and merges fused scores by parent document.

    Parameters:
    - vector_results: List of results from vector-based search.
    - keyword_results: List of results from keyword-based search.
    - k: Hyperparameter for RRF to adjust the influence of rank (default is 60).

    Output:
    - Returns a reranked list of parent documents with their combined scores and associated chunks.
    """

    fused_scores = {}
    parent_scores = {}
    parent_documents = {}
 
    for result_list in vector_results:   
        for result in result_list:   
            doc_id = result["id"]
            parent = result["parent"]
            if parent not in parent_documents:
                parent_documents[parent] = []
            parent_documents[parent].append(result["document"])

            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (result["rank"] + k)
 
    for result_list in keyword_results:   
        for result in result_list:   
            doc_id = result["id"]
            parent = result["parent"]
            if parent not in parent_documents:
                parent_documents[parent] = []
            parent_documents[parent].append(result["document"])

            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (result["rank"] + k)
 
    for doc_id, score in fused_scores.items():
        parent = doc_id.split("_")[0]  
        parent_scores[parent] = parent_scores.get(parent, 0) + score
 
    reranked_results = sorted(
        [
            {
                "parent": parent,
                "score": score,
                "documents": parent_documents[parent]
            }
            for parent, score in parent_scores.items()
        ],
        key=lambda x: x["score"], reverse=True
    )
    return reranked_results

def filter_reranked_docs(reranked_results, top_percent=None, threshold=None):
    """
    Filters reranked documents based on a specified top percentage or a probability threshold.

    Parameters:
    - reranked_results: List of reranked documents with scores.
    - top_percent: Percentage of top documents to retain (optional, 0 < top_percent <= 100).
    - threshold: Minimum probability threshold for including documents (optional, 0 <= threshold <= 1).

    Output:
    - Returns a filtered list of documents meeting the specified criteria.
    """

    if not reranked_results:
        return []
 
    reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
 
    scores = [doc["score"] for doc in reranked_results] 
    sum_scores = sum(scores)
    probabilities = [score / sum_scores for score in scores]
     
    print("\nDocuments and Scores")
    for doc, prob in zip(reranked_results, probabilities):
        doc["probability"] = prob
        print(f'{doc["parent"]} - {doc["score"]:.4f}, {doc["probability"]:.4f}')
         
    if top_percent:
        cutoff_index = int(len(reranked_results) * (top_percent / 100))
        filtered_docs = reranked_results[:cutoff_index+1]
    else:
        filtered_docs = reranked_results
 
    if threshold:
        filtered_docs = [doc for doc in filtered_docs if doc["probability"] >= threshold]
 
    if filtered_docs:
        last_prob = filtered_docs[-1]["probability"]
        filtered_docs.extend(
            doc for doc in reranked_results[len(filtered_docs):] if doc["probability"] == last_prob
        )

    if filtered_docs == []:
        filtered_docs = [reranked_results[0]]

    return filtered_docs

def retrieve_parent_document(parent_doc_id):
    """
    Retrieves the content of a parent document from a specified directory.

    Parameters:
    - parent_doc_id: The name of the parent document to retrieve.

    Output:
    - Returns the content of the document as a string if found.
    - Returns None if the document is not found.

    Raises:
    - FileNotFoundError: If the file is not found in the specified directory.
    """
    
    file_path = os.path.join(TXT_DIR, parent_doc_id)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file {parent_doc_id} was not found in the directory {TXT_DIR}.")
        return None
    
def wrap_text(text, width = 100): 
    """
    Wraps text to fit within a specified line width.

    Parameters:
    - text: The input text to be wrapped.
    - width: The maximum width of each line (default is 100 characters).

    Output:
    - Returns the wrapped text where each line does not exceed the specified width.
    """
    
    lines = text.split('\n') 
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines] 
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def generate_responses_with_reranking(user_query, model_name, prompt_template, k, top_n=10, top_percent=None, threshold=None):
    """
    Generates responses for filtered and reranked parent documents using an LLM.

    Parameters:
    - user_query (str): The query to retrieve documents.
    - model_name (str): The name of the LLM model to generate responses.
    - prompt_template (str): Template for prompt generation.
    - k (int): Number of query variations to generate.
    - top_n (int): Number of top results to retrieve from each search.
    - top_percent (float): Percentage of top documents to keep (0 < top_percent <= 100).
    - threshold (float): Minimum probability to include a document (0 <= threshold <= 1).

    Returns:
    - dict: A dictionary mapping parent documents to their LLM-generated responses.
    """

    collection = init_chromadb(COLLECTION_NAME)
    ollama_model = OllamaLLM(model=model_name)

    print(f"User Query \n{user_query}")

    queries = generate_query_variations(user_query, ollama_model, k)     

    vector_results = []
    keyword_results = []

    print("\nRetrieved Documents")
    for i, query in enumerate(queries):
        print(f"Query - '{query}'")
        vector_result = vector_search(query, collection, top_n)
        vector_ids = ' | '.join(item["id"] for item in vector_result if "id" in item) if vector_result else "No results"
        print(f"Vector search results - {vector_ids}")
        vector_results.append(vector_result) 
        keyword_result = keyword_search(query, top_n)
        keyword_ids = ' | '.join(item["id"] for item in keyword_result if "id" in item) if keyword_result else "No results"
        print(f"Keyword search results - {keyword_ids}")
        keyword_results.append(keyword_result)
 
    reranked_results = reciprocal_reranking(vector_results, keyword_results)  
          
    filtered_docs = filter_reranked_docs(reranked_results, top_percent=top_percent, threshold=threshold) 
    print("\nFiltered Document")
    for doc in filtered_docs:
        parent_doc_id = doc["parent"]
        probability = doc.get("probability", 0.0)
        print(parent_doc_id, ' - ', probability)
 
    responses = {}

    for doc in filtered_docs:
        parent_doc_id = doc["parent"] 
        probability = doc.get("probability", 0.0) 
        context = retrieve_parent_document(parent_doc_id)

        if context:
            print("File content loaded successfully!")
        else:
            print("File could not be loaded.") 

       
        print(f"\nGenerating response for parent document: {parent_doc_id} (Probability: {probability:.4f})")

        prompt = PromptTemplate(input_variables=["query", "context"], template=prompt_template)  
        chain = prompt | ollama_model
        response = chain.invoke({"query":user_query, "context":context}) 
        responses[parent_doc_id] = response 
        
        print(wrap_text(response), sep='\n')

    return responses

def find_judgements(query):
    prompt_template = """
    Given the following context related to Indian legal judgments:
    '{context}'

    Please perform the following tasks:
    1. Answer the question: '{query}' based on the provided context.
    2. Identify and list all acts, laws, rules, statutes, and legal provisions explicitly mentioned in the context.
    Provide a clear and concise response for each task.
    """
    no_of_queries = 3
    no_of_docs = 5
 
    response = generate_responses_with_reranking(query, model_name, prompt_template, no_of_queries, no_of_docs, top_percent=None, threshold=0.4)
    return response