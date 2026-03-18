from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from simple_rag import encode_pdf
from helper_functions import retrieve_context_per_question


# Create vector store
path = "/Users/yairfartouh/Downloads/tailored_resume_124327138.pdf"
chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is Yair's profession?"
relevant_docs = chunks_query_retriever.invoke(query)

context = retrieve_context_per_question(query, chunks_query_retriever)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant that answers questions based only on the provided context. "
            "If the answer is not in the context, say you don't know."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        ),
    ]
)

llm = ChatOllama(model="llama3")

response = llm.invoke(
    prompt_template.format_messages(
        context=context,
        question=query
    )
)

print(response.content)