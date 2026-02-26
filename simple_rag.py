from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import embeddings

loader = PyPDFLoader("/Users/yairfartouh/Downloads/YairFartouh_AIEngineer.pdf")
loaded_docs = loader.load()

print(len(loaded_docs))
print(loaded_docs[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
document_chunks = splitter.split_documents(loaded_docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(document_chunks, embeddings)
retriever = vector_store.as_retriever(k=3)

query = "What is yair's profession?"
relevant_docs = retriever.invoke(query)
context = "\n\n".join(d.page_content for d in relevant_docs)

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
print(response)
