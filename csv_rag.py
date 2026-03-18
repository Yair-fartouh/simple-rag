import faiss
import pandas as pd
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def export_csv_to_list(file_path: str = "data/customers-100.csv"):
    data = pd.read_csv(file_path)
    print(f"Raw data preview:\n{data.head()}")

    loader = CSVLoader(file_path=file_path)
    docs = loader.load_and_split()
    return docs


def retrieval_chain(llm, vector_store):
    retriever = vector_store.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if __name__ == '__main__':
    docs = export_csv_to_list()

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    dimension = len(embeddings.embed_query(" "))
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=docs)

    llm = ChatOllama(model="llama3")

    rag_chain = retrieval_chain(llm, vector_store)

    question = "Which company does Sheryl Baxter work for?"
    answer = rag_chain.invoke({"input": question})
    print(f"Q: {question}")
    print(f"A: {answer['answer']}")