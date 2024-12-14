from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_FOLDER = "../data/"
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../faiss_index"))

class Retriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self):

        if os.path.exists(PERSIST_DIR):
            print("Loading vectorstore from local storage.")
            return FAISS.load_local(PERSIST_DIR, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Building vectorstore.")
            return self._build_vectorstore()

    def _build_vectorstore(self):
        #load data from the data folder
        documents = {}
        for filename in os.listdir(DATA_FOLDER):
            print(filename)
            if filename.endswith(".txt"):
                filepath = os.path.join(DATA_FOLDER, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents[os.path.splitext(filename)[0]] = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        split_documents = []

        # Split the text documents into smaller chunks
        for doc_id, content in documents.items():
            print(f"Splitting document '{doc_id}'.")
            print(content)
            chunks = text_splitter.split_text(content)
            split_documents.extend([Document(page_content=chunk, metadata={"source": doc_id}) for chunk in chunks])

        # Create the FAISS vectorstore
        self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)

        # Save the vector store
        self.vectorstore.save_local(PERSIST_DIR)

        return self.vectorstore

    def add_document(self, filepath):
        """Add a single document to the vectorstore."""
        with open(filepath, 'r') as f:
            text = f.read()
        self.vectorstore.add_texts([text])
        self.vectorstore.save_local("faiss_index")
        print(f"Document '{filepath}' added to vectorstore.")

    def search(self, query,top_k=3, vectorstore_path=PERSIST_DIR):
        """Search for the top_k most relevant documents."""
        #vectorstore = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        documents = self.vectorstore.docstore

        results = self.vectorstore.similarity_search(query, k=top_k)
        retrieved_context = " ".join([doc.page_content for doc in results])

        if results:
            return retrieved_context  # Return the most relevant result
        else:
            return None


