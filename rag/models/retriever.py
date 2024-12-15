import os
import dotenv
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from google.cloud import storage

dotenv.load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
GC_ACCESS_CREDENTIALS = os.getenv("GC_ACCESS_CREDENTIALS")
GC_BUCKET_NAME = os.getenv("GC_BUCKET_NAME")
VECTORSTORE_PREFIX = "faiss_index"
DATA_PREFIX = "data"


class Retriever:
    def __init__(self):

        """
        Initialize the Retriever class. This includes connecting to Google Cloud Storage (if credentials are available),
        loading the embeddings model, and initializing or building the vectorstore.
        """

        self.cloud = True

        try:
            self.client = storage.Client().from_service_account_json(GC_ACCESS_CREDENTIALS)
            self.bucket = self.client.bucket(GC_BUCKET_NAME)
            print("Connected to Google Cloud.")

        except Exception as e:
            print(f"Error: {e}")
            self.cloud = False
            print("Could not connect to Google Cloud. Running in local mode.")

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("Embeddings model loaded.")

        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> FAISS:
        """
        Load the vectorstore from either local storage or Google Cloud Storage.

        Returns:
            FAISS: The loaded vectorstore object.
        """

        # Check if the vectorstore exists in the cloud
        print(f"Checking if vectorstore exists in GCS: {VECTORSTORE_PREFIX}")

        if any(self.bucket.list_blobs(prefix=VECTORSTORE_PREFIX, max_results=1)):
            print(f"Folder '{VECTORSTORE_PREFIX}' exists in GCS. Downloading its content...")

            # Move one level up to the parent directory
            parent_dir = os.path.dirname(os.getcwd())

            # Create a folder named after the prefix one level above
            vectorstore_dir = os.path.join(parent_dir, VECTORSTORE_PREFIX)

            print(f"Download directory: {vectorstore_dir}")
            os.makedirs(vectorstore_dir, exist_ok=True)

            # List all blobs in the folder and download each
            blobs = self.bucket.list_blobs(prefix=VECTORSTORE_PREFIX)
            for blob in blobs:
                if blob.name.endswith('/'):  # Skip folder-like entries
                    continue
                print(f"Downloading file: {blob.name}")
                filename = os.path.basename(blob.name)

                # Construct the full local path for the downloaded file.
                vectorfile_local_path = os.path.join(vectorstore_dir, filename)

                print(f"Downloading file: {blob.name} to {vectorfile_local_path}")
                blob.download_to_filename(vectorfile_local_path)

            print("Successfully downloaded vectorstore from GCS.")

        if os.path.exists("../faiss_index"):
            print("Loading vectorstore from local storage...")
            return FAISS.load_local("../" + VECTORSTORE_PREFIX, self.embeddings, allow_dangerous_deserialization=True)

        else:
            print("No existing vectorstore. Building vectorstore...")
            return self._build_vectorstore()

    def _save_vectorstore_gcs(self, prefix: str) -> None:
        """
        Save the local vectorstore files to Google Cloud Storage.

        Args:
            prefix (str): The folder prefix in GCS where the files will be saved.
        """

        local_dir = os.path.join(os.path.dirname(__file__), "..", prefix)

        for root, _, files in os.walk(local_dir):
            for f in files:
                local_path = os.path.join(root, f)
                # Construct blob path by joining prefix and relative path
                blob_path = os.path.join(prefix, os.path.relpath(local_path, local_dir))
                # Replace backslashes with slashes
                blob_path = blob_path.replace("\\", "/")
                self.bucket.blob(blob_path).upload_from_filename(local_path)

        print("Vectorstore saved to GCS.")

    def _build_vectorstore(self) -> FAISS:

        """
        Build a new vectorstore by loading data, splitting text into chunks, and creating embeddings.

        Returns:
            FAISS: The built vectorstore object.
        """

        # load data from the data folder
        documents = {}
        if self.cloud:
            try:
                blobs = self.client.list_blobs(GC_BUCKET_NAME, prefix="data")
                for blob in blobs:
                    if blob.name.endswith(".txt"):  # Only process text files
                        print(f"Downloading and processing file: {blob.name}")
                        content = blob.download_as_text()  # Read the content of the file
                        doc_id = os.path.splitext(os.path.basename(blob.name))[0]  # Get file name without extension
                        documents[doc_id] = content
                print("Successfully loaded documents from GCS.")

            except Exception as e:
                self.cloud = False
                print(f"Failed to load documents from GCS. Error: {e}")

        else:
            for filename in os.listdir("../" + DATA_PREFIX):
                print(filename)
                if filename.endswith(".txt"):
                    filepath = os.path.join("../" + DATA_PREFIX, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        documents[os.path.splitext(filename)[0]] = f.read()
            print("Successfully loaded documents from local storage.")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        split_documents = []

        # Split the text documents into smaller chunks
        for doc_id, content in documents.items():
            print(f"Splitting document '{doc_id}'.")
            chunks = text_splitter.split_text(content)
            split_documents.extend([Document(page_content=chunk, metadata={"source": doc_id}) for chunk in chunks])

        # Create the FAISS vectorstore
        self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)

        # Save the vector store
        self.vectorstore.save_local("../" + VECTORSTORE_PREFIX)
        self._save_vectorstore_gcs(VECTORSTORE_PREFIX)
        return self.vectorstore

    def add_document(self, filepath: str) -> None:
        """
        Add a single document to the vectorstore.

        Args:
            filepath (str): The path to the text document.
        """

        self.add_documents([filepath])

    def add_documents(self, filepaths: list[str])-> None:
        """
        Add multiple documents to the vectorstore at once.

        Args:
            filepaths list[str]: A list of file paths to text documents.
        """

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")

        text = ""

        if self.cloud:
            for filepath in filepaths:
                filename = os.path.basename(filepath)
                blob_path = os.path.join(DATA_PREFIX, filename).replace("\\", "/")

                blob = self.bucket.blob(blob_path)

                # Upload the file to GCS
                print(f"Uploading document '{filepath}' to GCS at '{blob_path}'.")
                blob.upload_from_filename(filepath)

                # Download the content of the blob as text (to ensure consistency with cloud storage)
                print(f"Downloading document '{blob_path}' from GCS for processing.")
                text = blob.download_as_text()

                # Extract document ID from filename (without extension)
                doc_id = os.path.splitext(filename)[0]

                chunks = text_splitter.split_text(text)
                self.vectorstore.add_texts(chunks, metadatas=[{"source": doc_id}] * len(chunks))

        else:
            for filepath in filepaths:
                print(f"Adding document '{filepath}' to vectorstore.")
                doc_id = os.path.splitext(os.path.basename(filepath))[0]
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                chunks = text_splitter.split_text(text)
                self.vectorstore.add_texts(chunks, metadatas=[{"source": doc_id}] * len(chunks))

        self.vectorstore.save_local("../" + VECTORSTORE_PREFIX)

        self._save_vectorstore_gcs(VECTORSTORE_PREFIX)

        print("Vectorstore updated after adding multiple documents.")

    def search(self, query, top_k=3):
        """
        Search for the top_k most relevant documents.

        Args:
            query (str): The search query.
            top_k (int): Number of top documents to retrieve. Defaults to 3.

        Returns:
            Optional[str]: A string containing the concatenated content of the top_k documents, or None if no results are found.
        """

        results = self.vectorstore.similarity_search(query, k=top_k)
        retrieved_context = " ".join([doc.page_content for doc in results])

        if results:
            return retrieved_context  # Return the most relevant result
        else:
            return None
