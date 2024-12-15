import bentoml

with bentoml.importing():
    from models.retriever import Retriever
    from models.generator import Generator
    from models.rag_system import RAGSystem

@bentoml.service(name="ragservice", http={
    "cors": {
        "enabled": True,
        "access_control_allow_origins": ["http://localhost:3000/", "*"],
        "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
        "access_control_allow_credentials": True,
        "access_control_allow_headers": ["*"],
        "access_control_max_age": 1200,
        "access_control_expose_headers": ["Content-Length"]
    }
})
class RAGService:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.rag_system = RAGSystem(self.retriever, self.generator)

    @bentoml.api()
    def ask(self, query: str, retriever_top_k=3) -> dict:
        response = self.rag_system.ask(query, retriever_top_k)
        return response

    @bentoml.api()
    def add_document(self, filepath: str) -> None:
        self.retriever.add_document(filepath)
        print(f"Document '{filepath}' successfully added to the retriever.")

    @bentoml.api()
    def add_documents(self, filepaths: list) -> None:
        self.retriever.add_documents(filepaths)
        print(f"Document '{filepaths}' successfully added to the retriever.")

