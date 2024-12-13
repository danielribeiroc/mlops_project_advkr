import bentoml
from bentoml.io import JSON, Text

with bentoml.importing():
    from models.retriever import Retriever
    from models.generator import Generator
    from models.rag_system import RAGSystem

"""# Export retriever as a BentoML model
class RetrieverService:
    def __init__(self):
        self.retriever = Retriever()

    def save(self):
        bentoml.picklable_model.save_model(
            "retriever_model",
            self.retriever,
            signatures={
                "search": {
                    "batchable": False,
                    "input_spec": JSON(),
                    "output_spec": JSON()
                },
                "add_document": {
                    "batchable": False,
                    "input_spec": JSON(),
                    "output_spec": Text()
                }
            },
        )

# Export generator as a BentoML model
class GeneratorService:
    def __init__(self):
        self.generator = Generator()

    def save(self):
        bentoml.transformers.save_model(
            "generator_model",
            self.generator,
            signatures={
                "generate": {
                    "batchable": False,
                    "input_spec": JSON(),
                    "output_spec": Text()
                }
            },
        )

# Create the BentoML Service for RAG System"""


class RAGService:
    def __init__(self):
        #self.retriever = bentoml.picklable_model.load_model("retriever_model")
        #self.generator = bentoml.picklable_model.load_model("generator_model")
        self.retriever = Retriever()
        self.generator = Generator()

        self.rag_system = RAGSystem(self.retriever, self.generator)

    def ask(self, query, retriever_top_k=3):
        response = self.rag_system.ask(query, retriever_top_k)
        return response

    def add_document(self, filepath):
        self.retriever.add_document(filepath)
        return f"Document '{filepath}' successfully added to the retriever."

# BentoML Service Definition
#
svc = bentoml.Service("RAG_Service")
@svc.api(input=JSON(), output=Text())
def ask_rag(input_data):
    query = input_data.get("query")
    retriever_top_k = input_data.get("retriever_top_k", 3)

    rag_service = RAGService()
    response = rag_service.ask(query, retriever_top_k)
    return response

@svc.api(input=JSON(), output=Text())
def add_document(input_data):
    filepath = input_data.get("filepath")

    rag_service = RAGService()
    response = rag_service.add_document(filepath)
    return response
