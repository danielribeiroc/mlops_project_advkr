from retriever import Retriever
from generator import Generator
from rag_system import RAGSystem

if __name__ == '__main__':
    rag = RAGSystem(Retriever(), Generator())
    print(rag.ask("What is THE HOBBY OF Ruben?"))