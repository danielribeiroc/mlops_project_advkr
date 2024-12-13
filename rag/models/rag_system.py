from generator import Generator
from retriever import Retriever

class RAGSystem:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def format_prompt(self, contexts, question):
        """
        Formats the combined prompt for the generator.

        Args:
        - contexts: List of context strings retrieved from the retriever.
        - question: The question to ask the generator.

        Returns:
        - A formatted prompt string.
        """
        # Limit the number of contexts to fit within token limit

        # Remove unnecessary start and end tokens
        cleaned_context = contexts.replace("<|startoftext|>", "").replace("<|endoftext|>", ""). replace("[INST]", "").replace("[/INST]", "")

        # Create the final prompt
        prompt = f"""
                You are a highly intelligent and accurate assistant. Below is some context followed by a question. Use the context to provide a precise and concise answer. If the context does not provide sufficient information, respond with "The information is not available in the given context."

                Context: {cleaned_context}

                Question: {question}

                ###STARTANSWER###
                """

        return prompt

    def ask(self, query, retriever_top_k=3, generator_args=None):
        """
        Handles the retrieval-augmented generation process.

        Parameters:
        - query: The input question or prompt.
        - retriever_top_k: Number of top documents to retrieve for the context.
        - generator_args: Additional arguments for the generator (e.g., max_length, temperature).

        Returns:
        - The generated response.
        """
        # Step 1: Retrieve relevant documents
        context = self.retriever.search(query)

        print("Step 1 : \n", context)
        # print(context)
        if not context:
            return "No relevant information found in the knowledge base."

        formatted_prompt = self.format_prompt(context, query)
        print("\n\n Step 2 : \n", formatted_prompt)

        # Combine the retrieved context with the query
        # context_text = "\n".join(context)

        # Step 2: Generate a response using the generator
        # generator_args = generator_args or {}

        gen_response = self.generator.generate(formatted_prompt)

        start_keyword = "###STARTANSWER###"

        stop_keyword = "###ENDANSWER###"

        final_output = gen_response.split(start_keyword)[-1].split(stop_keyword)[0].strip()

        return final_output
