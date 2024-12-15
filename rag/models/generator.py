import bentoml
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import dotenv

dotenv.load_dotenv()

MODEL_ID = os.getenv("GENERATOR_MODEL")  # distilbert/distilbert-base-uncased" #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Generator:
    def __init__(self):
        self.pipe = self._load_generator()

    def _load_generator(self) -> pipeline:
        """
        Load the text generation model and tokenizer. If the model is not found in BentoML, create and save a new one.

        Returns:
            transformers.pipeline: The text generation pipeline.
        """
        try:
            model_ref = bentoml.transformers.get("llama_generation:latest")
            model = model_ref.load_model()
            tokenizer = AutoTokenizer.from_pretrained(model_ref.path_of("tokenizer"))
            print("Model loaded successfully.")
        except:
            print("Model not found. Creating a new one")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
            model.eval()
            model_ref = bentoml.transformers.save_model(
                "llama_generation",
                model,
                labels={"framework": "transformers", "type": "causal-lm"},
                metadata={
                    "framework_version": "transformers-4.x",
                    "tokenizer_name": MODEL_ID,
                },
            )
            tokenizer_path = model_ref.path_of("tokenizer")
            tokenizer.save_pretrained(tokenizer_path)
            print("Model Created successfully.")

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        pipe = pipeline(
            "text-generation",
            model=model.to(DEVICE),
            tokenizer=tokenizer,
            torch_dtype="float16",  # float32 if cpu's memory is enough
            device_map=-1
        )
        return pipe

    def generate(self, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7, top_k: int = 50,
                 top_p: float = 0.9,
                 num_beams: int = 1) -> str:
        """
        Generate text based on the given prompt using the loaded model pipeline.

        Args:
            prompt (str): The input text prompt for generation.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 300.
            temperature (float): Sampling temperature. Higher values make output more random. Defaults to 0.7.
            top_k (int): Limits the sampling pool to top_k tokens. Defaults to 50.
            top_p (float): Cumulative probability for nucleus sampling. Defaults to 0.9.
            num_beams (int): Number of beams for beam search. Defaults to 1.

        Returns:
            str: The generated text.
        """
        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            num_beams=num_beams,
        )[0]["generated_text"]
