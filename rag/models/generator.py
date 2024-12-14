from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import bentoml

MODEL_ID ="meta-llama/Llama-3.2-1B" #"distilbert/distilbert-base-uncased" #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Generator:
    def __init__(self):
        self.pipe = self._load_generator()

    def _load_generator(self):
        model_ref = bentoml.pytorch.get("llama_generation:latest")
        model = model_ref.load_model()
        tokenizer = AutoTokenizer.from_pretrained(model_ref.path_of("tokenizer"))
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        pipe = pipeline(
            "text-generation",
            model=model.to(DEVICE),
            tokenizer=tokenizer,
            torch_dtype="float16",
            device_map=-1
        )
        return pipe

    def generate(self, prompt, max_new_tokens=300, temperature=0.7, top_k=50, top_p=0.9, num_beams=1):
        # return self.pipe(prompt, max_length=max_length, num_return_sequences=1)[0]["generated_text"]
        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            num_beams=num_beams,
        )[0]["generated_text"]



generator = Generator()

"""bentoml.transformers.save_model(
    "llama_generator",
    generator.pipe.model,
    tokenizer=generator.pipe.tokenizer,
    signatures={"__call__": {"batchable": False}}
)"""
