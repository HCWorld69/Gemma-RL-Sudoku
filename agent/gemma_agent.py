import os
from llama_cpp import Llama

class GemmaAgent:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Initializes the Gemma Agent using llama.cpp.
        n_gpu_layers=-1 will offload all layers to the GPU.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
            
        print(f"Loading model {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("Model loaded successfully.")

    def generate_strategy(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Given a prompt, generates a response from the model.
        Since it's an instruction-tuned model, we wrap it in a chat template format if needed.
        Gemma format: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
        """
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<end_of_turn>", "<eos>"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()

def extract_function(text: str) -> str:
    """Extract Python function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        
        # Sometimes the model writes ```python
        if text[first:first+6].lower() == "python":
            first += 6
            
        second = text.find("```", first)
        if second != -1:
            fx = text[first:second].strip()
            # Find the start of the function
            def_idx = fx.find("def ")
            if def_idx != -1:
                return fx[def_idx:]
                
    # Fallback: if no backticks, try to find def strategy
    def_idx = text.find("def strategy(board, initial):")
    if def_idx != -1:
        return text[def_idx:].strip()
        
    return ""
