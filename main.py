import argparse
import sys
from agent.gemma_agent import GemmaAgent
from rl.loop import in_context_rl_loop

def main():
    parser = argparse.ArgumentParser(description="Sudoku In-Context RL using local Gemma model.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="path/to/your/google_gemma-4-26B-A4B-it-Q5_K_M.gguf",
        help="Path to the .gguf Gemma model file"
    )
    parser.add_argument("--iterations", type=int, default=10, help="Number of RL iterations")
    parser.add_argument("--difficulty", type=int, default=40, help="Sudoku difficulty (cells removed)")
    args = parser.parse_args()

    print(f"Loading Gemma model from: {args.model_path}")
    
    try:
        # Note: If your system runs out of VRAM, decrease n_gpu_layers
        agent = GemmaAgent(model_path=args.model_path, n_ctx=4096, n_gpu_layers=-1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid path to your Gemma .gguf model file using --model_path.")
        sys.exit(1)
        
    print("\nStarting In-Context RL Loop...")
    in_context_rl_loop(agent, iterations=args.iterations, difficulty=args.difficulty)

if __name__ == "__main__":
    main()

