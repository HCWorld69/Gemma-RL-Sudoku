# 🧠 Gemma-RL-Sudoku: In-Context Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-Backend-orange?logo=c%2B%2B&logoColor=white)](https://github.com/ggerganov/llama.cpp)
[![Model](https://img.shields.io/badge/Model-Gemma--4--26B-green?logo=google&logoColor=white)](https://ai.google.dev/gemma)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Sudoku_Puzzle_by_L2G-20050714_solution_standardized_layout.svg/1280px-Sudoku_Puzzle_by_L2G-20050714_solution_standardized_layout.svg.png" alt="Sudoku AI" width="400">
</p>

## 🎯 The Goal

This project aims to teach a large language model (**Google's Gemma-4 26B**) how to solve Sudoku puzzles using a purely **In-Context Reinforcement Learning** approach. 

Unlike traditional methods that require massive GPU clusters to update model weights via techniques like GRPO/PPO, this project runs entirely locally on a standard RTX GPU using quantized GGUF models. The model learns "in-context" by proposing Python code to solve the puzzle, executing it in a secure environment, and receiving immediate text-based feedback on its performance to refine its strategy in subsequent iterations.

---

## 🏗️ Architecture

The pipeline consists of three main components:

1. **🎮 The Environment (`env/sudoku.py`)**: A robust, stateful Sudoku game engine that generates puzzles, validates moves, tracks game state, and safely executes the model's generated Python strategies with strict time limits and anti-cheating mechanisms.
2. **🤖 The Agent (`agent/gemma_agent.py`)**: A lightweight wrapper around `llama-cpp-python` that interfaces directly with a locally downloaded quantized Gemma `.gguf` model, bypassing the need for heavy training frameworks.
3. **🔄 The RL Loop (`rl/loop.py`)**: The orchestrator that prompts the model, extracts the code, runs it in the environment, evaluates the outcome (valid moves vs. invalid moves), and injects a descriptive "reward/penalty" back into the model's context for the next turn.

---

## 🚀 Getting Started

### Prerequisites

*   Linux/WSL environment.
*   Python 3.10+
*   A local quantized Gemma model (e.g., `google_gemma-4-26B-A4B-it-Q5_K_M.gguf`).
*   A dedicated GPU (RTX series recommended) for optimal inference speed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HCWorld69/Gemma-RL-Sudoku.git
   cd Gemma-RL-Sudoku
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This installs `llama-cpp-python` for native inference).*

---

## 🕹️ Usage

To kick off the In-Context Reinforcement Learning loop, simply run the provided bash script.

```bash
./run.sh --iterations 10 --difficulty 40 --model_path /path/to/your/gemma.gguf
```

### Arguments

*   `--iterations`: The number of times the model will attempt to generate a strategy, receive feedback, and try again.
*   `--difficulty`: The number of empty cells in the generated Sudoku puzzle (e.g., 40 = Medium).
*   `--model_path`: **(Required)** The absolute path to your locally downloaded `.gguf` file.

### How it Looks in Action

1. The script prompts Gemma with the Sudoku rules and the current board state.
2. Gemma generates a `def strategy(board, initial):` Python function.
3. The environment executes the code securely.
4. You see live feedback:
   > *Result - Valid Moves: 40/40, State: failed*
   > *Feedback: Good attempt. Your function made 40 valid moves before failing... Please improve the strategy.*
5. Gemma uses this feedback to write better code on the next iteration!

---

## 🤝 Contributing

Feel free to fork this project, submit pull requests, or open issues if you find ways to improve the prompt engineering, reward shaping, or environment security!

<div align="center">
  <i>Built with ❤️ by HCWorld69</i>
</div>
