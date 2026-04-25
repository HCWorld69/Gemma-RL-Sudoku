import ast
from typing import Tuple, Dict
from env.sudoku import SudokuGame, execute_strategy
from agent.gemma_agent import GemmaAgent, extract_function

def check_python_modules(code_string: str) -> Tuple[bool, str]:
    """Ensure the model didn't use any import statements (anti-cheating)."""
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Importing modules is not allowed."
        return True, "Safe."
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"

def create_locked_down_function(code_string: str):
    """Executes the function code in a locked environment and returns the callable."""
    local_env = {}
    
    # Simple restriction to only allow standard Python built-ins.
    # No globals to avoid hacking the game.
    exec(code_string, {"__builtins__": __builtins__}, local_env)
    
    if "strategy" not in local_env:
        raise ValueError("Function 'strategy' not found in generated code.")
        
    return local_env["strategy"]

def in_context_rl_loop(agent: GemmaAgent, iterations: int = 10, difficulty: int = 40):
    """
    Runs an in-context reinforcement learning loop.
    The model receives the base prompt, generates code, gets evaluated,
    and the text-based feedback (reward) is appended to its context for the next iteration.
    """
    
    base_prompt = """
Create a Sudoku solving strategy using only native Python built-in functions without any import statements.
You are given two lists of lists (9x9 grids):
- board: current state (0 means empty)
- initial: starting puzzle (0 means was empty, numbers are fixed)

Return a tuple (row, col, number) for the next move.
- row: 0-8 (row index)
- col: 0-8 (column index)
- number: 1-9 (digit to place)

Only place numbers in cells that are BOTH empty in initial AND empty in board (initial[row][col] == 0 AND board[row][col] == 0)
Use Sudoku rules: no duplicates in rows, columns, or 3x3 boxes.
Output your function in backticks:
```python
def strategy(board, initial):
    # Your logic here
    return (row, col, number)
```
All helper functions must be inside def strategy. Output only the function.
"""

    current_prompt = base_prompt.strip()

    best_score = -1
    best_strategy = ""

    for i in range(iterations):
        print(f"\n========== ITERATION {i+1} ==========")
        print("Agent is thinking (running inference)...")
        
        # 1. Generate Strategy
        response_text = agent.generate_strategy(current_prompt, max_tokens=600)
        
        # 2. Extract function
        function_code = extract_function(response_text)
        
        if not function_code:
            feedback = "Feedback: You failed to provide the Python code inside backticks. Please follow the format."
            print("Failed to extract function.")
            current_prompt += f"\n\n{feedback}"
            continue
            
        print("Generated Strategy Code:\n" + "-"*40)
        print(function_code)
        print("-" * 40)
        
        # 3. Security Check
        is_safe, error_msg = check_python_modules(function_code)
        if not is_safe:
            feedback = f"Feedback: Code evaluation failed. {error_msg}. Do not use imports or write invalid syntax."
            print(f"Safety/Syntax check failed: {error_msg}")
            current_prompt += f"\n\n{feedback}"
            continue
            
        # 4. Parse Strategy
        try:
            strategy_fn = create_locked_down_function(function_code)
        except Exception as e:
            feedback = f"Feedback: Failed to compile function. Error: {str(e)}."
            print(f"Compilation error: {e}")
            current_prompt += f"\n\n{feedback}"
            continue
            
        # 5. Evaluate Environment
        game = SudokuGame(difficulty=difficulty, seed=42 + i)
        print("Executing strategy in the Sudoku environment...")
        try:
            valid_moves, state = execute_strategy(strategy_fn, game, timeout=10)
        except Exception as e:
            feedback = f"Feedback: Strategy crashed during execution. Error: {str(e)}."
            print(f"Execution crashed: {e}")
            current_prompt += f"\n\n{feedback}"
            continue
            
        # 6. RL Feedback (Reward evaluation converted to text prompt)
        print(f"Result - Valid Moves: {valid_moves}/{difficulty}, State: {state}")
        
        # Update best score
        if valid_moves > best_score:
            best_score = valid_moves
            best_strategy = function_code

        if state == "success":
            print("🎉 SUCCESS! The agent solved the Sudoku puzzle! 🎉")
            break
        elif valid_moves > 0:
            feedback = f"Feedback: Good attempt. Your function made {valid_moves} valid moves before failing or making an invalid move. State: {state}. Please improve the strategy to solve more empty cells without violating Sudoku rules."
        else:
            feedback = f"Feedback: The strategy failed immediately making 0 valid moves. The first proposed move was invalid or crashed. State: {state}. Review your logic carefully."
            
        print(feedback)
        
        # Append the attempt and feedback to the prompt for In-context Learning
        current_prompt = base_prompt.strip() + f"\n\nPrevious Attempt:\n```python\n{function_code}\n```\n{feedback}\n\nPlease try again and fix the issues."

    print("\n========== RL LOOP FINISHED ==========")
    print(f"Best valid moves: {best_score}")
    if best_strategy:
        print("Best Strategy found:\n", best_strategy)
