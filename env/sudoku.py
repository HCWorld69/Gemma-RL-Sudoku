from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random
import copy
import multiprocessing

def _is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[r][col] for r in range(9)]:
        return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False

    return True

def _solve_sudoku(board: List[List[int]]) -> bool:
    """Solve sudoku using backtracking (for puzzle generation)."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def _generate_complete_board(rng: random.Random) -> List[List[int]]:
    """Generate a complete valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]

    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(3):
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]

    # Solve the rest
    _solve_sudoku(board)
    return board

@dataclass
class SudokuGame:
    difficulty: int = 40  # Number of cells to remove (20 = easy, 40 = medium, 50 = hard)
    seed: Optional[int] = None
    _rng: random.Random = field(init = False, repr = False)
    _board: List[List[int]] = field(init = False, repr = False)
    _solution: List[List[int]] = field(init = False, repr = False)
    _initial_board: List[List[int]] = field(init = False, repr = False)
    _moves: int = field(default = 0, init = False, repr = False)
    _state: str = field(default = "ongoing", init = False, repr = False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

        # Generate complete board
        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)

        # Remove cells to create puzzle
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(cells)

        for r, c in cells[:self.difficulty]:
            self._board[r][c] = 0

        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> List[List[int]]:
        """Return current board state."""
        return [row[:] for row in self._board]

    def initial_board(self) -> List[List[int]]:
        """Return initial puzzle state."""
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        """Return game state: 'ongoing', 'success', or 'failed'."""
        return self._state

    def moves(self) -> int:
        """Return number of moves made."""
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        """Place a number on the board. Returns True if valid move."""
        # Validate input
        if not (0 <= row < 9 and 0 <= col < 9):
            self._state = "failed"
            return False

        if not (1 <= num <= 9):
            self._state = "failed"
            return False

        # Can't modify initial cells
        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False
        if self._board[row][col] != 0:
            self._state = "failed"
            return False
        # Check if placement is valid
        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False

        # Place number
        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def _update_state(self) -> None:
        """Update game state based on current board."""
        # Check if puzzle is complete
        if all(self._board[r][c] != 0 for r in range(9) for c in range(9)):
            # Verify solution is correct
            if self._board == self._solution:
                self._state = "success"
            else:
                self._state = "failed"
        else:
            self._state = "ongoing"

    def pretty(self, colors: bool = False) -> str:
        """Pretty print the Sudoku board."""
        lines = []
        lines.append("┌───────┬───────┬───────┐")

        for row in range(9):
            row_str = "│ "
            for col in range(9):
                num = self._board[row][col]
                row_str += str(num) if num != 0 else "."

                if col % 3 == 2:
                    row_str += " │ "
                else:
                    row_str += " "

            lines.append(row_str.rstrip())

            if row == 8:
                lines.append("└───────┴───────┴───────┘")
            elif row % 3 == 2:
                lines.append("├───────┼───────┼───────┤")

        return "\n".join(lines)


def _strategy_runner(strategy, board, initial, return_dict):
    try:
        res = strategy(board, initial)
        return_dict['result'] = res
    except Exception as e:
        return_dict['error'] = str(e)


def execute_strategy(strategy: Callable, game: SudokuGame, timeout: int = 10):
    """Execute a strategy function on a Sudoku game with a timeout."""
    assert callable(strategy)

    max_moves = 100
    valid_moves = 0  # Track successful moves

    while game.state() == "ongoing" and valid_moves < max_moves:
        board = game.board()
        initial = game.initial_board()
        
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=_strategy_runner, args=(strategy, board, initial, return_dict))
        p.start()
        p.join(timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return valid_moves, "failed" # Timeout

        if 'error' in return_dict:
            return valid_moves, "failed"
            
        if 'result' not in return_dict:
            return valid_moves, "failed"
            
        result = return_dict['result']

        # Validate result format
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            return valid_moves, "failed"

        row, col, num = result

        # Validate types
        if not all(isinstance(x, int) for x in [row, col, num]):
            return valid_moves, "failed"

        # Try to place number
        success = game.place_number(row, col, num)

        if success:
            valid_moves += 1  # Count this valid move
        else:
            return valid_moves, "failed"

    if valid_moves >= max_moves and game.state() == "ongoing":
        return valid_moves, "failed"

    return valid_moves, game.state()
