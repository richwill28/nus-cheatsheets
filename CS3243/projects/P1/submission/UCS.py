from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import total_ordering
from typing import Any, DefaultDict, List, Tuple
import heapq
import sys


#############################################################################
# Piece
#############################################################################

@total_ordering
class Piece(ABC):
    def __init__(self, position: Tuple[int, int]):
        self.col = position[0]
        self.row = position[1]

    def position(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @abstractmethod
    def notation(self) -> str:
        pass

    @abstractmethod
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        pass

    def __hash__(self) -> int:
        return hash((self.notation(), self.position()))

    def __eq__(self, other: State) -> bool:
        return (self.notation(), self.position()) == (other.notation(), other.position())

    def __lt__(self, other: State) -> bool:
        return (self.notation(), self.position()) < (other.notation(), other.position())


class King(Piece):
    # Override
    def notation(self) -> str:
        return "K"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)) and board.is_empty((x, y)) and not board.is_attacked((x, y)):
                legal_moves.append((x, y))

        return legal_moves


class Rook(Piece):
    # Override
    def notation(self) -> str:
        return "R"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if not board.is_empty((x, y)):
                    break
                legal_moves.append((x, y))
                x += delta_x
                y += delta_y

        return legal_moves


class Bishop(Piece):
    # Override
    def notation(self) -> str:
        return "B"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if not board.is_empty((x, y)):
                    break
                legal_moves.append((x, y))
                x += delta_x
                y += delta_y

        return legal_moves


class Queen(Piece):
    # Override
    def notation(self) -> str:
        return "Q"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if not board.is_empty((x, y)):
                    break
                legal_moves.append((x, y))
                x += delta_x
                y += delta_y

        return legal_moves


class Knight(Piece):
    # Override
    def notation(self) -> str:
        return "N"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)) and board.is_empty((x, y)):
                legal_moves.append((x, y))

        return legal_moves


class Ferz(Piece):
    # Override
    def notation(self) -> str:
        return "F"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)) and board.is_empty((x, y)):
                legal_moves.append((x, y))

        return legal_moves


class Princess(Piece):
    # Override
    def notation(self) -> str:
        return "P"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        # Bishop's moves
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if not board.is_empty((x, y)):
                    break
                legal_moves.append((x, y))
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)) and board.is_empty((x, y)):
                legal_moves.append((x, y))

        return legal_moves


class Empress(Piece):
    # Override
    def notation(self) -> str:
        return "E"

    # Override
    def legal_moves(self, board: Board) -> List[Tuple[int, int]]:
        legal_moves = []

        # Rook's moves
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if not board.is_empty((x, y)):
                    break
                legal_moves.append((x, y))
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)) and board.is_empty((x, y)):
                legal_moves.append((x, y))

        return legal_moves


#############################################################################
# Obstacle
#############################################################################

@total_ordering
class Obstacle:
    def __init__(self, position: Tuple[int, int]):
        self.col = position[0]
        self.row = position[1]

    def position(self) -> Tuple[int, int]:
        return (self.col, self.row)

    def notation(self) -> str:
        return "X"

    def __hash__(self) -> int:
        return hash((self.notation(), self.position()))

    def __eq__(self, other: State) -> bool:
        return (self.notation(), self.position()) == (other.notation(), other.position())

    def __lt__(self, other: State) -> bool:
        return (self.notation(), self.position()) < (other.notation(), other.position())


#############################################################################
# Board
#############################################################################

@total_ordering
class Board:
    SQUARE_ATTACKED = 0

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cost: DefaultDict[Tuple[int, int], int] = defaultdict(lambda: 1)
        self.grid: List[List[Any]] = [[self.cost[(x, y)] for x in range(width)] for y in range(height)]

    def __str__(self) -> str:
        string = ""
        for row in self.grid:
            for square in row:
                string += str(square) + " "
            string += "\n"
        return string

    def set_square_costs(self, square_cost: DefaultDict[Tuple[int, int], int]):
        self.cost = square_cost
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.is_empty((x, y)):
                    self.grid[y][x] = square_cost[(x, y)]

    def place_obstacles(self, obstacles: List[Obstacle]):
        for obstacle in obstacles:
            self.grid[obstacle.row][obstacle.col] = obstacle.notation()

    def place_enemy_pieces(self, enemy_pieces: List[Piece]):
        for piece in enemy_pieces:
            self.grid[piece.row][piece.col] = piece.notation()

        # Reset attacked squares
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.is_empty((x, y)):
                    self.grid[y][x] = self.cost[(x, y)]

        # Update attacked squares
        for piece in enemy_pieces:
            for x, y in piece.legal_moves(self):
                self.grid[y][x] = Board.SQUARE_ATTACKED

    def is_inside_grid(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        if (x >= 0 and x < self.width) and (y >= 0 and y < self.height):
            return True
        return False

    def is_empty(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        # Only occupied squares (either by obstacles or enemy pieces) are denoted with string
        return False if isinstance(self.grid[y][x], str) else True

    def is_attacked(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return True if (self.grid[y][x] == Board.SQUARE_ATTACKED) else False

    def __hash__(self) -> int:
        return hash(self.grid)

    def __eq__(self, other: State) -> bool:
        return self.grid == self.grid

    def __lt__(self, other: State) -> bool:
        return self.grid < self.grid


#############################################################################
# State
#############################################################################

@total_ordering
class State:
    def __init__(self, king: King, board: Board):
        self.parent = None
        self.cost = 0
        self.king = king
        self.board = board

    def __str__(self) -> str:
        string = "Parent King: "
        if self.parent:
            string += str(to_chess_coord(self.parent.king.position())) + "\n"
        else:
            string += "-\n"
        string += "Current King: " + str(to_chess_coord(self.king.position())) + "\n"
        string += "Board: \n" + str(self.board)
        return string

    def estimate_cost_to_goal(self, goals: List[Tuple[int, int]], heuristic=lambda: 0) -> int:
        # Assertion: goal is represented as (col, row)
        estimation = heuristic(self.king.position(), goals[0])
        for goal in goals:
            estimation = min(estimation, heuristic(self.king.position(), goal))
        return int(estimation)

    def is_goal(self, goals: List[Tuple[int, int]]) -> bool:
        for goal in goals:
            # Assertion: goal is represented as (col, row)
            if goal == self.king.position():
                return True
        return False

    def path(self) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        path = deque()
        current = self
        parent = self.parent
        while parent:
            path.appendleft([to_chess_coord(parent.king.position()), to_chess_coord(current.king.position())])
            current = parent
            parent = current.parent
        return list(path)

    def set_cost(self, cost: int):
        self.cost = cost

    def set_parent(self, parent: State):
        self.parent = parent

    def successors(self) -> List[State]:
        successors: List[State] = []
        for x, y in self.king.legal_moves(self.board):
            successor = State(King((x, y)), self.board)
            successor.set_cost(self.cost + successor.board.grid[y][x])
            successor.set_parent(self)
            successors.append(successor)
        return successors

    def __hash__(self) -> int:
        # Assuming board is static and same for all states
        return hash(self.king)

    def __eq__(self, other: State) -> bool:
        # Assuming board is static and same for all states
        return self.king == other.king

    def __lt__(self, other: State) -> bool:
        # Assuming board is static and same for all states
        return self.king < other.king


#############################################################################
# Utility Functions
#############################################################################

def construct_piece(name: str, position: Tuple[int, int]) -> Piece:
    if name == "King":
        return King(position)
    elif name == "Rook":
        return Rook(position)
    elif name == "Bishop":
        return Bishop(position)
    elif name == "Queen":
        return Queen(position)
    elif name == "Knight":
        return Knight(position)
    elif name == "Ferz":
        return Ferz(position)
    elif name == "Princess":
        return Princess(position)
    elif name == "Empress":
        return Empress(position)


def initialize_board(width: int, height: int, square_cost: DefaultDict[Tuple[int, int], int], obstacles: List[Obstacle], enemy_pieces: List[Piece]) -> Board:
    board = Board(width, height)
    board.set_square_costs(square_cost)
    board.place_obstacles(obstacles)
    board.place_enemy_pieces(enemy_pieces)
    return board


def initialize_enemy_pieces(enemy_pieces: List[Tuple[str, Tuple[int, int]]]) -> List[Piece]:
    enemies: List[Piece] = []
    for name, position in enemy_pieces:
        position = swap_tuple_elements(position)
        enemy = construct_piece(name, position)
        enemies.append(enemy)
    return enemies


def initialize_goals(goals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    for i in range(len(goals)):
        goals[i] = swap_tuple_elements(goals[i])
    return goals


def initialize_obstacles(grid: List[List[int]]) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] == -1:
                obstacles.append(Obstacle((x, y)))
    return obstacles


def initialize_own_piece(own_pieces: List[Tuple[str, Tuple[int, int]]]) -> Tuple[int, int]:
    return swap_tuple_elements(own_pieces[0][1])


def initialize_square_cost(grid: List[List[int]]) -> DefaultDict[Tuple[int, int], int]:
    square_cost = defaultdict(lambda: 1)
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            if grid[y][x] > 1:
                square_cost[(x, y)] = grid[y][x]
    return square_cost


def swap_tuple_elements(tup: Tuple[Any, Any]) -> Tuple[Any, Any]:
    return (tup[1], tup[0])


def to_chess_coord(position: Tuple[int, int]) -> Tuple[str, int]:
    col, row = position
    return (chr(col + 97), row)


#############################################################################
# Priority Queue
#############################################################################

class PriorityQueue:
    def __init__(self, order="min", priority=lambda x: x):
        self.heap = []
        # Ignore error, initialize either a min or max priority queue
        self.priority = priority if (order == "min") else lambda x: -priority(x)

    def push(self, item: Any):
        # Naive implementation to ensure no duplicate items
        for i in range(len(self.heap)):
            if self.heap[i][1] == item:
                del self.heap[i]
                heapq.heapify(self.heap)
                break
        heapq.heappush(self.heap, (self.priority(item), item))

    def pop(self) -> Any:
        if not self.heap:
            # Warning: silent failure
            return None
        return heapq.heappop(self.heap)[1]

    def __len__(self) -> int:
        return len(self.heap)

    def __contains__(self, key) -> bool:
        for _, value in self.heap:
            if value == key:
                return True
        return False


#############################################################################
# Implement Search Algorithm
#############################################################################

def search(rows: int, cols: int, grid: List[List[int]], enemy_pieces: List[Tuple[str, Tuple[int, int]]], own_pieces: List[Tuple[str, Tuple[int, int]]], goals: List[Tuple[int, int]]) -> Tuple[List[Tuple[Tuple[str, int], Tuple[str, int]]], int]:
    # Position of a chess piece should be denoted by (x, y) = (col, row).
    # A pre-processing stage has to be conducted to the parameters to ensure
    # consistency with the rest of the control flow.

    # Pre-processing stage
    square_cost = initialize_square_cost(grid)
    obstacles = initialize_obstacles(grid)
    enemy_pieces = initialize_enemy_pieces(enemy_pieces)
    own_piece = initialize_own_piece(own_pieces)
    goals = initialize_goals(goals)
    board = initialize_board(cols, rows, square_cost, obstacles, enemy_pieces)

    # Initialize the required data structures
    initial_state = State(King(own_piece), board)
    visited: DefaultDict[State, int] = defaultdict(lambda: 0)
    frontier = PriorityQueue("min", lambda state: visited[state])
    frontier.push(initial_state)

    # Main routine of the search algorithm
    while frontier:
        current: State = frontier.pop()
        if current.is_goal(goals):
            return (current.path(), visited[current])
        for successor in current.successors():
            if successor not in visited or successor.cost < visited[successor]:
                # Order of execution is important because frontier depends on visited
                visited[successor] = successor.cost
                frontier.push(successor)

    # Return empty list and zero path cost if no goal can be found
    return ([], 0)


#############################################################################
# Parser function and helper functions
#############################################################################

### DO NOT EDIT/REMOVE THE FUNCTION BELOW###
# Return number of rows, cols, grid containing obstacles and step costs of coordinates, enemy pieces, own piece, and goal positions
def parse(testcase):
    handle = open(testcase, "r")

    def get_par(x): return x.split(":")[1]
    rows = int(get_par(handle.readline()))  # Integer
    cols = int(get_par(handle.readline()))  # Integer
    # Dictionary, label empty spaces as 1 (Default Step Cost)
    grid = [[1 for j in range(cols)] for i in range(rows)]
    enemy_pieces = []  # List
    own_pieces = []  # List
    goals = []  # List

    handle.readline()  # Ignore number of obstacles
    for ch_coord in get_par(handle.readline()).split():  # Init obstacles
        r, c = from_chess_coord(ch_coord)
        grid[r][c] = -1  # Label Obstacle as -1

    handle.readline()  # Ignore Step Cost header
    line = handle.readline()
    while line.startswith("["):
        line = line[1:-2].split(",")
        r, c = from_chess_coord(line[0])
        # Reinitialize step cost for coordinates with different costs
        grid[r][c] = int(line[1]) if grid[r][c] == 1 else grid[r][c]
        line = handle.readline()

    line = handle.readline()  # Read Enemy Position
    while line.startswith("["):
        line = line[1:-2]
        piece = add_piece(line)
        enemy_pieces.append(piece)
        line = handle.readline()

    # Read Own King Position
    line = handle.readline()[1:-2]
    piece = add_piece(line)
    own_pieces.append(piece)

    # Read Goal Positions
    for ch_coord in get_par(handle.readline()).split():
        r, c = from_chess_coord(ch_coord)
        goals.append((r, c))

    return rows, cols, grid, enemy_pieces, own_pieces, goals


def add_piece(comma_seperated) -> Piece:
    piece, ch_coord = comma_seperated.split(",")
    r, c = from_chess_coord(ch_coord)
    return [piece, (r, c)]


def from_chess_coord(ch_coord):
    return (int(ch_coord[1:]), ord(ch_coord[0]) - 97)


### DO NOT EDIT/REMOVE THE FUNCTION HEADER BELOW###
# To return: List of moves and nodes explored
def run_UCS():
    testcase = sys.argv[1]
    rows, cols, grid, enemy_pieces, own_pieces, goals = parse(testcase)
    moves, pathcost = search(rows, cols, grid, enemy_pieces, own_pieces, goals)
    return moves, pathcost
