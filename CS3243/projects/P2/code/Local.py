from __future__ import annotations, division
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from random import random
from typing import Any, Dict, List, Tuple
import sys


#############################################################################
# Piece
#############################################################################

class Piece(ABC):
    # This attribute should be overriden by subclasses
    NAME = ""

    def __init__(self, position: Tuple[int, int]):
        self.col = position[0]
        self.row = position[1]

    def position(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @abstractmethod
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        pass

    def __hash__(self) -> int:
        return hash((self.NAME, self.position()))

    def __eq__(self, other: Piece) -> bool:
        return (self.NAME, self.position()) == (other.NAME, other.position())

    @staticmethod
    def construct(name: str, position: Tuple[int, int]) -> Piece:
        if name == King.NAME:
            return King(position)
        elif name == Rook.NAME:
            return Rook(position)
        elif name == Bishop.NAME:
            return Bishop(position)
        elif name == Queen.NAME:
            return Queen(position)
        elif name == Knight.NAME:
            return Knight(position)
        elif name == Ferz.NAME:
            return Ferz(position)
        elif name == Princess.NAME:
            return Princess(position)
        elif name == Empress.NAME:
            return Empress(position)


class King(Piece):
    # Override
    NAME = "King"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if not board.is_obstacle((x, y)):
                    positions.append((x, y))
        return positions


class Rook(Piece):
    # Override
    NAME = "Rook"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_obstacle((x, y)):
                    break
                positions.append((x, y))
                x += delta_x
                y += delta_y
        return positions


class Bishop(Piece):
    # Override
    NAME = "Bishop"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_obstacle((x, y)):
                    break
                positions.append((x, y))
                x += delta_x
                y += delta_y
        return positions


class Queen(Piece):
    # Override
    NAME = "Queen"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_obstacle((x, y)):
                    break
                positions.append((x, y))
                x += delta_x
                y += delta_y
        return positions


class Knight(Piece):
    # Override
    NAME = "Knight"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if not board.is_obstacle((x, y)):
                    positions.append((x, y))
        return positions


class Ferz(Piece):
    # Override
    NAME = "Ferz"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if not board.is_obstacle((x, y)):
                    positions.append((x, y))
        return positions


class Princess(Piece):
    # Override
    NAME = "Princess"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []

        # Bishop's moves
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_obstacle((x, y)):
                    break
                positions.append((x, y))
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if not board.is_obstacle((x, y)):
                    positions.append((x, y))

        return positions


class Empress(Piece):
    # Override
    NAME = "Empress"

    # Override
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        # Assumption: no other pieces will be attacked (due to local search formulation)

        positions = []

        # Rook's moves
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_obstacle((x, y)):
                    break
                positions.append((x, y))
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if not board.is_obstacle((x, y)):
                    positions.append((x, y))

        return positions


#############################################################################
# Board
#############################################################################

class Board:
    SQUARE_ATTACKED = 1
    SQUARE_EMPTY = 0
    SQUARE_OBSTACLE = -1
    SQUARE_PIECE = -2

    def __init__(self, width: int, height: int, grid: List[List[int]]):
        self.width = width
        self.height = height
        self.grid = grid

    def is_inside_grid(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return (x >= 0 and x < self.width) and (y >= 0 and y < self.height)

    def is_attacked(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] >= Board.SQUARE_ATTACKED

    def is_empty(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] == Board.SQUARE_EMPTY

    def is_obstacle(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] == Board.SQUARE_OBSTACLE

    def is_piece(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] == Board.SQUARE_PIECE

    def place(self, piece: Piece) -> int:
        count = 0
        if self.is_attacked(piece.position()) or self.is_obstacle(piece.position()) or self.is_piece(piece.position()):
            return -1
        self.grid[piece.row][piece.col] = Board.SQUARE_PIECE
        for x, y in piece.attacks(self):
            if self.is_piece((x, y)):
                return -1
            elif self.is_empty((x, y)):
                count += 1
            self.grid[y][x] += Board.SQUARE_ATTACKED
        return count


#############################################################################
# State
#############################################################################

class State:
    def __init__(self, pieces: List[Piece], board: Board, value: int):
        self.pieces = pieces
        self.board = board
        self.value = value

    def is_goal(self, k: int) -> bool:
        # Assumption: there are no pieces attacking another pieces
        return len(self.pieces) >= k

    # Generate valid neighbors according to local search formulation
    def neighbors(self, pieces: List[Piece]) -> List[State]:
        neighbors = []
        for piece in pieces:
            if piece not in self.pieces:
                neighbor_board = deepcopy(self.board)
                count = neighbor_board.place(piece)
                if count >= 0:
                    neighbor_value = self.value - count
                    neighbor_pieces = copy(self.pieces)
                    neighbor_pieces.append(piece)
                    neighbors.append(State(neighbor_pieces, neighbor_board, neighbor_value))
        return neighbors


#############################################################################
# Utility Functions
#############################################################################

def to_chess_coord(position: Tuple[int, int]) -> Tuple[str, int]:
    col, row = position
    return (chr(col + 97), row)


def output(pieces: List[Piece]) -> Dict[Tuple[str, int], str]:
    return {to_chess_coord(piece.position()): piece.NAME for piece in pieces}


def swap_tuple_elements(tup: Tuple[Any, Any]) -> Tuple[Any, Any]:
    return (tup[1], tup[0])


def initialize_pieces(position_name: Dict[Tuple[int, int], str]) -> List[Piece]:
    pieces = []
    for position, name in position_name.items():
        pieces.append(Piece.construct(name, swap_tuple_elements(position)))
    return pieces


#############################################################################
# Implement Search Algorithm
#############################################################################

def search(rows, cols, grid, pieces, k):
    k = int(k)
    board = Board(cols, rows, grid)
    pieces = initialize_pieces(pieces)
    max_value = board.width * board.height

    initial = State([], deepcopy(board), max_value)
    if initial.is_goal(k):
        return output(initial.pieces)

    BEAM = 8
    stochastic = False

    while True:
        neighbors = initial.neighbors(pieces)
        neighbors.sort(key=lambda s: s.value, reverse=True)

        # Non-stochastic
        currents = neighbors[:BEAM]
        for current in currents:
            if current.is_goal(k):
                return output(current.pieces)

        # Stochastic
        if stochastic:
            idx = 0
            for neighbor in neighbors:
                if random() < neighbor.value / max_value:
                    currents[idx] = neighbor
                    idx += 1
                    if idx == BEAM:
                        break
            for current in currents:
                if current.is_goal(k):
                    return output(current.pieces)

        while True:
            neighbors = []
            for current in currents:
                neighbors.extend(current.neighbors(pieces))
            if neighbors == []:
                stochastic = True
                break
            neighbors.sort(key=lambda s: s.value, reverse=True)

            # Non-stochastic
            currents = neighbors[:BEAM]
            for current in currents:
                if current.is_goal(k):
                    return output(current.pieces)

            # Stochastic
            idx = 0
            for neighbor in neighbors:
                if random() < neighbor.value / max_value:
                    currents[idx] = neighbor
                    idx += 1
                    if idx == BEAM:
                        break
            for current in currents:
                if current.is_goal(k):
                    return output(current.pieces)


#############################################################################
# Parser function and helper functions
#############################################################################
### DO NOT EDIT/REMOVE THE FUNCTION BELOW###
def parse(testcase):
    handle = open(testcase, "r")

    def get_par(x): return x.split(":")[1]
    rows = int(get_par(handle.readline()))
    cols = int(get_par(handle.readline()))
    grid = [[0 for j in range(cols)] for i in range(rows)]
    k = 0
    pieces = {}

    num_obstacles = int(get_par(handle.readline()))
    if num_obstacles > 0:
        for ch_coord in get_par(handle.readline()).split():  # Init obstacles
            r, c = from_chess_coord(ch_coord)
            grid[r][c] = -1
    else:
        handle.readline()

    k = handle.readline().split(":")[1].strip()  # Read in value of k

    piece_nums = get_par(handle.readline()).split()
    num_pieces = 0
    for num in piece_nums:
        num_pieces += int(num)

    handle.readline()  # Ignore header
    for i in range(num_pieces):
        line = handle.readline()[1:-2]
        coords, piece = add_piece(line)
        pieces[coords] = piece

    return rows, cols, grid, pieces, k


def add_piece(comma_seperated):
    piece, ch_coord = comma_seperated.split(",")
    r, c = from_chess_coord(ch_coord)
    return [(r, c), piece]

# Returns row and col index in integers respectively
def from_chess_coord(ch_coord):
    return (int(ch_coord[1:]), ord(ch_coord[0]) - 97)

### DO NOT EDIT/REMOVE THE FUNCTION HEADER BELOW###
# To return: Goal State which is a dictionary containing a mapping of the position of the grid to the chess piece type.
# Chess Pieces (String): King, Queen, Knight, Bishop, Rook (First letter capitalized)
# Positions: Tuple. (column (String format), row (Int)). Example: ('a', 0)

# Goal State to return example: {('a', 0) : Queen, ('d', 10) : Knight, ('g', 25) : Rook}

def run_local():
    testcase = sys.argv[1]  # Do not remove. This is your input testfile.
    rows, cols, grid, pieces, k = parse(testcase)
    goalstate = search(rows, cols, grid, pieces, k)
    return goalstate  # Format to be returned

