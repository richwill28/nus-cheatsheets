from __future__ import annotations, division
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
        # Assumption: no other pieces will be attacked (due to CSP)

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
# Global variables
#############################################################################

piece_names = [King.NAME, Queen.NAME, Bishop.NAME, Rook.NAME,
               Knight.NAME, Ferz.NAME, Princess.NAME, Empress.NAME]

variable_order = {
    Queen.NAME: 0,
    Empress.NAME: 1,
    Princess.NAME: 2,
    Rook.NAME: 3,
    Bishop.NAME: 4,
    Knight.NAME: 5,
    King.NAME: 6,
    Ferz.NAME: 7
}


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
        self.initialize_domain()

    def initialize_domain(self):
        self.domain = set()     # A set of valid positions according to CSP
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_obstacle((x, y)):
                    self.domain.add((x, y))

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

    def get_rank(self, piece: Piece) -> int:
        rank = 0    # How many empty squares are attacked as a result of placing the piece
        for x, y in piece.attacks(self):
            if self.is_empty((x, y)):
                rank += 1
            elif self.is_piece((x, y)):
                return -1
        return rank

    def place(self, piece: Piece):
        # Assumption: the piece will be placed in a valid position
        self.grid[piece.row][piece.col] = Board.SQUARE_PIECE
        self.domain.remove(piece.position())
        for x, y in piece.attacks(self):
            if self.is_empty((x, y)):
                self.domain.remove((x, y))
            self.grid[y][x] += Board.SQUARE_ATTACKED

    def remove(self, piece: Piece):
        # Assumption: the piece had been placed to the grid before
        self.grid[piece.row][piece.col] = Board.SQUARE_EMPTY
        self.domain.add(piece.position())
        for x, y in piece.attacks(self):
            self.grid[y][x] -= Board.SQUARE_ATTACKED
            if self.is_empty((x, y)):
                self.domain.add((x, y))


#############################################################################
# State
#############################################################################

class State:
    def __init__(self, board: Board, variables: List[str], assignment: Dict[Tuple[int, int], str]):
        self.board = board
        self.variables = variables
        self.assignment = assignment

    def is_complete(self) -> bool:
        return len(self.variables) == 0

    def select_unassigned_variable(self) -> str:
        return self.variables.pop()

    def restore_unassigned_variable(self, variable: str):
        # Assumption: the variable had been selected before
        global variable_order
        self.variables.append(variable)
        self.variables.sort(key=lambda name: variable_order[name], reverse=True)

    def order_domain_values(self, variable: str) -> List[Tuple[int, int]]:
        position_rank = []
        for position in self.board.domain:
            piece = Piece.construct(variable, position)
            rank = self.board.get_rank(piece)
            if rank > -1:
                position_rank.append((position, rank))
        return [position for position, _ in sorted(position_rank, key=lambda item: item[1])]

    def assign(self, piece: Piece):
        # Assumption: the position of the piece is valid (i.e. inside the board domain)
        self.assignment[piece.position()] = piece.NAME
        self.board.place(piece)

    def unassign(self, piece: Piece):
        # Assumption: the piece had been assigned before
        del self.assignment[piece.position()]
        self.board.remove(piece)


#############################################################################
# Utility Functions
#############################################################################

def to_chess_coord(position: Tuple[int, int]) -> Tuple[str, int]:
    col, row = position
    return (chr(col + 97), row)


def output(assignment: Dict[Tuple[int, int], str]) -> Dict[Tuple[str, int], str]:
    return {to_chess_coord(position): name for position, name in assignment.items()}


def initialize_variables(num_pieces: List[int]) -> List[str]:
    global piece_names, variable_order
    variables = []
    for i in range(len(num_pieces)):
        for _ in range(num_pieces[i]):
            variables.append(piece_names[i])
    variables.sort(key=lambda name: variable_order[name], reverse=True)
    return variables


#############################################################################
# Implement Search Algorithm
#############################################################################

def backtrack(csp: State) -> Dict[Tuple[str, int], str]:
    if csp.is_complete():
        return output(csp.assignment)

    variable = csp.select_unassigned_variable()
    domain = csp.order_domain_values(variable)

    for value in domain:
        piece = Piece.construct(variable, value)
        csp.assign(piece)
        result = backtrack(csp)
        if result is not None:
            return result
        csp.unassign(piece)

    csp.restore_unassigned_variable(variable)
    return None


def search(rows, cols, grid, num_pieces):
    board = Board(cols, rows, grid)
    variables = initialize_variables(num_pieces)
    assignment = dict()
    csp = State(board, variables, assignment)
    return backtrack(csp)


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

    num_obstacles = int(get_par(handle.readline()))
    if num_obstacles > 0:
        for ch_coord in get_par(handle.readline()).split():  # Init obstacles
            r, c = from_chess_coord(ch_coord)
            grid[r][c] = -1
    else:
        handle.readline()

    piece_nums = get_par(handle.readline()).split()
    # List in the order of King, Queen, Bishop, Rook, Knight
    num_pieces = [int(x) for x in piece_nums]

    return rows, cols, grid, num_pieces


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

def run_CSP():
    testcase = sys.argv[1]  # Do not remove. This is your input testfile.
    rows, cols, grid, num_pieces = parse(testcase)
    goalstate = search(rows, cols, grid, num_pieces)
    return goalstate  # Format to be returned
