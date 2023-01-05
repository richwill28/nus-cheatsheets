from __future__ import annotations, division
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import sys


#############################################################################
# Piece
#############################################################################

class Piece:
    WHITE = 1
    BLACK = -1

    # This attribute should be overriden by subclasses
    NAME = ""
    NOTATION = ""
    VALUE = 0

    def __init__(self, color: int, position: Tuple[int, int]):
        self.color = color
        self.col = position[0]
        self.row = position[1]

    def position(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @abstractmethod
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def attacks(self, board: Board) -> List[Tuple[int, int]]:
        pass

    @staticmethod
    def construct(name: str, color: str, position: Tuple[int, int]) -> Piece:
        color = Piece.WHITE if color == "White" else Piece.BLACK
        if name == King.NAME:
            return King(color, position)
        elif name == Rook.NAME:
            return Rook(color, position)
        elif name == Bishop.NAME:
            return Bishop(color, position)
        elif name == Queen.NAME:
            return Queen(color, position)
        elif name == Knight.NAME:
            return Knight(color, position)
        elif name == Ferz.NAME:
            return Ferz(color, position)
        elif name == Princess.NAME:
            return Princess(color, position)
        elif name == Empress.NAME:
            return Empress(color, position)
        elif name == Pawn.NAME:
            return Pawn(color, position)


class King(Piece):
    # Override
    NAME = "King"
    NOTATION = "K"
    VALUE = 1000000

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)) and board.is_attack((x, y), self.color):
                    positions.append((x, y))
        return positions


class Rook(Piece):
    # Override
    NAME = "Rook"
    NOTATION = "R"
    VALUE = 500

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)):
                    if board.is_attack((x, y), self.color):
                        positions.append((x, y))
                    break
                x += delta_x
                y += delta_y
        return positions


class Bishop(Piece):
    # Override
    NAME = "Bishop"
    NOTATION = "B"
    VALUE = 300

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)):
                    if board.is_attack((x, y), self.color):
                        positions.append((x, y))
                    break
                x += delta_x
                y += delta_y
        return positions


class Queen(Piece):
    # Override
    NAME = "Queen"
    NOTATION = "Q"
    VALUE = 900

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)):
                    if board.is_attack((x, y), self.color):
                        positions.append((x, y))
                    break
                x += delta_x
                y += delta_y
        return positions


class Knight(Piece):
    # Override
    NAME = "Knight"
    NOTATION = "N"
    VALUE = 300

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)) and board.is_attack((x, y), self.color):
                    positions.append((x, y))
        return positions


class Ferz(Piece):
    # Override
    NAME = "Ferz"
    NOTATION = "F"
    VALUE = 200

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)) and board.is_attack((x, y), self.color):
                    positions.append((x, y))
        return positions


class Princess(Piece):
    # Override
    NAME = "Princess"
    NOTATION = "S"
    VALUE = 500

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []

        # Bishop's moves
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)):
                    if board.is_attack((x, y), self.color):
                        positions.append((x, y))
                    break
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)) and board.is_attack((x, y), self.color):
                    positions.append((x, y))

        return positions


class Empress(Piece):
    # Override
    NAME = "Empress"
    NOTATION = "E"
    VALUE = 700

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []

        # Rook's moves
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            while board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)):
                    if board.is_attack((x, y), self.color):
                        positions.append((x, y))
                    break
                x += delta_x
                y += delta_y

        # Knight's moves
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta_x, delta_y in deltas:
            x = self.col + delta_x
            y = self.row + delta_y
            if board.is_inside_grid((x, y)):
                if board.is_empty((x, y)):
                    positions.append((x, y))
                elif board.is_piece((x, y)) and board.is_attack((x, y), self.color):
                    positions.append((x, y))

        return positions


class Pawn(Piece):
    # Override
    NAME = "Pawn"
    NOTATION = "P"
    VALUE = 100

    # Override
    def moves(self, board: Board) -> List[Tuple[int, int]]:
        positions = []

        if self.color == Piece.WHITE:
            position = (self.col, self.row + 1)
            if board.is_inside_grid(position) and board.is_empty(position):
                positions.append(position)
            position = (self.col - 1, self.row + 1)
            if board.is_inside_grid(position) and board.is_piece(position) and board.is_attack(position, self.color):
                positions.append(position)
            position = (self.col + 1, self.row + 1)
            if board.is_inside_grid(position) and board.is_piece(position) and board.is_attack(position, self.color):
                positions.append(position)
        elif self.color == Piece.BLACK:
            position = (self.col, self.row - 1)
            if board.is_inside_grid(position) and board.is_empty(position):
                positions.append(position)
            position = (self.col - 1, self.row - 1)
            if board.is_inside_grid(position) and board.is_piece(position) and board.is_attack(position, self.color):
                positions.append(position)
            position = (self.col + 1, self.row - 1)
            if board.is_inside_grid(position) and board.is_piece(position) and board.is_attack(position, self.color):
                positions.append(position)

        return positions


#############################################################################
# Board
#############################################################################

class Board:
    def __init__(self, width: int, height: int, grid: List[List[Piece]]):
        self.width = width
        self.height = height
        self.grid = grid

    def is_inside_grid(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return (x >= 0 and x < self.width) and (y >= 0 and y < self.height)

    def is_empty(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] is None

    def is_piece(self, position: Tuple[int, int]) -> bool:
        x = position[0]
        y = position[1]
        return self.grid[y][x] is not None

    def is_attack(self, position: Tuple[int, int], color: int) -> bool:
        x = position[0]
        y = position[1]
        return color != self.grid[y][x].color

    def is_check(self, piece: Piece) -> bool:
        for x, y in piece.moves(self):
            if self.is_piece((x, y)):
                if self.grid[y][x].NAME == King.NAME:
                    return True
        return False

    @staticmethod
    def construct(width: int, height: int, gameboard: Dict[Tuple[str, int], Tuple[str, str]]) -> Board:
        grid = [[None for _ in range(width)] for _ in range(height)]
        for position, piece in gameboard.items():
            position = from_chess_pos(position)
            grid[position[1]][position[0]] = Piece.construct(
                piece[0], piece[1], position)
        return Board(width, height, grid)

    def __str__(self) -> str:
        string = ""
        for row in self.grid:
            for square in row:
                if square is None:
                    string += " "
                else:
                    string += square.NOTATION
                string += " "
            string += "\n"
        return string


#############################################################################
# State
#############################################################################

class State:
    def __init__(self, board: Board):
        self.board = board

    # Possible actions sorted by utility in descending order
    # 0. Capture King
    # 1. Check King and capture another piece
    # 2. Capture Queen
    # 3. Capture Empress
    # 4. Capture Princess
    # 5. Capture Rook
    # 6. Capture Bishop
    # 7. Capture Knight
    # 8. Capture Ferz
    # 9. Capture Pawn
    # 10. Check King without capturing another piece
    # 11. No capture
    def actions(self, agent: int):
        actions = []
        position = [0 for _ in range(11)]   # based on utility
        for row in range(self.board.height):
            for col in range(self.board.width):
                if self.board.is_piece((col, row)):
                    piece = self.board.grid[row][col]
                    if piece.color == agent:
                        for x, y in piece.moves(self.board):
                            if self.board.is_piece((x, y)):
                                if self.board.grid[y][x].NAME == King.NAME:
                                    actions.insert(0, ((col, row), (x, y)))
                                    for i in range(1, 11):
                                        position[i] += 1
                                elif self.board.is_check(Piece.construct(piece.NAME, piece.color, (x, y))):
                                    actions.insert(position[1], ((col, row), (x, y)))
                                    for i in range(2, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Queen.NAME:
                                    actions.insert(position[2], ((col, row), (x, y)))
                                    for i in range(3, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Empress.NAME:
                                    actions.insert(position[3], ((col, row), (x, y)))
                                    for i in range(4, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Princess.NAME:
                                    actions.insert(position[4], ((col, row), (x, y)))
                                    for i in range(5, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Rook.NAME:
                                    actions.insert(position[5], ((col, row), (x, y)))
                                    for i in range(6, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Bishop.NAME:
                                    actions.insert(position[6], ((col, row), (x, y)))
                                    for i in range(7, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Knight.NAME:
                                    actions.insert(position[7], ((col, row), (x, y)))
                                    for i in range(8, 11):
                                        position[i] += 1
                                elif self.board.grid[y][x].NAME == Ferz.NAME:
                                    actions.insert(position[8], ((col, row), (x, y)))
                                    for i in range(9, 11):
                                        position[i] += 1
                                else:
                                    actions.insert(position[9], ((col, row), (x, y)))
                                    for i in range(10, 11):
                                        position[i] += 1
                            else:
                                if self.board.is_check(Piece.construct(piece.NAME, piece.color, (x, y))):
                                    actions.insert(position[10], ((col, row), (x, y)))
                                else:
                                    actions.append(((col, row), (x, y)))
        return actions

    def result(self, action: Tuple[Tuple[int, int], Tuple[int, int]]) -> State:
        grid = [row[:] for row in self.board.grid]
        piece = deepcopy(self.board.grid[action[0][1]][action[0][0]])
        piece.col, piece.row = action[1]
        grid[action[1][1]][action[1][0]] = piece
        grid[action[0][1]][action[0][0]] = None
        return State(Board(self.board.width, self.board.height, grid))

    def is_terminal(self):
        king_count = 0
        for row in range(self.board.height):
            for col in range(self.board.width):
                if self.board.is_piece((col, row)):
                    piece = self.board.grid[row][col]
                    if piece.NAME == King.NAME:
                        king_count += 1
        return king_count < 2

    def utility(self):
        score = 0
        for row in range(self.board.height):
            for col in range(self.board.width):
                if self.board.is_piece((col, row)):
                    piece = self.board.grid[row][col]
                    score += piece.VALUE * piece.color
        return score


#############################################################################
# Utility Functions
#############################################################################

def swap_tuple_elements(tup: Tuple[Any, Any]) -> Tuple[Any, Any]:
    return (tup[1], tup[0])


def from_chess_pos(position: Tuple[str, str]) -> Tuple[int, int]:
    col, row = position
    return (ord(col) - 97, int(row))


def to_chess_pos(position: Tuple[int, int]) -> Tuple[str, int]:
    col, row = position
    return (chr(col + 97), row)


#############################################################################
# Minimax with alpha-beta pruning
#############################################################################

MAX_AGENT = Pawn.WHITE
MIN_AGENT = Pawn.BLACK

memo: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = dict()

def ab(state: State, alpha: int, beta: int, agent: int, depth: int) -> Tuple[int, Tuple[Tuple[int, int], Tuple[int, int]]]:
    move = None
    if depth == 0 or state.is_terminal():
        return (state.utility(), move)

    if agent == MAX_AGENT:
        value = float("-inf")

        key = str(agent) + str(state.board)
        if key not in memo:
            memo[key] = state.actions(agent)

        actions = memo[key]
        # actions = state.actions(agent)
        if len(actions) == 0:
            return 0, None

        for action in actions:
            action_value, _ = ab(state.result(action), alpha, beta, MIN_AGENT, depth - 1)
            if action_value > value:
                value = action_value
                move = action
            if value >= beta:
                return (value, move)
            alpha = max(alpha, value)

        return (value, move)

    else:
        value = float("inf")

        key = str(agent) + str(state.board)
        if key not in memo:
            memo[key] = state.actions(agent)

        actions = memo[key]
        # actions = state.actions(agent)
        if len(actions) == 0:
            return 0, None

        for action in actions:
            action_value, _ = ab(state.result(action), alpha, beta, MAX_AGENT, depth - 1)
            if action_value < value:
                value = action_value
                move = action
            if value <= alpha:
                return (value, move)
            beta = min(beta, value)

        return (value, move)


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
    gameboard = {}

    enemy_piece_nums = get_par(handle.readline()).split()
    num_enemy_pieces = 0  # Read Enemy Pieces Positions
    for num in enemy_piece_nums:
        num_enemy_pieces += int(num)

    handle.readline()  # Ignore header
    for i in range(num_enemy_pieces):
        line = handle.readline()[1:-2]
        coords, piece = add_piece(line)
        gameboard[coords] = (piece, "Black")

    own_piece_nums = get_par(handle.readline()).split()
    num_own_pieces = 0  # Read Own Pieces Positions
    for num in own_piece_nums:
        num_own_pieces += int(num)

    handle.readline()  # Ignore header
    for i in range(num_own_pieces):
        line = handle.readline()[1:-2]
        coords, piece = add_piece(line)
        gameboard[coords] = (piece, "White")

    return rows, cols, gameboard


def add_piece(comma_seperated) -> Piece:
    piece, ch_coord = comma_seperated.split(",")
    r, c = from_chess_coord(ch_coord)
    return [(r, c), piece]


def from_chess_coord(ch_coord):
    return (int(ch_coord[1:]), ord(ch_coord[0]) - 97)


### DO NOT EDIT/REMOVE THE FUNCTION HEADER BELOW###
# Chess Pieces: King, Queen, Knight, Bishop, Rook, Princess, Empress, Ferz, Pawn (First letter capitalized)
# Colours: White, Black (First Letter capitalized)
# Positions: Tuple. (column (String format), row (Int)). Example: ('a', 0)

# Parameters:
# gameboard: Dictionary of positions (Key) to the tuple of piece type and its colour (Value). This represents the current pieces left on the board.
# Key: position is a tuple with the x-axis in String format and the y-axis in integer format.
# Value: tuple of piece type and piece colour with both values being in String format. Note that the first letter for both type and colour are capitalized as well.
# gameboard example: {('a', 0) : ('Queen', 'White'), ('d', 10) : ('Knight', 'Black'), ('g', 25) : ('Rook', 'White')}
#
# Return value:
# move: A tuple containing the starting position of the piece being moved to the new ending position for the piece. x-axis in String format and y-axis in integer format.
# move example: (('a', 0), ('b', 3))

def studentAgent(gameboard):
    # You can code in here but you cannot remove this function, change its parameter or change the return type
    state = State(Board.construct(7, 7, gameboard))
    _, move = ab(state, float("-inf"), float("inf"), MAX_AGENT, 3)
    move = (to_chess_pos(move[0]), to_chess_pos(move[1]))
    return move # Format to be returned (('a', 0), ('b', 3))
