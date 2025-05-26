import numpy as np
import random
import time
import math
from copy import deepcopy

class ReversiBoard:
    """
    A class representing the Reversi game board and rules.
    """
    # Direction vectors for checking in all 8 directions
    DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1 for black, 2 for white
        
        # Initialize the starting position
        mid = size // 2
        self.board[mid-1][mid-1] = 2  # White
        self.board[mid][mid] = 2      # White
        self.board[mid-1][mid] = 1    # Black
        self.board[mid][mid-1] = 1    # Black

    def get_opponent(self, player):
        """Return the opponent's number."""
        return 3 - player  # 1 -> 2, 2 -> 1
    
    def is_on_board(self, row, col):
        """Check if a position is on the board."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def count_pieces(self):
        """Count the number of pieces for each player."""
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == 2)
        return black_count, white_count
    
    def get_valid_moves(self, player=None):
        """Get all valid moves for the given player."""
        if player is None:
            player = self.current_player
        
        valid_moves = []
        
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        
        return valid_moves
    
    def is_valid_move(self, row, col, player=None):
        """Check if a move is valid for the given player."""
        if player is None:
            player = self.current_player
            
        # Cell must be empty
        if self.board[row][col] != 0:
            return False
        
        opponent = self.get_opponent(player)
        
        # Check in all eight directions
        for dx, dy in self.DIRECTIONS:
            r, c = row + dx, col + dy
            
            # There must be at least one opponent piece in this direction
            if not (self.is_on_board(r, c) and self.board[r][c] == opponent):
                continue
                
            # Continue in this direction
            r += dx
            c += dy
            
            # Keep going until we find an empty cell or go off the board
            while self.is_on_board(r, c) and self.board[r][c] != 0:
                # If we find our own piece, this is a valid move
                if self.board[r][c] == player:
                    return True
                    
                r += dx
                c += dy
        
        return False
    
    def make_move(self, row, col, player=None):
        """Make a move for the given player."""
        if player is None:
            player = self.current_player
            
        if not self.is_valid_move(row, col, player):
            return False
        
        # Place the piece
        self.board[row][col] = player
        
        # Flip opponent pieces
        opponent = self.get_opponent(player)
        pieces_to_flip = []
        
        for dx, dy in self.DIRECTIONS:
            r, c = row + dx, col + dy
            temp_flip = []
            
            # Check if there are opponent pieces to flip in this direction
            while self.is_on_board(r, c) and self.board[r][c] == opponent:
                temp_flip.append((r, c))
                r += dx
                c += dy
                
            # If we reached one of our own pieces, flip all pieces in between
            if self.is_on_board(r, c) and self.board[r][c] == player:
                pieces_to_flip.extend(temp_flip)
        
        # Flip all captured pieces
        for r, c in pieces_to_flip:
            self.board[r][c] = player
            
        # Switch players for next turn
        self.current_player = self.get_opponent(player)
        
        return True
    
    def has_valid_moves(self, player=None):
        """Check if the player has any valid moves."""
        if player is None:
            player = self.current_player
            
        return len(self.get_valid_moves(player)) > 0
    
    def is_game_over(self):
        """Check if the game is over."""
        return not (self.has_valid_moves(1) or self.has_valid_moves(2))
    
    def get_winner(self):
        """Get the winner of the game."""
        black_count, white_count = self.count_pieces()
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return 2
        else:
            return 0  # Draw
    
    def copy(self):
        """Create a deep copy of the board."""
        new_board = ReversiBoard(self.size)
        new_board.board = deepcopy(self.board)
        new_board.current_player = self.current_player
        return new_board
    
    def print_board(self, highlighted_pos=None):
        """
        Print the current board state with a nicer display.
        
        Args:
            highlighted_pos: Optional (row, col) tuple to highlight a position
        """
        # Get valid moves for current player to mark with asterisk
        valid_moves = self.get_valid_moves()
        
        # Print column headers (A through H)
        col_headers = "    " + "   ".join([chr(65 + i) for i in range(self.size)])
        print(col_headers)
        
        # Print the top border
        print("  +" + "---+" * self.size)
        
        # Print each row
        for i in range(self.size):
            row_str = f"{i+1} |"
            for j in range(self.size):
                # Determine cell content
                if self.board[i][j] == 1:
                    cell = "1"  # Black
                elif self.board[i][j] == 2:
                    cell = "0"  # White
                else:
                    cell = " "  # Empty
                
                # Highlight the cell if it's the selected position
                if highlighted_pos and highlighted_pos[0] == i and highlighted_pos[1] == j:
                    row_str += f"\033[7m {cell} \033[0m|"  # Inverse video for highlighting
                else:
                    row_str += f" {cell} |"
                    
            print(row_str)
            print("  +" + "---+" * self.size)
        
        black_count, white_count = self.count_pieces()
        print(f"1: Black: {black_count}  |  0: White: {white_count}")

class ReversiPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.move_number = 0
        self.total_time = 0
        self.max_time_to_make_move = 0
        self.move_with_max_time = 0
        
        # Node counting
        self.total_nodes_created = 0
        self.total_nodes_explored = 0
        self.nodes_created_per_move = []
        self.nodes_explored_per_move = []
        self.max_nodes_created_in_a_move = 0
        self.move_with_max_nodes_created = 0
        self.max_nodes_explored_in_a_move = 0
        self.move_with_max_nodes_explored = 0
    
    def get_move(self, board: ReversiBoard):
        pass

    def get_move_timed(self, board: ReversiBoard):
        nodes_created_before = self.total_nodes_created
        nodes_explored_before = self.total_nodes_explored
        
        start_time = time.time()
        move = self.get_move(board)
        total_time = time.time() - start_time

        nodes_created_this_move = self.total_nodes_created - nodes_created_before
        nodes_explored_this_move = self.total_nodes_explored - nodes_explored_before
        
        self.total_time += total_time
        self.move_number += 1
        self.nodes_created_per_move.append(nodes_created_this_move)
        self.nodes_explored_per_move.append(nodes_explored_this_move)
        
        if total_time > self.max_time_to_make_move:
            self.max_time_to_make_move = total_time
            self.move_with_max_time = self.move_number
            
        if nodes_created_this_move > self.max_nodes_created_in_a_move:
            self.max_nodes_created_in_a_move = nodes_created_this_move
            self.move_with_max_nodes_created = self.move_number
        
        if nodes_explored_this_move > self.max_nodes_explored_in_a_move:
            self.max_nodes_explored_in_a_move = nodes_explored_this_move
            self.move_with_max_nodes_explored = self.move_number
            
        return move
    
    def average_time(self):
        return self.total_time / self.move_number if self.move_number > 0 else 0
    
    def average_created_nodes_per_move(self):
        return self.total_nodes_explored / self.move_number if self.move_number > 0 else 0
    
    def average_explored_nodes_per_move(self):
        return self.total_nodes_explored / self.move_number if self.move_number > 0 else 0

class MinimaxPlayer(ReversiPlayer):
    """
    AI player using Minimax algorithm with Alpha-Beta pruning.
    """
    # Weights available to all instace of MinimaxPlayer
    WEIGHTS: dict[int, np.ndarray]

    def __init__(self, player_number, depth=4):
        super().__init__(player_number)
        self.depth = depth
        
        # Node counting for current move
        self.nodes_created = 0
        self.nodes_explored = 0
    
    def get_move(self, board: ReversiBoard):
        """Get the best move using Minimax with Alpha-Beta pruning."""
        # Get the appropriate weights for the board size
        # Weights for the board evaluation
        self.weights = MinimaxPlayer.WEIGHTS[board.size]

        # Reset node counter for this move
        self.nodes_created = 0
        self.nodes_explored = 0
        
        valid_moves = board.get_valid_moves(self.player_number)
        
        if not valid_moves:
            return None
        
        # Count the root node
        self.nodes_created += 1
        self.nodes_explored += 1

        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            self.total_nodes_created += self.nodes_created
            self.total_nodes_explored += self.nodes_explored
            return valid_moves[0]
        
        best_value = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
            # Make a copy of the board and make the move
            board_copy = board.copy()
            # Count the board copy as a node
            self.nodes_created += 1
            board_copy.make_move(move[0], move[1], self.player_number)
            
            # Get the value of this move
            value = self.minimax(board_copy, self.depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, best_value)
        
        # Update total nodes
        self.total_nodes_created += self.nodes_created
        self.total_nodes_explored += self.nodes_explored
        
        return best_move
    
    def minimax(self, board: ReversiBoard, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with Alpha-Beta pruning.
        
        Args:
            board: Current board state
            depth: How many more levels to search
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: Whether this is a maximizing level
            
        Returns:
            The score for the current board state
        """
          # Count each node exploration
        self.nodes_explored += 1
        
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        current_player = self.player_number if is_maximizing else 3 - self.player_number
        valid_moves = board.get_valid_moves(current_player)
        
        if not valid_moves:
            # If the current player has no valid moves, pass to the opponent
            board_copy = board.copy()
            board_copy.current_player = 3 - current_player
            return self.minimax(board_copy, depth - 1, alpha, beta, not is_maximizing)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                board_copy = board.copy()
                # Count the board copy as a node
                self.nodes_created += 1
                board_copy.make_move(move[0], move[1], current_player)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff - nodes after this aren't explored
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                board_copy = board.copy()
                # Count the board copy as a node
                self.nodes_created += 1
                board_copy.make_move(move[0], move[1], current_player)
                eval = self.minimax(board_copy, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff - nodes after this aren't explored
            return min_eval
    
    def evaluate(self, board: ReversiBoard):
        """
        Evaluate the current board state from the perspective of this player.
        Higher scores are better for this player.
        """
        # If the game is over, return a high value if we won, low if we lost
        if board.is_game_over():
            winner = board.get_winner()
            if winner == self.player_number:
                return 10000  # We won
            elif winner == 0:
                return 0      # Draw
            else:
                return -10000  # We lost
        
        # Use a weighted matrix for position evaluation
        my_pieces = np.zeros((board.size, board.size), dtype=bool)
        opponent_pieces = np.zeros((board.size, board.size), dtype=bool)
        
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == self.player_number:
                    my_pieces[i][j] = True
                elif board.board[i][j] == 3 - self.player_number:
                    opponent_pieces[i][j] = True
        
        # Calculate positional score
        my_score = np.sum(self.weights[my_pieces])
        opponent_score = np.sum(self.weights[opponent_pieces])
        
        # Calculate mobility score (number of valid moves)
        my_mobility = len(board.get_valid_moves(self.player_number))
        opponent_mobility = len(board.get_valid_moves(3 - self.player_number))
        
        # Combine scores with different weights
        positional_weight = 10
        mobility_weight = 5
        
        total_score = (positional_weight * (my_score - opponent_score) + 
                      mobility_weight * (my_mobility - opponent_mobility))
        
        return total_score

class MCTSPlayer(ReversiPlayer):
    class MonteCarloNode:
        """A node in the Monte Carlo Search Tree."""
        def __init__(self, board: ReversiBoard, parent=None, move=None):
            self.board = board
            self.parent = parent
            self.move = move  # The move that led to this board state
            self.children: list[MCTSPlayer.MonteCarloNode] = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = board.get_valid_moves()
        
        def select_child(self):
            """Select a child node using UCB1 formula."""
            # UCB1 formula: wins/visits + C * sqrt(ln(parent visits) / visits)
            C = 1.41  # Exploration parameter
            
            best_score = float('-inf')
            best_child = None
            
            for child in self.children:
                # Avoid division by zero
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploitation = child.wins / child.visits
                    exploration = C * math.sqrt(math.log(self.visits) / child.visits)
                    score = exploitation + exploration
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            return best_child
        
        def add_child(self, move):
            """Add a child node with the given move."""
            board_copy = self.board.copy()
            board_copy.make_move(move[0], move[1])
            
            child = MCTSPlayer.MonteCarloNode(board_copy, parent=self, move=move)
            self.untried_moves.remove(move)
            self.children.append(child)
            
            return child
        
        def update(self, result):
            """Update this node with a simulation result."""
            self.visits += 1
            self.wins += result
            
    """
    AI player using Monte Carlo Tree Search.
    """
    def __init__(self, player_number, iterations=1000):
        super().__init__(player_number)
        self.iterations = iterations
        
        # Node counting for current move
        self.nodes_explored = 0
        self.tree_nodes_created = 0  # Nodes actually added to the tree
        self.simulation_nodes = 0     # Nodes explored during simulations
    
    def get_move(self, board: ReversiBoard):
        """Get the best move using Monte Carlo Tree Search."""
        # Reset counters for this move
        self.nodes_explored = 0
        self.tree_nodes_created = 0
        self.simulation_nodes = 0
        
        valid_moves = board.get_valid_moves(self.player_number)
        
        if not valid_moves:
            return None
        
        # Count the root node
        self.nodes_explored = 1
        
        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            self.nodes_explored = 1
            self.total_nodes_explored += self.nodes_explored
            return valid_moves[0]
        
        # Create the root node
        board_copy = board.copy()
        # Make sure it's our turn
        board_copy.current_player = self.player_number
        root = MCTSPlayer.MonteCarloNode(board_copy)
        self.tree_nodes_created += 1  # Count the root node
        
        # Run MCTS iterations
        for _ in range(self.iterations):
            # Selection: Select a node to expand
            node = root
            path_length = 0
            
            while node.untried_moves == [] and node.children:
                node = node.select_child()
                path_length += 1
            
            # Expansion: Add a new child node
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)
                self.tree_nodes_created += 1
                path_length += 1
            
            # Simulation: Play a random game from this node
            board_copy = node.board.copy()
            simulation_moves = 0
            
            while not board_copy.is_game_over():
                valid_moves = board_copy.get_valid_moves()
                if not valid_moves:
                    # If no valid moves, switch players
                    board_copy.current_player = 3 - board_copy.current_player
                    continue
                
                random_move = random.choice(valid_moves)
                board_copy.make_move(random_move[0], random_move[1])
                simulation_moves += 1
            
            self.simulation_nodes += simulation_moves
            
            # Backpropagation: Update all nodes in the path
            winner = board_copy.get_winner()
            backprop_nodes = 0
            
            while node:
                if winner == self.player_number:
                    result = 1.0  # We won
                elif winner == 0:
                    result = 0.5  # Draw
                else:
                    result = 0.0  # We lost
                
                node.update(result)
                node = node.parent
                backprop_nodes += 1
        
        # Count total nodes explored this move
        # Tree nodes + simulation nodes (simulation nodes already counted above)
        self.nodes_explored = self.tree_nodes_created + self.simulation_nodes
        self.total_nodes_explored += self.nodes_explored
        
        # Choose the move with the highest number of visits
        best_visits = -1
        best_move = None
        
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move
        
        return best_move

class ReversiGame:
    def __init__(self, player1: ReversiPlayer, player2: ReversiPlayer, board_size=8, print_board=False):
        self.board = ReversiBoard(board_size)
        self.players = {1: player1, 2: player2}
        self.print_board = print_board
        self.winner = 0
        self.black_count = 0
        self.white_count = 0

    def play_game(self):
        """Play a game between two AI players."""
        while not self.board.is_game_over():
            if self.board.has_valid_moves():
                player = self.players[self.board.current_player]
                move = player.get_move_timed(self.board)
                self.board.make_move(move[0], move[1])
            else:
                # No valid moves, switch players
                self.board.current_player = 3 - self.board.current_player
        
        self.winner = self.board.get_winner()
        self.black_count, self.white_count = self.board.count_pieces()
        
        if self.print_board:
            print("Game over!")
            self.board.print_board()
            if self.winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {self.winner} wins!")
        
        return self.winner, self.black_count, self.white_count

class CompareResult:
    """Result of the Reversi game to print to csv."""

    class AlgorithmStatistic:
        """Statistic for each algorithm player."""
        def __init__(self, depth):
            self.depth = depth
            self.win = 0
            self.win_rate = 0

    class GameResult:
        """Statistic for each game and its accompanying players."""

        class PlayerStatistic:
            def __init__(self, player: ReversiPlayer):
                self.total_time = player.total_time
                self.move_number = player.move_number
                self.average_time = player.average_time()
                self.max_time_to_make_move = player.max_time_to_make_move
                self.move_with_max_time = player.move_with_max_time

                # Node counting
                self.total_nodes_created = player.total_nodes_created
                self.total_nodes_explored = player.total_nodes_explored
                self.max_nodes_created_in_a_move = player.max_nodes_created_in_a_move
                self.move_with_max_nodes_created = player.move_with_max_nodes_created
                self.max_nodes_explored_in_a_move = player.max_nodes_explored_in_a_move
                self.move_with_max_nodes_explored = player.move_with_max_nodes_explored
                self.average_nodes_created = player.average_created_nodes_per_move()
                self.average_nodes_explored = player.average_explored_nodes_per_move()
        
        def __init__(self, number, player1: ReversiPlayer, player2: ReversiPlayer, winner):
            check_player1_is_minimax = player1.__class__.__name__ == MinimaxPlayer.__name__

            self.number = number
            self.minimax_player = 1 if check_player1_is_minimax else 2
            self.mcts_player = 2 if check_player1_is_minimax else 1
            self.winner = winner
            self.minimax = CompareResult.GameResult.PlayerStatistic(player1 if check_player1_is_minimax else player2)
            self.mcts = CompareResult.GameResult.PlayerStatistic(player2 if check_player1_is_minimax else player1)

    def __init__(self, num_games=10, board_size=8, minimax_depth = 3, mcts_itereation = 10):
        self.num_games = num_games
        self.board_size = board_size
        self.draws = 0
        self.minimax = CompareResult.AlgorithmStatistic(minimax_depth)
        self.mcts = CompareResult.AlgorithmStatistic(mcts_itereation)
        self.game: list[CompareResult.GameResult] = []
    
    def add_game_result(self, number, player1: ReversiPlayer, player2: ReversiPlayer, winner):
        game_result = CompareResult.GameResult(number, player1, player2, winner)
        self.game.append(game_result)
    
    def change_win_count(self, win, player_type = 'minimax'):
        if player_type == 'minimax':
            player: CompareResult.AlgorithmStatistic = self.minimax
        else:
            player: CompareResult.AlgorithmStatistic = self.mcts
        
        player.win = win
        player.win_rate = win / self.num_games