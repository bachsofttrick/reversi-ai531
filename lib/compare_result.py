from lib.reversi_classes import  *

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
        
        def __init__(self, number, player1: ReversiPlayer, player2: ReversiPlayer, winner, playing_too_long):
            check_player1_is_minimax = player1.is_using_minimax()

            self.number = number
            self.minimax_player = 1 if check_player1_is_minimax else 2
            self.mcts_player = 2 if check_player1_is_minimax else 1
            self.winner = winner
            self.playing_too_long = playing_too_long
            self.minimax = CompareResult.GameResult.PlayerStatistic(player1 if check_player1_is_minimax else player2)
            self.mcts = CompareResult.GameResult.PlayerStatistic(player2 if check_player1_is_minimax else player1)

    def __init__(self, num_games=10, board_size=8, minimax_depth = 3, mcts_itereation = 10):
        self.num_games = num_games
        self.board_size = board_size
        self.draws = 0
        self.minimax = CompareResult.AlgorithmStatistic(minimax_depth)
        self.mcts = CompareResult.AlgorithmStatistic(mcts_itereation)
        self.game: list[CompareResult.GameResult] = []
    
    def add_game_result(self, number, player1: ReversiPlayer, player2: ReversiPlayer, winner, playing_too_long):
        game_result = CompareResult.GameResult(number, player1, player2, winner, playing_too_long)
        self.game.append(game_result)
    
    def change_win_count(self, win, player_type = 'minimax'):
        if player_type == 'minimax':
            player: CompareResult.AlgorithmStatistic = self.minimax
        else:
            player: CompareResult.AlgorithmStatistic = self.mcts
        
        player.win = win
        player.win_rate = win / self.num_games
