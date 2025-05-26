from lib.compare_result import  *
import csv

def save_to_csv(results: list[CompareResult], report_file, report_node_file):
    """Save Reversi algorithm comparison results to CSV file."""
    with open(report_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = [
            "Board Size", "Total Games", "Draws", "Minimax Depth", "Minimax Wins", "Minimax Win Rate",
            "MCTS Iterations", "MCTS Wins", "MCTS Win Rate",
            
            "Game #", "Winner", "Minimax Player", "MCTS Player", "Stop Early",

            "Minimax Total Time", "Minimax Moves", "Minimax Avg Time", 
            "Minimax Max Time Per Move", "Minimax Max Time Move",

            "MCTS Total Time", "MCTS Moves", "MCTS Avg Time",
            "MCTS Max Time Per Move", "MCTS Max Time Move"
        ]
        writer.writerow(header)
        
        # Write data for each CompareResult
        for result in results:
            # Overall statistics for each row
            overall_stats = [
                result.board_size, result.num_games, result.draws,
                result.minimax.depth, result.minimax.win, result.minimax.win_rate,
                result.mcts.depth, result.mcts.win, result.mcts.win_rate
            ]
            # Write individual game results
            for game in result.game:
                # Individual game data
                game_data = [
                    game.number, game.winner, game.minimax_player, game.mcts_player, game.playing_too_long,

                    round(game.minimax.total_time, 6), game.minimax.move_number, round(game.minimax.average_time, 6),
                    round(game.minimax.max_time_to_make_move, 6), game.minimax.move_with_max_time,

                    round(game.mcts.total_time, 6), game.mcts.move_number, round(game.mcts.average_time, 6),
                    round(game.mcts.max_time_to_make_move, 6), game.mcts.move_with_max_time
                ]
                
                # Combine arrays using unpacking
                row = [*overall_stats, *game_data]
                writer.writerow(row)
            
            # Add empty row for separation between different result sets
            writer.writerow([])

    """Save Reversi algorithm node comparison results to CSV file."""
    with open(report_node_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = [
            "Board Size", "Minimax Depth",
            "MCTS Iterations",
            
            "Game #",

            "Minimax Total Nodes Created", "Minimax Max Nodes Created in a Move",
            "Minimax Max Nodes Created Move", "Minimax Avg Nodes Created",
            "Minimax Total Nodes Explored", "Minimax Max Nodes Explored in a Move",
            "Minimax Max Nodes Explored Move", "Minimax Avg Nodes Explored",

            "MCTS Total Nodes Created", "MCTS Max Nodes Created in a Move",
            "MCTS Max Nodes Created Move", "MCTS Avg Nodes Created",
            "MCTS Total Nodes Explored", "MCTS Max Nodes Explored in a Move",
            "MCTS Max Nodes Explored Move", "MCTS Avg Nodes Explored",
        ]
        writer.writerow(header)
        
        # Write data for each CompareResult
        for result in results:
            # Overall statistics for each row
            overall_stats = [
                result.board_size,
                result.minimax.depth,
                result.mcts.depth
            ]
            # Write individual game results
            for game in result.game:
                # Individual game data
                game_data = [
                    game.number,

                    game.minimax.total_nodes_created, game.minimax.max_nodes_created_in_a_move,
                    game.minimax.move_with_max_nodes_created, game.minimax.average_nodes_created,
                    game.minimax.total_nodes_explored, game.minimax.max_nodes_explored_in_a_move,
                    game.minimax.move_with_max_nodes_explored, game.minimax.average_nodes_explored,

                    game.mcts.total_nodes_created, game.mcts.max_nodes_created_in_a_move,
                    game.mcts.move_with_max_nodes_created, game.mcts.average_nodes_created,
                    game.mcts.total_nodes_explored, game.mcts.max_nodes_explored_in_a_move,
                    game.mcts.move_with_max_nodes_explored, game.mcts.average_nodes_explored
                ]
                
                # Combine arrays using unpacking
                row = [*overall_stats, *game_data]
                writer.writerow(row)
            
            # Add empty row for separation between different result sets
            writer.writerow([])