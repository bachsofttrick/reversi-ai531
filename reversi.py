from reversi_classes import  *
import numpy as np
import multiprocessing as mp
import argparse
import csv
import json

num_games = 5
debug = False
print_solution = False
show_progress = False
report_file = 'results.csv'
report_node_file = 'results_node.csv'
minimax_weight_file = 'weights.json'

def compare_algorithms(num_games=10, board_size=8, minimax_depth = 3, mcts_itereation = 10):
    """Compare Minimax with Alpha-Beta pruning against Monte Carlo Tree Search."""
    result = CompareResult(num_games*2, board_size, minimax_depth, mcts_itereation)

    # Count number of wins for players and draws
    minimax_wins = 0
    mcts_wins = 0
    draws = 0
    
    print(f"Playing {num_games*2} games of (board size, Minimax depth, MCTS iterations): ({board_size},{minimax_depth},{mcts_itereation})")
    
    for i in range(num_games):
        # Play with Minimax as black (player 1)
        print(f"\n({board_size},{minimax_depth},{mcts_itereation})", end=' ')
        print(f"Game {i*2+1}: Minimax(Black) vs MCTS(White)")
        minimax_player = MinimaxPlayer(1, minimax_depth)
        mcts_player = MCTSPlayer(2, mcts_itereation)
        game1 = ReversiGame(minimax_player, mcts_player, board_size, print_solution, show_progress)
        winner, _, _ = game1.play_game()
        
        # Debug info for game 1
        if debug:
            print(f"  Winner: {winner} ({'Minimax' if winner == 1 else 'MCTS' if winner == 2 else 'Draw'})")
            print(f"  Minimax - Time: {minimax_player.total_time:.3f}s, Moves: {minimax_player.move_number}, Avg: {minimax_player.average_time():.3f}s")
            print(f"  MCTS    - Time: {mcts_player.total_time:.3f}s, Moves: {mcts_player.move_number}, Avg: {mcts_player.average_time():.3f}s")

        # Increase win
        if winner == 1:
            minimax_wins += 1
        elif winner == 2:
            mcts_wins += 1
        else:
            draws += 1
        
        result.add_game_result(i, minimax_player, mcts_player, winner)

        # Play with MCTS as black (player 1)
        print(f"\n({board_size},{minimax_depth},{mcts_itereation})", end=' ')
        print(f"Game {i*2+2}: MCTS(Black) vs Minimax(White)")
        minimax_player = MinimaxPlayer(2, minimax_depth)
        mcts_player = MCTSPlayer(1, mcts_itereation)
        game2 = ReversiGame(mcts_player, minimax_player, board_size, print_solution, show_progress)
        winner, _, _ = game2.play_game()

        # Debug info for game 2
        if debug:
            print(f"  Winner: {winner} ({'MCTS' if winner == 1 else 'Minimax' if winner == 2 else 'Draw'})")
            print(f"  MCTS    - Time: {mcts_player.total_time:.3f}s, Moves: {mcts_player.move_number}, Avg: {mcts_player.average_time():.3f}s")
            print(f"  Minimax - Time: {minimax_player.total_time:.3f}s, Moves: {minimax_player.move_number}, Avg: {minimax_player.average_time():.3f}s")
        
        # Increase win
        if winner == 2:
            minimax_wins += 1
        elif winner == 1:
            mcts_wins += 1
        else:
            draws += 1
        
        result.add_game_result(i+1, mcts_player, minimax_player, winner)

    # Final summary
    print(f"\n({board_size},{minimax_depth},{mcts_itereation})")
    print(f"FINAL RESULTS after {num_games*2} games:")
    print(f"Minimax wins: {minimax_wins} ({minimax_wins/(num_games*2)*100:.1f}%)")
    print(f"MCTS wins: {mcts_wins} ({mcts_wins/(num_games*2)*100:.1f}%)")
    print(f"Draws: {draws} ({draws/(num_games*2)*100:.1f}%)\n")

    # Add win counts from Minimax and MCTS to result
    result.change_win_count(minimax_wins, 'minimax')
    result.change_win_count(mcts_wins, 'mcts')
    result.draws = draws

    return result

def run_experiments(multiprocess=False):
    """
    Test Minimax vs Monte Carlo across different board sizes and parameters.
    """
    board_sizes = [4, 6, 8, 10, 12, 14, 16]
    minimax_depths = [3, 4, 5, 6]
    monte_carlo_iterations = [10, 20, 50, 100, 200, 500, 1000]

    # List to store all results
    results_list: list[CompareResult] = []

    if multiprocess:
        # Use all available CPU cores
        num_processes = mp.cpu_count()
        # List to store all tasks
        tasks = []

        for m in minimax_depths:
            for n in monte_carlo_iterations:
                for b in board_sizes:
                    task = (num_games, b, m, n)
                    tasks.append(task)
        
        print(f"Running experiments using {num_processes} processes")
        print(f"Starting {len(tasks)} tasks...")

        # Create a pool of worker processes
        with mp.Pool(processes=num_processes) as pool:
            # Map tasks to the worker function
            results_list = pool.starmap(compare_algorithms, tasks)
    else:
        print("Running in single-process mode...")

        for m in minimax_depths:
            for n in monte_carlo_iterations:
                for b in board_sizes:
                    result = compare_algorithms(num_games, b, m, n)
                    results_list.append(result)

    return results_list

def save_to_csv(results: list[CompareResult]):
    """Save Reversi algorithm comparison results to CSV file."""
    with open(report_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = [
            "Board Size", "Total Games", "Draws", "Minimax Depth", "Minimax Wins", "Minimax Win Rate",
            "MCTS Iterations", "MCTS Wins", "MCTS Win Rate",
            
            "Game #", "Winner", "Minimax Player", "MCTS Player",

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
                    game.number, game.winner, game.minimax_player, game.mcts_player,

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

def import_weights_json():
    """
    Load board weights from a JSON file and convert each board size entry into a NumPy array.
    
    Returns:
        dict[int, np.ndarray]: Dictionary mapping board size to its weight matrix as a NumPy array.
    """
    with open(minimax_weight_file, "r") as f:
        data = json.load(f)

    weights = {int(size): np.array(matrix) for size, matrix in data.items()}
    MinimaxPlayer.WEIGHTS = weights

def main():
    # Access global variable
    global debug, print_solution, show_progress

    parser = argparse.ArgumentParser(description="Reversi: MInimax vs Monte Carlo")
    parser.add_argument('--multi', action='store_true', help="Run in multiprocessing mode")
    parser.add_argument('--print-solution', action='store_true', help="Print the solved board")
    parser.add_argument('--show-progress', action='store_true', help="Print the percent completed of the game")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")

    args = parser.parse_args()
    # Change variables if they are offered in arguments
    if args.debug != debug: debug = args.debug
    if args.show_progress != show_progress: show_progress = args.show_progress
    if args.print_solution != print_solution: print_solution = args.print_solution

    import_weights_json()
    results = run_experiments(args.multi)
    save_to_csv(results)

# Main program
if __name__ == "__main__":
    main()
