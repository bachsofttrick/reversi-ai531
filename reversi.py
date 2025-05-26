from lib.save_to_csv import *
from lib.reversi_classes import  *
from lib.compare_result import  *
import multiprocessing as mp
import argparse

num_games = 5
debug = False
print_solution = False
show_progress = False
stop_early=True
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
        winner, playing_too_long = game1.play_game(stop_early)
        
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
        
        result.add_game_result(i*2+1, minimax_player, mcts_player, winner, playing_too_long)

        # Play with MCTS as black (player 1)
        print(f"\n({board_size},{minimax_depth},{mcts_itereation})", end=' ')
        print(f"Game {i*2+2}: MCTS(Black) vs Minimax(White)")
        minimax_player = MinimaxPlayer(2, minimax_depth)
        mcts_player = MCTSPlayer(1, mcts_itereation)
        game2 = ReversiGame(mcts_player, minimax_player, board_size, print_solution, show_progress)
        winner, playing_too_long = game2.play_game(stop_early)

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
        
        result.add_game_result(i*2+2, mcts_player, minimax_player, winner, playing_too_long)

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
    board_sizes = [4]
    minimax_depths = [3]
    monte_carlo_iterations = [10]

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

    MinimaxPlayer.import_weights_json(minimax_weight_file)
    results = run_experiments(args.multi)
    save_to_csv(results, report_file, report_node_file)

# Main program
if __name__ == "__main__":
    main()
