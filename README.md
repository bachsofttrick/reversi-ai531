# Play Reversi against AI
AI has two modes:
- Minimax with Alpha-Beta Pruning
- Monte Carlo Tree Search
This program can be used to compare between two algorithms, or you can play against either of them.
The board has multiple sizes: 4,6,8,10,12,14,16.
We play 10 games each, 5 with black being Minimax, 5 with black being Monte Carlo
Minimax change the depth of tree: 3,4,5,6,7,8
Monte Carlo change the iteration count: 10, 20, 50, 100, 200, 500
For both minimax and mcts:
- Total move, total time spent thinking for each algorithm,
- Average time spent per moves for each algorithm,
- Which moves cost the most time for each algorithm,
- Win rate
