# Play Reversi against AI
AI has two modes:
- Minimax with Alpha-Beta Pruning
- Monte Carlo Tree Search

This program is used to compare between two algorithms.
The board has multiple sizes: 4,6,8,10,12,14,16.  
We play 10 games each, 5 with black being Minimax, 5 with black being Monte Carlo.  
Minimax change the depth of tree: 3,4,5,6,7,8.  
Monte Carlo change the iteration count: 10, 20, 50, 100, 200, 500.  
We compare:
- Total move, total time spent thinking for each algorithm
- Average time spent per move for each algorithm
- Which move costs the most time for each algorithm
- Win rate
