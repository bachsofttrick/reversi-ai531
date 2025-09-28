# Reversi: Minimax vs Monte Carlo
### By Mykyta Synytsia, Phan Xuan Bach

* [Paper](https://github.com/bachsofttrick/reversi-ai531/blob/master/paper.pdf)

This program is used to compare between two algorithms.
- Minimax with Alpha-Beta Pruning
- Monte Carlo Tree Search

The board has multiple sizes: 4,8,12.  
We play 6 games each, 3 with black being Minimax, 3 with black being Monte Carlo.  
Minimax change the depth of tree: 3,4,5,6.  
Monte Carlo change the iteration count: 10, 100, 1000, 10000.
We compare:
- Total move, total time spent thinking for each algorithm
- Average time spent per move for each algorithm
- Which move costs the most time for each algorithm
- Win rate
