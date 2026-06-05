\ Exact one-hot deterministic governed stochastic program
Minimize
 obj: 5439 z_0 + 10354 z_1 + 5749 z_2 + 10366.5 z_3 + 10354 z_4 + 5439 z_5 + 5749 z_6 + 10366.5 z_7 + 5749 z_8 + 5749 z_9 + 754 z_10 + 10761.5 z_11 + 10369 z_12 + 10369 z_13 + 10764 z_14 + 10384 z_15
Subject To
 one_sequence: z_0 + z_1 + z_2 + z_3 + z_4 + z_5 + z_6 + z_7 + z_8 + z_9 + z_10 + z_11 + z_12 + z_13 + z_14 + z_15 = 1
 budget_prob_catastrophic: 0.5 z_0 + 1 z_1 + 0.5 z_2 + 1 z_3 + 1 z_4 + 0.5 z_5 + 0.5 z_6 + 1 z_7 + 0.5 z_8 + 0.5 z_9 + 0 z_10 + 1 z_11 + 1 z_12 + 1 z_13 + 1 z_14 + 1 z_15 <= 0
 budget_prob_degraded: 0.5 z_0 + 0 z_1 + 1 z_2 + 0 z_3 + 0 z_4 + 0.5 z_5 + 1 z_6 + 0 z_7 + 1 z_8 + 1 z_9 + 1 z_10 + 1 z_11 + 0 z_12 + 0 z_13 + 1 z_14 + 0 z_15 <= 0
Binary
 z_0
 z_1
 z_2
 z_3
 z_4
 z_5
 z_6
 z_7
 z_8
 z_9
 z_10
 z_11
 z_12
 z_13
 z_14
 z_15
End
