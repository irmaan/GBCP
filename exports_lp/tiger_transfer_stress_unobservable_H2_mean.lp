\ Exact one-hot deterministic governed stochastic program
Minimize
 obj: 120 z_0 + 550 z_1 + 550 z_2 + 120 z_3 + 550 z_4 + 540 z_5 + 1000 z_6 + 550 z_7 + 550 z_8 + 1000 z_9 + 540 z_10 + 550 z_11 + 120 z_12 + 550 z_13 + 550 z_14 + 120 z_15
Subject To
 one_sequence: z_0 + z_1 + z_2 + z_3 + z_4 + z_5 + z_6 + z_7 + z_8 + z_9 + z_10 + z_11 + z_12 + z_13 + z_14 + z_15 = 1
 budget_prob_catastrophic: 0 z_0 + 0.5 z_1 + 0.5 z_2 + 0 z_3 + 0.5 z_4 + 0.5 z_5 + 1 z_6 + 0.5 z_7 + 0.5 z_8 + 1 z_9 + 0.5 z_10 + 0.5 z_11 + 0 z_12 + 0.5 z_13 + 0.5 z_14 + 0 z_15 <= 0
 budget_prob_degraded: 1 z_0 + 1 z_1 + 1 z_2 + 1 z_3 + 1 z_4 + 0.5 z_5 + 0 z_6 + 1 z_7 + 1 z_8 + 0 z_9 + 0.5 z_10 + 1 z_11 + 1 z_12 + 1 z_13 + 1 z_14 + 1 z_15 <= 0
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
