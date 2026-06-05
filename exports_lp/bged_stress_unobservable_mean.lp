\ Exact one-hot deterministic governed stochastic program
Minimize
 obj: 8234 z_0 + 11134 z_1 + 16339 z_2 + 11146.5 z_3 + 11134 z_4 + 8234 z_5 + 16339 z_6 + 11146.5 z_7 + 16339 z_8 + 16339 z_9 + 16344 z_10 + 16351.5 z_11 + 11149 z_12 + 11149 z_13 + 16354 z_14 + 11164 z_15
Subject To
 one_sequence: z_0 + z_1 + z_2 + z_3 + z_4 + z_5 + z_6 + z_7 + z_8 + z_9 + z_10 + z_11 + z_12 + z_13 + z_14 + z_15 = 1
 budget_prob_catastrophic: 0.5 z_0 + 1 z_1 + 1 z_2 + 1 z_3 + 1 z_4 + 0.5 z_5 + 1 z_6 + 1 z_7 + 1 z_8 + 1 z_9 + 1 z_10 + 1 z_11 + 1 z_12 + 1 z_13 + 1 z_14 + 1 z_15 <= 0
 budget_prob_offtarget: 0.5 z_0 + 0 z_1 + 1 z_2 + 0 z_3 + 0 z_4 + 0.5 z_5 + 1 z_6 + 0 z_7 + 1 z_8 + 1 z_9 + 1 z_10 + 1 z_11 + 0 z_12 + 0 z_13 + 1 z_14 + 0 z_15 <= 0
 budget_prob_immune: 0.5 z_0 + 0 z_1 + 1 z_2 + 0 z_3 + 0 z_4 + 0.5 z_5 + 1 z_6 + 0 z_7 + 1 z_8 + 1 z_9 + 1 z_10 + 1 z_11 + 0 z_12 + 0 z_13 + 1 z_14 + 0 z_15 <= 0
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
