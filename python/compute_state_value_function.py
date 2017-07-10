state_prob_dict = {
    1: {'l': 0, 'r': 2, 'u': 1, 'd': 5},
    2: {'l': 1, 'r': 3, 'u': 2, 'd': 6},
    3: {'l': 2, 'r': 3, 'u': 3, 'd': 7},
    4: {'l': 4, 'r': 5, 'u': 0, 'd': 8},
    5: {'l': 4, 'r': 6, 'u': 1, 'd': 9},
    6: {'l': 5, 'r': 7, 'u': 2, 'd': 10},
    7: {'l': 6, 'r': 7, 'u': 3, 'd': 11},
    8: {'l': 8, 'r': 9, 'u': 4, 'd': 12},
    9: {'l': 8, 'r': 10, 'u': 5, 'd': 13},
    10: {'l': 9, 'r': 11, 'u': 6, 'd': 14},
    11: {'l': 10, 'r': 11, 'u': 7, 'd': 15},
    12: {'l': 12, 'r': 13, 'u': 8, 'd': 12},
    13: {'l': 12, 'r': 14, 'u': 9, 'd': 13},
    14: {'l': 13, 'r': 15, 'u': 10, 'd': 14},
}


def prob(new_state, reward, current_state, action):
    # terminate state
    if current_state == 1 and action=='l' and new_state == 0 and reward == 0:
        return 1.0
    if current_state == 4 and action=='u' and new_state == 0 and reward == 0:
        return 1.0
    if current_state == 11 and action=='d' and new_state == 15 and reward == 0:
        return 1.0
    if current_state == 14 and action=='r' and new_state == 15 and reward == 0:
        return 1.0

    if reward != -1:
        return 0.0

    if not (current_state in state_prob_dict):
        return 0.0

    cur_state_dict = state_prob_dict[current_state]
    if not (action in cur_state_dict):
        return 0.0
    else:
        expected_next = cur_state_dict[action]
        if expected_next == new_state:
            return 1.0
        else:
            return 0.0


if __name__ == '__main__':

    # given ANY state, it is always four direction left, right, up, down even they are at the edge.
    pi_val = 1.0/4.0
    gamma = 1.0

    # state from 1 to 14
    max_state = 15
    # mapping from state to value
    V_dict = {}
    for i in range(1, max_state):
        V_dict[i] = 0.0

    epsilon = 0.00000001
    iteration_count = 0
    while True:
        delta = 0.0
        for s in range(1, max_state):
            v = V_dict[s]
            vs = 0.0
            for a in ['l', 'r', 'u', 'd']:
                temp = 0.0
                for new_state in range(1, max_state):
                    reward = -1
                    temp = temp + prob(new_state, reward, s, a) * (reward + gamma * V_dict[new_state])
                vs = vs + pi_val * temp
            V_dict[s] = vs

            delta = max(delta, abs(v - vs))

        iteration_count += 1
        print 'iterated', iteration_count
        if delta < epsilon:
            break

    print 'Finished'
    for s in range(1, max_state):
        print 'state', s, ':', V_dict[s]
