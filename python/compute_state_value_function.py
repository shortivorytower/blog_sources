state_transitions = {
    0: {'l': 0, 'r': 1, 'u': 0, 'd': 4},
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
    15: {'l': 14, 'r': 15, 'u': 11, 'd': 15}
}


def transition_prob(new_state, reward, current_state, action):
    if reward != -1:
        raise Exception('Expecting all reward value to be -1')

    if current_state not in state_transitions:
        raise Exception('Unexpected current_state value {0}'.format(current_state))

    cur_state_dict = state_transitions[current_state]
    if action not in cur_state_dict:
        raise Exception('action {0} is not defined in the state {1}'.format(action, current_state))
    else:
        expected_next_state = cur_state_dict[action]
        if expected_next_state == new_state:
            return 1.0
        else:
            return 0.0


if __name__ == '__main__':

    # we don't discount any future reward
    gamma = 1.0

    # state from 1 to 14
    max_state = 15
    # mapping from state to value
    # i.e. the final result we want to compute.
    state_val_dict = {}
    # initialize all to zeros
    for i in range(0, max_state + 1):
        state_val_dict[i] = 0.0

    theta = 0.0000001
    # reward is always -1 for all actions (i.e. any moves)
    reward = -1
    k = 0
    while True:
        delta = 0.0
        new_state_val_dict = {}
        for state in range(0, max_state + 1):
            v = state_val_dict[state]
            new_v = 0.0
            # given ANY state, it is always four direction left, right, up, down even they are at the edge.
            # The probability of any direction is 1/4
            # Unless it is already at terminal state that the policy is doing nothing so action probability is zero.
            action_prob = 0.25 if 0 < state < max_state else 0.0

            for action in ['l', 'r', 'u', 'd']:
                exp_reward_for_action = 0.0
                for new_state in range(0, max_state + 1):
                    exp_reward_for_action += transition_prob(new_state, reward, state, action) * (reward + gamma * state_val_dict[new_state])

                new_v += action_prob * exp_reward_for_action

            # store the result in another new dictionary.
            new_state_val_dict[state] = new_v
            delta = max(delta, abs(v - new_v))

        # replace the result dictionary
        state_val_dict = new_state_val_dict

        k += 1
        if k % 100 == 0:
            print('iterated {0} times'.format(k))
        if delta < theta:
            break

    print('Iterative Policy Evaluation Result')
    for state in range(0, max_state + 1):
        print('state {0} value: {1:.2f}'.format(state, state_val_dict[state]))
