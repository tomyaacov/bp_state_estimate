def one_to_many(state):
    return [int(state / 4), state % 4]


def many_to_one(state):
    return state[0] * 4 + state[1]
