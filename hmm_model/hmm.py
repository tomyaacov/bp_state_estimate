from hmmlearn.hmm import MultinomialHMM
from numpy import genfromtxt
import numpy as np


def build_hmm(trans, emis, seed=1):
    """Builds and returns hmm_model given the transition and emission probabilities.

    Parameters
    ----------
    trans : array-like
        Parameters of distribution for each row of the transition probabilities.

    emis : array-like
        Probability of emitting a given symbol when in each state.

    Returns
    -------
    hmm_model : MultinomialHMM
        Resulting hmm_model.
        """
    hmm = MultinomialHMM(n_components=trans.shape[0],
                         algorithm="viterbi",
                         random_state=seed)
    hmm.__setattr__("n_features", emis.shape[1])
    hmm.__setattr__("emissionprob_", emis)
    hmm.__setattr__("startprob_", np.array([1] + [0] * (trans.shape[0] - 1)))  # We will always start at the first state
    hmm.__setattr__("transmat_", trans)
    return hmm

def one_to_many(state):
    return [int(state / 4), state % 4]


def many_to_one(state):
    return state[0] * 4 + state[1]


if __name__ == "__main__":
    trans = genfromtxt('hmm_transition_matrix.csv', delimiter=',')
    emis = genfromtxt('hmm_observations_emission.csv', delimiter=',')
    hmm = build_hmm(trans, emis)
    print(hmm.sample(200)[1].ravel())
