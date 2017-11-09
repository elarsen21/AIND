import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for word_id in range(0, len(test_set.get_all_Xlengths())):
            X, lengths = test_set.get_item_Xlengths(word_id)  # IS THIS THE CORRECT WAY TO GET X, LENGTHS?
            likelihoods = {}
            for word, model in models.items():
                try:
                    the_score = models[word].score(X, lengths)  # FIX THIS PART AND EVERYTHING WILL BE OK,  model = best model
                except:
                    the_score = float("-inf")
                likelihoods[word] = the_score
            probabilities.append(likelihoods)
            guess = max(probabilities[word_id], key = probabilities[word_id].get)
            guesses.append(guess)
    return (probabilities, guesses)
