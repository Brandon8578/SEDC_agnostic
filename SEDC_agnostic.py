""" Import libraries """
import time
import numpy as np
from itertools import compress
import pandas as pd

class SEDC_Explainer(object):
    def __init__(self, feature_names, classifier_fn, threshold_classifier,
                 max_iter=50, max_explained=1, BB=True, max_features=30,
                 time_maximum=120,verbose = True):
        self.feature_names = feature_names
        self.classifier_fn = classifier_fn
        self.threshold_classifier = threshold_classifier
        self.max_iter = max_iter
        self.max_explained = max_explained
        self.BB = BB
        self.max_features = max_features
        self.time_maximum = time_maximum
        self.verbose = verbose

    def explanation(self,instance):
        """ Generates explanation(s) for a positively-predicted instance
        (more specifically, an instance with a predicted score lying above
        a certain threshold value)

        Args:
            instance: [numpy.array or sparse matrix] instance on which
            to explain the model prediction

        Returns:
            A tuple (explanation_set[0:self.max_explained], number_active_elements,
            number_explanations, minimum_size_explanation, time_elapsed,
            explanations_score_change[0:self.max_explained]), where:

                explanation_set: explanation(s) ranked from high to low change
                in predicted score or probability.
                The number of explanations shown depends on the argument max_explained.

                number_active_elements: number of active elements of
                the instance of interest.

                number_explanations: number of explanations found by algorithm.

                minimum_size_explanation: number of features in the smallest explanation.

                time_elapsed: number of seconds passed to generate explanation(s).

                explanations_score_change: change in predicted score/probability
                when removing the features in the explanation, ranked from
                high to low change.
        """
        start_time = time.time()
        elapsed_time = time.time()-start_time
        iteration = 0
        nb_explanations = 0
        minimum_size_explanation = np.nan
        explanations = []
        explanations_sets = []
        explanations_score_change = []
        score_predicted = self.classifier_fn(instance)  # predicted score
        indices_active_elements = list(np.nonzero(instance)[1])
        number_active_elements = len(indices_active_elements)

        #Initiate lists
        pruned_list = []
        expansion_candidates = [[]]#Empty list to initiate loop, we start from no perturbation
        scores_expansion_candidates = [score_predicted]#no perturbation has score_predicted as score
        len_expansion_candidates = [0]
        explanations = []
        scores_explanations = []
        len_explanations = []

        #Start expanding
        while (iteration<self.max_iter)&(elapsed_time<self.time_maximum)&(expansion_candidates!=[])&(nb_explanations<self.max_explained):
            #1)Select combination with lowest score to expand further
            index_to_expand = np.argmin(scores_expansion_candidates)
            to_expand = expansion_candidates[index_to_expand]

            pruned_list.append(to_expand)#Add to_expand to list of pruned_list combinations
            expansion_candidates.pop(index_to_expand)#remove to_expand from list of not_expanded_combinations
            scores_expansion_candidates.pop(index_to_expand)
            len_expansion_candidates.pop(index_to_expand)

            #2)Expand combination
            expansions = []
            indices_expansions = np.setdiff1d(indices_active_elements,to_expand).tolist() #prevent to expand on feature already in perturbation
            for index in indices_expansions:
                expansion = sorted(to_expand + [index])#Sorted to make sure they match previous expansions
                if not (any([set(explanation).issubset(expansion) for explanation in explanations])|(expansion in expansion_candidates)|(expansion in expansion_candidates)):
                    expansions.append(expansion)

            #3) get Scores for expansions
            perturbations_to_score = []
            for expansion in expansions:#todo what if expansions is empty?
                perturbation = instance.copy()
                perturbation[0,expansion]=0
                perturbations_to_score.append(perturbation)
            perturbations_to_score = np.concatenate(perturbations_to_score) #convert to array
            new_scores = list(self.classifier_fn(perturbations_to_score))
            new_len = [len(expansion) for expansion in expansions]

            #4) Add explanations and their socres to the correct lists

            explanations += list(compress(expansions,np.array(new_scores)<self.threshold_classifier))
            scores_explanations += list(compress(new_scores,np.array(new_scores)<self.threshold_classifier))
            len_explanations += list(compress(new_len,np.array(new_scores)<self.threshold_classifier))

            if len(to_expand)<(self.max_features-1):#if new features are smaller then max_features they are added to expansion_candidates
                expansion_candidates +=list(compress(expansions,np.array(new_scores) >= self.threshold_classifier))
                scores_expansion_candidates += list(compress(new_scores,np.array(new_scores) >= self.threshold_classifier))#todo: maybe don't add to expansion_candidates lists if score>score_predicted
                len_expansion_candidates += list(compress(new_len, np.array(new_scores) >= self.threshold_classifier))
            else:#if new expansions have length of max_features they are added to pruned_list list
                pruned_list += list(compress(expansions,np.array(new_scores) >= self.threshold_classifier))

            if self.BB&(explanations != []):
                min_length = min(len_explanations)
                pruned_list += list(compress(expansion_candidates,np.array(len_expansion_candidates) >= min_length))
                expansion_candidates =list(compress(expansion_candidates,np.array(len_expansion_candidates) < min_length))
                scores_expansion_candidates = list(compress(scores_expansion_candidates,np.array(len_expansion_candidates) < min_length))
                len_expansion_candidates = list(compress(len_expansion_candidates,np.array(len_expansion_candidates) < min_length))

            iteration +=1
            elapsed_time = time.time()-start_time
            nb_explanations = len(explanations)
            if self.verbose:
                print(f"iteration {iteration}, elapsed time: {round(elapsed_time)} seconds, number of explanations: {nb_explanations}")
        #

        if nb_explanations > 0:
            len_explanations = [len(explanation) for explanation in explanations]
            feature_explanations = [list(np.array(self.feature_names)[explanation]) for explanation in explanations]
            df_explanations = pd.concat([pd.Series(feature_explanations),
                                         pd.Series(len_explanations),
                                         pd.Series(scores_explanations)], axis=1)
            df_explanations = df_explanations.sort_values(by=[1, 2], ascending=[False, True], ignore_index=True)
            return_explanations = list(df_explanations.iloc[0:self.max_explained, 0])
            return_scores = list(df_explanations.iloc[0:self.max_explained, 2])
            minimum_size_explanation = df_explanations.iloc[0, 1]
        else:
            return_explanations = []
            return_scores = []


        return {'explanation set': return_explanations,
                'number active elements': number_active_elements,
                'number explanations found': nb_explanations,
                'size smallest explanation': minimum_size_explanation,
                'time elapsed':elapsed_time,
                'differences score': return_scores,
                'iterations': iteration}
#todo remove longer explanation if shorter is found which is subset of longer
