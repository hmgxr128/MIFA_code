import numpy as np
import pickle
import os

def gen_avail_prob_random(client_names, min_prob, max_prob, output_file, class_info):
    """
        For each client, randomly sample a participation probability from min_prob to max_prob.
    """

    np.random.seed(7)
    assert max_prob >= min_prob
    avail_prob = {}
    for key in client_names:
        avail_prob[key] = np.random.rand() * (max_prob - min_prob) + min_prob
    with open(output_file, 'wb') as f:
        pickle.dump(avail_prob, f, pickle.HIGHEST_PROTOCOL)

def gen_avail_prob_adversarial(client_names, min_prob, max_prob, output_file, class_info):
    """
        For each client, adversarially determine the participation probability from min_prob to max_prob,
             based on the two classes that the client contains.
        For 10-class classification problem, the labels are indexed with 0 ~ 9.
    """
    np.random.seed(7)
    assert max_prob >= min_prob
    avail_prob = {}
    for key in client_names:
        prob = min(class_info[key]) / (9)
        avail_prob[key] = prob * (max_prob - min_prob) + min_prob
    with open(output_file, 'wb') as f:
        pickle.dump(avail_prob, f, pickle.HIGHEST_PROTOCOL)

def gen_avail_prob_adversarial_dirichlet(client_names, min_prob, max_prob, output_file, class_info):
    """
        For each client, adversarially determine the participation probability from min_prob to max_prob,
             based on the two classes that the client contains.
        For 10-class classification problem, the labels are indexed with 0 ~ 9.
    """
    np.random.seed(7)
    assert max_prob >= min_prob
    avail_prob = {}
    for key in client_names:
        prob = class_info[key] / (9)
        avail_prob[key] = prob * (max_prob - min_prob) + min_prob
    with open(output_file, 'wb') as f:
        pickle.dump(avail_prob, f, pickle.HIGHEST_PROTOCOL)