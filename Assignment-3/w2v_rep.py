# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Assignment 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


import numpy as np
from utils import *


# Function to get word2vec representations
# Arguments:
# docs: A list of strings, each string represents a document
# w2v_dict: A dictionary returned from the load_w2v function
# Returns: mat (numpy.ndarray) of size (len(docs), dim)
# mat is a two-dimensional numpy array containing vector representation for ith document (in input list docs) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for the provided Google News pre-trained vectors
def word2vec_rep(docs: list[str], w2v_dict: dict[str, np.ndarray]) -> np.ndarray:
    # [Your code here]

    mat = np.zeros((len(docs), 300))
    for i, j in enumerate(docs):
        list1 = []
        list2 = []
        for token in get_tokens(j):
          if token.lower() not in get_stopwords():
            list1.append(token.lower())
        for token in list1:
          if token in w2v_dict:
             list2.append(w2v_dict[token])
        mat[i] =  np.average(list2, axis=0)
    return mat


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python w2v_rep.py
# It should produce the following output (from the template):
#
# Tokens for first document: ['Many', 'buildings', 'at', 'UIC', 'are', 'designed', 'in', 'the', 'brutalist', 'style']
# Is 'he' a stopword? True
# Is 'hello' a stopword? False
# Is 'she' a stopword? True
# Is 'uic' a stopword? False
def main():
    # Initialize the corpus
    sample_corpus = ['Many buildings at UIC are designed in the brutalist style.',
                    'Brutalist buildings are generally characterized by stark geometric lines and exposed concrete.',
                    'One famous proponent of brutalism was a Chicago architect named Walter Netsch.',
                    'Walter Netsch designed the entire UIC campus in the early 1960s.',
                    'When strolling the campus and admiring the brutalism, remember to think of Walter Netsch!']
    
    # We can tokenize the first document as
    tokens = get_tokens(sample_corpus[0])
    print("Tokens for first document: {0}".format(tokens))

    # We can fetch stopwords and check if a word is a stopword
    stopwords = get_stopwords()
    for word in ['he', 'hello', 'she', 'uic']:
        answer = word in stopwords
        print("Is '{0}' a stopword? {1}".format(word, answer))
    
    # # We can load numpy word vectors using load_w2v as
    # w2v = load_w2v('w2v.pkl')
    # # And access these vectors using the dictionary
    # print(w2v['not'])
    
    

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
