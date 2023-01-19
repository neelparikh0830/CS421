# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Assignment 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import re



# Function to convert a given string into a list of lowercase tokens (or words)
# Args:
#   inp_str: input string 
# Returns: word list, dtype: list of strings
def get_tokens(inp_str):
    candidate_tokens = re.split("[^A-Za-z0-9_-]", inp_str)
    filtered_tokens =  filter(len, candidate_tokens)
    lowercase_tokens = [t.lower() for t in filtered_tokens]
    return lowercase_tokens


# Function to compute Word Error Rate (WER) between a given reference phrase and hypothesis
# Args:
#   ref: reference text, dtype: string
#   hyp: hypothesis text, dtype: string
#   insertion: insertion cost, dtype: float
#   deletion: deletion cost, dtype: float
#   substitution: substitution cost, dtype: float
# Returns: Word Error Rate (WER), dtype: float
#
# Hint: you need to compute the minimum edit distance between sequence of words (returned by get_tokens) and count the number of
# operations (substitution, insertion, deletion) to get WER, you may use list of lists for matrix representation
def wer(ref, hyp, insertion, substitution, deletion):
    # Get reference and hypothesis words
    ref_words = get_tokens(ref)
    hyp_words = get_tokens(hyp)
    # [YOUR CODE HERE]
    wer = EditDistDP(ref_words,hyp_words, insertion, substitution, deletion)
    value = wer/len(ref_words)
    return value



# source : https://www.geeksforgeeks.org/edit-distance-dp-5/


def EditDistDP(ref, hyp, insertion, substitution, deletion):
     
    DP = [[0 for i in range(len(ref) + 1)]
             for j in range(2)];
 
    for i in range(0, len(ref) + 1):
        DP[0][i] = i
 
    for i in range(1, len(hyp) + 1):
         
        for j in range(0, len(ref) + 1):
            if (j == 0):
                DP[i % 2][j] = i
 
            elif(ref[j - 1] == hyp[i-1]):
                DP[i % 2][j] = DP[(i - 1) % 2][j - 1]
             
            else:
                DP[i % 2][j] = (min(DP[(i - 1) % 2][j]+insertion,
                                    min(DP[i % 2][j - 1]+deletion,
                                  DP[(i - 1) % 2][j - 1]+substitution)))
             
    return DP[len(hyp) % 2][len(ref)]



# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python wer.py
# It should produce the following output (with correct solution):
#     $ python3 wer.py
#     Reference words: ['i', 'love', 'nlp']
#     Hypothesis words: ['i', 'really', 'love', 'nlp']
#     Word Error Rate (WER): 0.3333333333333333
def main():
    # The reference and hypothesis strings
    ref_str = "I love NLP"
    hyp_str = "I really love NLP"

    # Convert to lowercase words
    ref_words = get_tokens(ref_str)
    hyp_words = get_tokens(hyp_str)

    print("Reference words:", ref_words)
    print("Hypothesis words:", hyp_words)

    # Define costs
    INS_COST = 1
    SUB_COST = 2
    DEL_COST = 1

    # Call wer function on reference and hypothesis strings to compute word error rate
    # It just returns 0 at this point
    # In this example, the correct WER should be 1/3 or 0.3333
    wer_val = wer(ref_str, hyp_str, INS_COST, SUB_COST, DEL_COST)
    print("Word Error Rate (WER):", wer_val)


if __name__ == '__main__':
    main()