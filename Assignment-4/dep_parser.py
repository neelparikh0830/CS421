# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Assignment 4
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

from nltk.parse.corenlp import CoreNLPDependencyParser



# Function: get_dependency_parse(input)
# This function accepts a raw string input and returns a CoNLL-formatted output
# string with each line indicating a word, its POS tag, the index of its head
# word, and its relation to the head word.
# Parameters:
# input - A string containing a single text input (e.g., a sentence).
# Returns:
# output - A string containing one row per word, with each row containing the
#          word, its POS tag, the index of its head word, and its relation to
#          the head word.
def get_dependency_parse(input: str):
    output = ""

    # Make sure your server is running!  Otherwise this line will not work.
    dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")

    # Write your code here.  You'll want to make use of the
    # CoreNLPDependencyParser's raw_parse() method, which will return an
    # iterable object containing DependencyGraphs in order of likelihood of
    # being the correct parse.  Hint: You'll only need to keep the first one!
    #
    # You'll also likely want to make use of the DependencyGraph's to_conll()
    # method---check out the docs to see which style (3, 4, or 10) to select:
    # https://www.nltk.org/_modules/nltk/parse/dependencygraph.html#DependencyGraph.to_conll
    nike, = dep_parser.raw_parse(input)
    output = nike.to_conll(4)

    return output


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python dep_parser.py
# It should produce the following output (if the solution is correct):
# I       PRP     2       nsubj
# love    VBP     0       ROOT
# NLP     NNP     2       obj
# !       .       2       punct
def main():
    output = get_dependency_parse("I love NLP!")
    print(output)



################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
