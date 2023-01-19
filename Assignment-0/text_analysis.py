# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Assignment 0
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


import json


# Function to read a file
# filepath: full path to the file
#
# Returns: a string containing file contents
def read_file(filepath):
    # [YOUR CODE HERE]
    f = open(filepath, "r")
    data = f.read()
    return data


# Function to convert a string to a list of lowercase words
# in_str: a string containing the text
#
# Returns: a list containing lowercase words
def conver_to_words(in_str):
    # [YOUR CODE HERE]
    temp = in_str.lower()
    mwords = temp.split()
    return mwords


# Function to count the words
# words: a list containing words
#
# Returns: a dictionary where keys are words and corresponding values are counts
def get_wc(words):
    word_counts = dict()
    # [YOUR CODE HERE]
    for i in words:
        if (i in word_counts):
            word_counts[i] += 1
        else:
            word_counts[i] = 1
    return word_counts


# Function to store the dictionary as JSON
# dictionary: a python dictionary
# out_filepath: path to output file to store the JSON data
#
# Returns: a dictionary where keys are words and corresponding values are counts
def to_json(dictionary, out_filepath):
    # [YOUR CODE HERE]
    json_object = json.dumps(dictionary, indent=4)
    f = open(out_filepath, "w")
    f.write(json_object)
    return None




# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python text_analysis.py
# It should produce the following output (with correct solution):
#       $ python text_analysis.py
#       File loaded: CHAPTER I.
#       Down the Rabbit-Hol...
#       Words: ['chapter', 'i.', 'down', 'the', 'rabbit-hole']
#       The word alice appeared 221 times

def main():
    # Read the entire file in a string
    content = read_file('alice.txt')
    # Print first 30 characters
    print(f'File loaded: {content[:30]}...')

    # Obtain words from the content
    words = conver_to_words(content)
    # Print the first 5 words
    print(f'Words: {words[:5]}')

    # Count the words
    word_counts = get_wc(words)
    # Print word counts for alice
    word = 'alice'
    if word in word_counts:
        print(f'The word {word} appeared {word_counts[word]} times')
    else:
        print("Word not found")

    # Save these counts as a JSON file
    to_json(word_counts, 'word_counts.json')








################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()

