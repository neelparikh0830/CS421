# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Project Part 1
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import string
import re


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info, see project statement for more details
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = user_input
    match = re.search(r"((^)[A-Z]{1}[a-zA-Z\.&\'-]*(\s[A-Z]{1}[a-zA-Z\.&\'-]*){1,3}(\s))|((\s)[A-Z]{1}[a-zA-Z\.&\'-]*(\s[A-Z]{1}[a-zA-Z\.&\'-]*){1,3}($))|((\s)[A-Z]{1}[a-zA-Z\.&\'-]*(\s[A-Z]{1}[a-zA-Z\.&\'-]*){1,3}(\s))|((^)[A-Z]{1}[a-zA-Z\.&\'-]*(\s[A-Z]{1}[a-zA-Z\.&\'-]*){1,3}($))", name)

    if match:
        return match.group(0).lstrip().rstrip()

    return ""



# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing, see project statement for more details
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    token = get_tokens(user_input)
    # !!! nlp!
    for x in token:
        if x not in string.punctuation:
            modified_input += " " + x
    modified_input = modified_input.lstrip().lower()
    return modified_input


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE]
    listOfVector = []
    for x in training_documents:
        x = preprocessing(x)
        listOfVector.append(x)
    tfidf_train = vectorizer.fit_transform(listOfVector)
    return vectorizer, tfidf_train


# Function: vectorize_test, see project statement for more details
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    # Initialize the TfidfVectorizer model and document-term matrix
    tfidf_test = None

    # [YOUR CODE HERE]
    user_input = preprocessing(user_input)
    user_input = [user_input] 
    tfidf_test = vectorizer.transform(user_input)

    return tfidf_test


# Function: train_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    training_data = training_data.todense()
    nb_model.fit(training_data, training_labels)

    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    # Initialize the output label
    label = 0
    label = nb_model.predict(tfidf_test.todense())
    # Write your code here.  You will need to make use of the GaussianNB
    # predict() function. You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]

    return label


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. In project components, this function might be graded, see rubric for details.
if __name__ == "__main__":
    # Display a welcome message to the user, and accept a user response of arbitrary length
    user_input = input("Welcome to the Neel's chatbot!  What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Query the user for a response
    user_input = input(f"Thanks {name}!  What do you want to talk about today?\n")

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")
    vectorizer, tfidf_train = vectorize_train(documents)
    nb_model = train_model(tfidf_train, labels)

    # Predict whether the user's sentiment is positive or negative
    tfidf_test = vectorize_test(vectorizer, user_input)

    label = get_model_prediction(nb_model, tfidf_test)
    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
