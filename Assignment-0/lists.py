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


# Function to calculate the squared list
# num_list: a list containing numerical values
#
# Returns: a list containing the squared values
def sq_list(num_list):
    # [YOUR CODE HERE]
    new_list = [];
    # sum = 0;
    for i in num_list:
        # sum = i*i;
        new_list.append(i*i);
    return new_list


# Function to add two lists
# list_a: a list of numerical values
# list_b: a list of numerical values
#
# Returns: a list containing the sum of corresponding values from list_a and list_b
def add_lists(list_a, list_b):
    # [YOUR CODE HERE]
    new_list = [];
    for i in range(len(list_a)):
        new_list.append(list_a[i]+list_b[i])
    return new_list


# Function to process the list by adding square of each element to itself
# num_list: a list containing numerical values
#
# Returns: a list containing the processed values
def process_list(num_list):
    # [YOUR CODE HERE]
    # new_list = sq_list(num_list);
    # new_list1 = add_lists(sq_list(num_list), num_list);
    return add_lists(sq_list(num_list), num_list)





# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python lists.py
# It should produce the following output (with correct solution):
#       $ python lists.py
#       The squared list is: [4, 9, 36]
#       The sum of lists is: [2, 6, 9]
#       The processed list is: [6, 20, 12]

def main():
    # Calculate and print the squared list
    squard_list = sq_list([2, 3, 6]) 
    print(f'The squared list is: {squard_list}')

    # Test add_lists
    list_sum  = add_lists([1, 5, 3], [1, 1, 6])
    print(f'The sum of lists is: {list_sum}')

    # Process the list
    proc_list = process_list([2, 4, 3])
    print(f'The processed list is: {proc_list}')



################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()

