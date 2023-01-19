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


# Function to return the sum of first n positive odd numbers
#
# Returns: sum as an integer
def sum_odd(n):
    # [YOUR CODE HERE]
    sum = 0
    curr = 1
    for i in range(n):
        sum += curr
        curr += 2
    return sum


# Function to calculate the sum of first N fibonacci numbers
#
# Returns: sum as an integer
def sum_fib(n):
    # [YOUR CODE HERE]
    def fib(n):
        if (n <= 1):
            return n
        return fib(n-1) + fib(n-2)

    sum = 0
    for i in range(0, n):
        sum = sum + fib(i)
    return sum






# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python loops.py
# It should produce the following output (with correct solution):
# 	    $ python loops.py
#       The sum of first 5 positive odd numbers is: 25
#       The sum of first 5 fibonacci numbers is: 7

def main():
    # Call the function to calculate sum
    osum = sum_odd(5) 

    # Print it out
    print(f'The sum of first 5 positive odd numbers is: {osum}')

    # Call the function to calculate sum of fibonacci numbers
    fsum = sum_fib(5)
    print(f'The sum of first 5 fibonacci numbers is: {fsum}')




################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()

