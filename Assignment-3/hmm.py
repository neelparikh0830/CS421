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
import copy



# Class definition for Hidden Markov Model (HMM)
# Do not make any changes to this class
# You are not required to understand the inner workings of this class
# However, you need the understand what each function does
class HMM:
    """
        Arguments:

        states: List of strings representing all states
        vocab: List of strings representing all unique observations
        trans_prob: Transition probability matrix. Each cell (i, j) contains P(states[j] | states[i])
        obs_likelihood: Observation likeliood matrix. Each cell (i, j) contains P(vocab[j] | states[i])
        initial_probs: Vector representing initial probability distribution. Each cell i contains P(states[i] | START)
    """
    def __init__(self, states: list[str], vocab: list[str], trans_prob: list[list[float]],
                    obs_likelihood: list[list[float]], initial_probs: list[float]) -> None:
        self.states = states[:]
        self.vocab = vocab[:]
        self.trans_prob = copy.deepcopy(trans_prob)
        self.obs_likelihood = copy.deepcopy(obs_likelihood)
        self.initial_probs = initial_probs[:]


    # Function to return transition probabilities P(to_state|from_state)
    # Arugmnets:
    # to_state (str): state to which we are transitions
    # from_state (str): state from which we are transitioning
    #
    # Returns:
    # float: The probability of transition
    def tprob(self, to_state: str, from_state: str) -> float:
        if not (to_state in self.states and from_state in ['START'] + self.states):
            raise ValueError("invalid input state(s)")
        to_state_idx = self.states.index(to_state)
        if from_state == 'START':
            return self.initial_probs[to_state_idx]
        from_state_idx = self.states.index(from_state)
        return self.trans_prob[from_state_idx][to_state_idx]
    

    # Function to return observation likelihood P(obs|state)
    # Arugmnets:
    # obs (str): the observation string
    # state (str): state at which the observation is made
    #
    # Returns:
    # float: The probability of observation at given state
    def oprob(self, obs: str, state: str) -> float:
        if not obs in self.vocab:
            raise ValueError('invalid observation')
        if not (state in self.states and state != 'START'):
            raise ValueError('invalid state')
        obs_idx = self.vocab.index(obs)
        state_idx = self.states.index(state)
        return self.obs_likelihood[obs_idx][state_idx]
    
    # Function to retrieve all states
    # Arugmnets: N/A
    # Returns: 
    # list[str]: A list of strings containig the states
    def get_states(self) -> list[str]:
        return self.states.copy()


# Function to initialize an HMM using the weather-icecream example in Figure 6.3 (Jurafsky & Martin v2)
# Do not make any changes to this function
# You are not required to understand the inner workings of this function
# Arugmnets: N/A
# Returns: 
# HMM: An instance of HMM class
def initialize_icecream_hmm() -> HMM:
    states = ['HOT', 'COLD']
    vocab = ['1', '2', '3']
    tprob_mat = [[0.7, 0.3], [0.4, 0.6]]
    obs_likelihood = [[0.2, 0.5], [0.4, 0.4], [0.4, 0.1]]
    initial_prob = [0.8, 0.2]
    hmm = HMM(states, vocab, tprob_mat, obs_likelihood, initial_prob)
    return hmm


# Function to implement viterbi algorithm
# Arguments:
# hmm (HMM): An instance of HMM class as defined in this file. Note that it can be any HMM, icecream hmm is just an example
# obs (str): A string of observations, e.g. ("132311")
#
# Returns: seq, prob
# Where, seq (list) is a list of states showing the most likely path and prob (float) is the probability of that path
# IMPORTANT NOTE: Seq sould not contain 'START' or 'END' and In case of a conflict, you should pick the state at lowest index
def viterbi(hmm: HMM, obs: str) -> tuple[list[str], float]:
    # [YOUR CODE HERE]


    # trellis = [[0 for _ in range()]]
    
    
#     {\displaystyle (O,S,\Pi ,Tm,Em):best\_path}
  
# {\displaystyle (O,S,\Pi ,Tm,Em):best\_path}   Tm: transition matrix   Em: emission matrix
    
  
    
#     {\displaystyle trellis\leftarrow matrix(length(S),length(O))}
  
# {\displaystyle trellis\leftarrow matrix(length(S),length(O))}      To hold probability of each state given each observation
    
  
    
#     {\displaystyle pointers\leftarrow matrix(length(S),length(O))}
  
# {\displaystyle pointers\leftarrow matrix(length(S),length(O))}    To hold backpointer to best prior state
#     for s in 
  
    
#     {\displaystyle range(length(S))}
  
# {\displaystyle range(length(S))}:                Determine each hidden state's probability at time 0…
        
  
    
#     {\displaystyle trellis[s,0]\leftarrow \Pi [s]\cdot Em[s,O[0]]}
  
# {\displaystyle trellis[s,0]\leftarrow \Pi [s]\cdot Em[s,O[0]]}
#     for o in 
  
    
#     {\displaystyle range(1,length(O))}
  
# {\displaystyle range(1,length(O))}:              …and after, tracking each state's most likely prior state, k
#         for s in 
  
    
#     {\displaystyle range(length(S))}
  
# {\displaystyle range(length(S))}:
            
  
    
#     {\displaystyle k\leftarrow \arg \max(k\ {\mathsf {in}}\ trellis[k,o-1]\cdot Tm[k,s]\cdot Em[s,o])}
  
# {\displaystyle k\leftarrow \arg \max(k\ {\mathsf {in}}\ trellis[k,o-1]\cdot Tm[k,s]\cdot Em[s,o])}
            
  
    
#     {\displaystyle trellis[s,o]\leftarrow trellis[k,o-1]\cdot Tm[k,s]\cdot Em[s,o]}
  
# {\displaystyle trellis[s,o]\leftarrow trellis[k,o-1]\cdot Tm[k,s]\cdot Em[s,o]}
            
  
    
#     {\displaystyle pointers[s,o]\leftarrow k}
  
# {\displaystyle pointers[s,o]\leftarrow k}
    
  
    
#     {\displaystyle best\_path\leftarrow list()}
  
# {\displaystyle best\_path\leftarrow list()}
    
  
    
#     {\displaystyle k\leftarrow \arg \max(k\ {\mathsf {in}}\ trellis[k,length(O)-1])}
  
# {\displaystyle k\leftarrow \arg \max(k\ {\mathsf {in}}\ trellis[k,length(O)-1])}    Find k of best final state
#     for o in 
  
    
#     {\displaystyle range(length(O)-1,-1,-1)}
  
# {\displaystyle range(length(O)-1,-1,-1)}:      Backtrack from last observation
        
  
    
#     {\displaystyle best\_path.insert(0,S[k])}
  
# {\displaystyle best\_path.insert(0,S[k])}                Insert previous state on most likely path
        
  
    
#     {\displaystyle k\leftarrow pointers[k,o]}
  
# {\displaystyle k\leftarrow pointers[k,o]}                      Use backpointer to find best previous state
#     return 
  
    
#     {\displaystyle best\_path}
  
# {\displaystyle best\_path}

    return ['HOT', 'Cold'], 0.2


# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. If you want, you may run the code from terminal as:
# python hmm.py
# It should produce the following output (from the template):
#
# States: ['HOT', 'COLD']
# P(HOT|COLD) = 0.4
# P(COLD|START) = 0.2
# P(1|COLD) = 0.5
# P(2|HOT) = 0.4
# Path: ['HOT', 'COLD']
# Probability: 0.2

def main():
    # We can initialize our HMM using initialize_icecream_hmm function
    hmm = initialize_icecream_hmm()

    # We can retrieve all states as
    print("States: {0}".format(hmm.get_states()))

    # We can get transition probability P(HOT|COLD) as
    prob = hmm.tprob('HOT', 'COLD')
    print("P(HOT|COLD) = {0}".format(prob))

    # We can get transition probability P(COLD|START) as
    prob = hmm.tprob('COLD', 'START')
    print("P(COLD|START) = {0}".format(prob))

    # We can get observation likelihood P(1|COLD) as
    prob = hmm.oprob('1', 'COLD')
    print("P(1|COLD) = {0}".format(prob))

    # We can get observation likelihood P(2|HOT) as
    prob = hmm.oprob('2', 'HOT')
    print("P(2|HOT) = {0}".format(prob))

    # You should call the viterbi algorithm as
    path, prob = viterbi(hmm, "13213")
    print("Path: {0}".format(path))
    print("Probability: {0}".format(prob))


################ Do not make any changes below this line ################
if __name__ == '__main__':
    exit(main())