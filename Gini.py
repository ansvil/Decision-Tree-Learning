import math
import random
from collections import Counter
header1 = ["color", "size", "act","age","inflated"]
header11 = ["color", "diameter", "label"]
header=['age', 'workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country',"label" ]

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def calc_entropy(data):
 #Calculate the length of the data-set
 entries = len(data)
 labels = {}
 #Read the class labels from the data-set file into the dict object "labels"
 for rec in data:
   label = rec[-1]
   if label not in labels.keys():
     labels[label] = 0
     labels[label] += 1
 #entropy variable is initialized to zero
 entropy = 0.0
 #For every class label (x) calculate the probability p(x)
 for key in labels:
   prob = float(labels[key])/entries
 #Entropy formula calculation
   entropy -= prob * math.log(prob,2)
 #print "Entropy -- ",entropy
 #Return the entropy of the data-set
 return entropy


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def dataset_split(data, arc, val):
    # declare a list variable to store the newly split data-set
    newData = []
    # iterate through every record in the data-set and split the data-set
    for rec in data:
        if rec[arc] == val:
            reducedSet = list(rec[:arc])
            reducedSet.extend(rec[arc + 1:])
            newData.append(reducedSet)
    # return the new list that has the data-set that is split on the selected attribute
    return newData


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

def prob(rows):
    #calculate num of different elems in sample
    #returns an array of elems
    #if were any dif elems returns first found elem
    n = len(rows[0]) - 1  # number of columns
    a=[]
    for i in range(len(rows)):
        a.append(rows[i][n])
    c=Counter(a)
    if len(c)==1:
        return  rows
    max=0
    el=None
    for elem in c.keys():
        if c.get(elem)>max:
            max, el= c.get(elem), elem
    for i in range(len(rows)):
        if rows[i][n]==el:
            return [rows[i]]


def find_best_split_entropy(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    n_features = len(rows[0]) - 1  # number of columns
    baseEntropy = calc_entropy(rows)
    max_InfoGain = 0.0 # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    bestAttr = -1
    for col in range(n_features):  # for each feature

        AttrLis = [row[col] for row in rows]  # unique values in the column
        AttrList=set(AttrLis)
        newEntropy = 0.0
        attrEntropy = 0.0
        for val in AttrList:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)
            newData = dataset_split(rows, col, val)
            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            prob = len(newData) / float(len(rows))
            # Calculate the information gain from this split
            newEntropy = prob * calc_entropy(newData)
            attrEntropy += newEntropy
        infoGain = baseEntropy - attrEntropy

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
        if infoGain >= max_InfoGain and attrEntropy != 0.0:
            max_InfoGain, best_question,bestAttr = infoGain, question,col

    return max_InfoGain, best_question

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows,f):# f=0 - cart,f=11-id3
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    if(f==0):
        gain, question = find_best_split(rows)
    else:
        gain, question = find_best_split_entropy(rows)
    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(prob(rows))



    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows,f)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows,f)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

#read from file
def read_file(filename):
    tmp=[]
    with open(filename) as file:
        for line in file.readlines():
            line.strip()
            tmp.append(line.strip().split(','))


    return tmp


def accuracy(data, tree):
    sum=0.0
    for row in data:
        f=0
        sum1=0.0
        c = classify(row, tree)
        v = c.items()
        tmp=row[-1]
        for val in v:
            if val[0] == tmp:
                sum = sum + 1
    return sum/len(data)

#forming an array of N random lines from data array
#M-size of data
def form_array(data,N,M):
    test=[]
    train=[]
    # form a training sample
    for i in range(N):
        t = random.randint(0, N)
        while data[t]==None:
            t = random.randint(0, N)
        train.append(data[t])
        data[t]=None
    # form a control sample
    for i in range(M-1):
        if data[i]!=None:
            test.append(data[i])
    return test, train


def main():
    """training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]
testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]"""


    #readd ata
    all_data = read_file("in.txt")
    test = read_file("test.txt")
    t=0
    #first method
    ind=[30, 60,100,150]
    testing_data=test[:100]
    for i in ind:
        print(str(i))
        print("\n")
        #form a training sample
        training_data=all_data[:i]
        my_tree = build_tree(training_data,1)
        print("id3\n")
        print(str(accuracy(testing_data, my_tree)))
        my_tree = build_tree(training_data, 0)
        print("cart\n")
        print(str(accuracy(testing_data,my_tree)))

    #adding data to training sample
    ind=[150,200]
    training_data = all_data[:150]
    for i in ind:
        print(str(i))
        print("\n")
        #form testing data sample
        testing_data=test[:i]
        my_tree = build_tree(training_data,1)
        print("id3\n")
        print(str(accuracy(testing_data, my_tree)))
        my_tree = build_tree(training_data, 0)
        print("cart\n")
        print(str(accuracy(testing_data,my_tree)))
"""
    #second option
    for i in range(10):
        #считываем данные
        all_data = read_file("all.txt")
        #form data samples
        testing_data, training_data=form_array(all_data,300,400)
        print(str(i))
        print("\n")
        #form training data
        my_tree = build_tree(training_data, 1)
        print("id3\n")
        print(str(accuracy(testing_data, my_tree)))
        my_tree = build_tree(training_data, 0)
        print("cart\n")
        print(str(accuracy(testing_data, my_tree)))"""


if __name__ == "__main__":
    main()
