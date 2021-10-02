## import modules here 

################# Question 0 #################

def add(a, b):  # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x):  # do not change the heading of the function
    MAX_ITER = 1000
    temp = float(x) / 2.0
    low = 0.0
    high = float(x)
    for i in range(MAX_ITER):
        if temp * temp > x:
            high = temp
        elif temp * temp < x:
            low = temp
        else:
            return int(temp)
        temp = (low + high) / 2.0

    return int(temp)


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON=1E-7, MAX_ITER=1000):  # do not change the heading of the function
    for ITER in range(MAX_ITER):
        if ITER == 0:
            x = x_0
        else:
            x = x_new
        x_new = x - f(x) / fprime(x)
        if abs(x - x_new) < EPSILON:
            return x_new


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)


# find the location of brackets
def find_brackets(tokens):
    stack = []
    brackets = []
    for i in range(len(tokens)):
        if tokens[i] == "[":
            stack.append(i)
        elif tokens[i] == "]":
            left = stack.pop()
            brackets.append([left, i])
    brackets.sort()
    return brackets


# get a dictionary {root: children}
def get_root_to_child(toks, loc_rots, brackets):
    root_to_child = {}
    for i in range(len(brackets)):
        loc_of_leaves = []
        for j in range(brackets[i][0], brackets[i][1] + 1):
            if toks[j] != "[" and toks[j] != "]":
                loc_of_leaves.append(j)
        root_to_child[loc_rots[i]] = loc_of_leaves

    # cut the Repetitive nodes
    for i in root_to_child:
        for j in root_to_child:
            if j in root_to_child[i]:
                root_to_child[i] = sorted(list(set(root_to_child[i]) - set(root_to_child[j])))
    return root_to_child


def recursion_make_tree(rot, toks, root_dic):
    tree = Tree(toks[rot])
    for child in root_dic[rot]:
        if child not in root_dic:
            tree.add_child(Tree(toks[child]))
        else:
            tree.add_child(recursion_make_tree(child, toks, root_dic))
    return tree


def make_tree(tokens):  # do not change the heading of the function
    loc_of_brackets = find_brackets(tokens)

    # according to loc_of_brackets, get all the location of roots
    loc_of_roots = []
    for item in loc_of_brackets:
        loc_of_roots.append(item[0] - 1)

    root_to_child = get_root_to_child(tokens, loc_of_roots, loc_of_brackets)

    tree = recursion_make_tree(loc_of_roots[0], tokens, root_to_child)
    return tree


def count_depth(root, dep, mx_dep):
    if len(root.children) > 0:
        dep = dep + 1
        for child in root.children:
            mx_dep = max(dep, count_depth(child, dep, mx_dep))
    return mx_dep


def max_depth(root):  # do not change the heading of the function
    if root is not None:
        depth = 1
        mx_depth = 1
        max_dep = count_depth(root, depth, mx_depth)
    else:
        max_dep = 0
    return max_dep
