# coding = utf-8
import numpy as np
from Datastructure import Queue

class Node():
    def __init__(self):
        self.last_right = None
        self.last_left = None
        self.next = None
        self.value = None
        self.grad = None
        self.sub_grad = None
        self.require_grad = True

class Input(Node):
    def __init__(self, X):
        super().__init__()
        self.value = X
        # self.require_grad = True

    def output_val(self):

        return self.value

    def compute_gradient(self):         ## compute gradient 是计算下面两个连接的节点，因此非operator的comput_gradient无作用
        pass

class Variable(Node):
    def __init__(self, X):
        super().__init__()
        # self.require_grad = True
        self.value = X

    def output_val(self):
        return self.value

    def compute_gradient(self):         ## compute gradient 是计算下面两个连接的节点，因此非operator的comput_gradient无作用
        pass

class Const(Node):
    def __init__(self, X):
        super().__init__()
        # self.require_grad = True
        self.require_grad = False
        self.value = X
    def output_val(self):
        return self.value

    def compute_gradient(self):         ## compute gradient 是计算下面两个连接的节点，因此非operator的comput_gradient无作用
        pass


class Add(Node):
    def __init__(self, left_node, right_node):
        super().__init__()
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self

    def output_val(self):
        if self.last_left.value.shape == self.last_right.value.shape:
            self.value = self.last_left.value + self.last_right.value

    def compute_gradient(self):
        self.last_left.sub_grad = np.ones(self.last_left.value.shape)
        self.last_right.sub_grad = np.ones(self.last_right.value.shape)


class Minus(Node):
    def __init__(self, left_node, right_node):
        super().__init__()
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self

    def output_val(self):
        if self.last_left.value.shape == self.last_right.value.shape:
            self.value = self.last_left.value - self.last_right.value

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = np.ones(self.last_left.value.shape)
        if self.last_right.require_grad:
            self.last_right.sub_grad = -1*np.ones(self.last_right.value.shape)



class Dot(Node):
    def __init__(self, left_node, right_node):
        super().__init__()
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self

    def output_val(self):
        self.value = np.dot(self.last_left.value, self.last_right.value)

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = self.last_right.value.T
        if self.last_right.require_grad:
            self.last_right.sub_grad = self.last_left.value.T


class Dense(Node):
    def __init__(self, left_node, output_num):
        '''
        :param left_node: 
        :param output_num: the number of the output neurons
        '''
        super().__init__()
        self.last_left = left_node
        self.out_size = output_num
        self.input = np.ones([self.last_left.value.shape[0], self.last_left.value.shape[1] + 1])
        self.input[:, 0:-2] = self.last_left.value
        self.weight_bias = np.random.normal(0, 0.1, [self.input.shape[1], self.out_size])
        left_node.next = self

    def output_val(self):
        self.value = np.dot(self.input, self.weight_bias)

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = self.weight_bias[0,-2,:].T
        self.grad = *****

    def update_param(self):
        pass

class L2_Norm(Node):
    def __init__(self, left_node):
        super().__init__()
        self.last_left = left_node
        left_node.next = self

    def output_val(self):
        self.value = np.dot(self.last_left.value.T, self.last_left.value)

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = self.last_left.value + self.last_left.value


def Forward(root):
    '''
    :param root: the last operater
    :return: output value 
    '''
    if root == None:
        return root

    Forward(root.last_left)
    Forward(root.last_right)
    root.output_val()

    return root

#breadth-first search
def Backprop(root):
    '''
    :param root:
    :return:
    '''
    if root == None or (root.last_left == None and root.last_right == None):
        return

    Q = Queue(20, root)
    Q.enqueue(root)

    while (Q.isEmpty() == False and not Q.isFull()):
        size = Q.queueSize()
        for i in range(size):
            temp = Q.dequeue()
            if temp.grad is None:
                temp.grad = np.ones(temp.value.shape)
                temp.compute_gradient()
                if temp.last_left != None and temp.last_left.require_grad:
                    temp.last_left.grad = temp.last_left.sub_grad
                    Q.enqueue(temp.last_left)
                if temp.last_right != None and temp.last_right.require_grad:
                    temp.last_right.grad = temp.last_right.sub_grad
                    Q.enqueue(temp.last_right)
            else:
                temp.compute_gradient()
                if temp.last_left != None and temp.last_left.require_grad:
                    temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
                    Q.enqueue(temp.last_left)
                if temp.last_right != None and temp.last_right.require_grad:
                    temp.last_right.grad = chain_rule(temp.grad, temp.last_right.sub_grad, 'r')
                    Q.enqueue(temp.last_right)


def chain_rule(par_1, par_2, node):
    '''
    :param par_1: gradient dL/dA (matrix)
    :param par_2: gradient dA/dx
    :param node: left node or right node (str)
    :return: dL/dx (matrix)
    '''
    if par_1.shape == par_2.shape:
        return par_1 * par_2

    elif node == 'l':
        return np.dot(par_1, par_2)

    elif node == 'r':
        return np.dot(par_2, par_1)




l1 = Input(np.array([[1,1,1],[2,2,2]]))
w = Variable(np.array([[4,2],[4,2],[4,2]]))
l2 = Dot(l1,w)
b = Variable(np.array([[1,1],[1,1]]))
l3 = Add(l2, b)
y = Const(np.array([[1,1],[2,2]]))
l4 = Minus(l3, y)
l5 = L2_Norm(l4)

o = Forward(l5)
Backprop(l5)
print(w.grad)