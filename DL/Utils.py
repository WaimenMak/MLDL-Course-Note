# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 18:05
# @Author  : Weiming Mai
# @FileName: Utils.py
# @Software: PyCharm

from Datastructure import Queue
from Nodes import *

def Dense(left_node, output_num):
    weight = Variable(np.random.normal(0, 0.1, [left_node.value.shape[1], output_num]))
    # weight = Variable(np.random.rand(left_node.value.shape[1], output_num))
    W_x = Dot(left_node, weight)
    W_x.output_val()
    output = Add(W_x, Variable(np.random.normal(0, 0.1, (1, W_x.value.shape[1]))))
    # output = Add(W_x, Variable(np.random.rand(1, W_x.value.shape[1])))
    output.output_val()

    return output


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


# #breadth-first search
# def Backprop(root):
#     '''
#     :param root:
#     :return:
#     '''
#     if root == None or (root.last_left == None and root.last_right == None):
#         return
#
#     Q = Queue(20, root)
#     Q.enqueue(root)
#
#     while (Q.isEmpty() == False and not Q.isFull()):
#         size = Q.queueSize()
#         for i in range(size):
#             temp = Q.dequeue()
#             if temp.grad is None:             # if is the first node
#                 if temp.name == 'Dense':
#                     temp.grad = temp.weight_bias.T
#                     temp.compute_gradient()
#                     if temp.last_left != None and temp.last_left.require_grad:
#                         temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
#                         Q.enqueue(temp.last_left)
#                 else:
#                     temp.grad = np.ones(temp.value.shape)
#                     temp.compute_gradient()
#                     if temp.last_left != None and temp.last_left.require_grad:
#                         temp.last_left.grad = temp.last_left.sub_grad
#                         Q.enqueue(temp.last_left)
#                     if temp.last_right != None and temp.last_right.require_grad:
#                         temp.last_right.grad = temp.last_right.sub_grad
#                         Q.enqueue(temp.last_right)
#             else:
#                 temp.compute_gradient()
#                 if temp.last_left != None and temp.last_left.require_grad:
#                     temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
#                     Q.enqueue(temp.last_left)
#                 if temp.last_right != None and temp.last_right.require_grad:
#                     temp.last_right.grad = chain_rule(temp.grad, temp.last_right.sub_grad, 'r')
#                     Q.enqueue(temp.last_right)


# breadth-first search
def Backprop(root):
    '''
    :param root:
    :return: need to be updated nodes
    '''
    if root == None or (root.last_left == None and root.last_right == None):
        return

    param = []  # record the nodes that need to be updated

    Q = Queue(20, root)
    Q.enqueue(root)

    while (Q.isEmpty() == False and not Q.isFull()):
        size = Q.queueSize()
        for i in range(size):
            temp = Q.dequeue()
            if temp.grad is None:  # if is the first node
                temp.grad = np.ones(temp.value.shape)
                temp.compute_gradient()
                if temp.need_update == True:
                    param.append(temp)
                if temp.last_left != None and temp.last_left.require_grad:
                    temp.last_left.grad = temp.last_left.sub_grad
                    Q.enqueue(temp.last_left)
                if temp.last_right != None and temp.last_right.require_grad:
                    temp.last_right.grad = temp.last_right.sub_grad
                    Q.enqueue(temp.last_right)
            else:
                temp.compute_gradient()
                if temp.need_update == True:
                    param.append(temp)
                if temp.last_left != None and temp.last_left.require_grad:
                    temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
                    Q.enqueue(temp.last_left)
                if temp.last_right != None and temp.last_right.require_grad:
                    temp.last_right.grad = chain_rule(temp.grad, temp.last_right.sub_grad, 'r')
                    Q.enqueue(temp.last_right)

    return param


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


def MSE(output, label):  # Mean Square Error
    '''

    :param output: the final output nodes of the network 
    :param label: ys
    :return: loss
    '''
    loss = F_Norm(Minus(output, label))
    return loss


def Cross_Entropy(output, label):  # Cross Entropy
    '''

    :param output: the final output nodes of the network 
    :return: loss
    '''
    pass


# GradientDescentOptimizer
class G_D_Optimizer():
    '''

    :param param: nodes that need to be updated
    :param root:  the output of the cost function
    :return: 
    '''
    def __init__(self, param, lr):
        self.param = param
        self.lr = lr            #learning rate

    def train(self):
        for i in range(len(self.param)):
            if self.param[i].value.shape == self.param[i].grad.shape:
                self.param[i].value = self.param[i].value - self.lr * self.param[i].grad
            else:
                self.param[i].value = self.param[i].value - self.lr * sum(self.param[i].grad)  # sum: column-wise

