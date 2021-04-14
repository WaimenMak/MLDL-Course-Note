
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
        self.need_update = False

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
        self.need_update = True

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
        elif self.last_left.value.shape[1] == self.last_right.value.shape[1]:      # 加bias的情况
            temp_val = np.ones(self.last_left.value.shape)
            for i in range(self.last_left.value.shape[0]):
                temp_val[i,:] = self.last_right.value[0,:]
            self.value = temp_val + self.last_left.value

    def compute_gradient(self):
        self.last_left.sub_grad = np.ones(self.last_left.value.shape)    ## 以左边节点大小为准
        self.last_right.sub_grad = np.ones(self.last_left.value.shape)


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

class F_Norm(Node):
    def __init__(self, left_node):
        super().__init__()
        self.last_left = left_node
        left_node.next = self

    def output_val(self):     #Frobenius Norm
        # self.value = np.dot(self.last_left.value, self.last_left.value.T)
        sum = 0
        for i in range(self.last_left.value.shape[0]):
            for j in range(self.last_left.value.shape[1]):
                sum += self.last_left.value[i,j]**2
        self.value = sum/self.last_left.value.shape[0]

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = self.last_left.value + self.last_left.value


class Sigmoid(Node):
    def __init__(self, left_node):
        super().__init__()
        self.last_left = left_node
        self.value = self.last_left.value
        left_node.next = self

    def output_val(self):
        self.value = 1/(1 + np.exp(-1 * self.last_left.value))

    def compute_gradient(self):
        if self.last_left.require_grad:
            self.last_left.sub_grad = np.multiply(1 / (1 + np.exp(-1 * self.last_left.value)),(1 - 1 / (1 + np.exp(-1 * self.last_left.value))))

# class Dense(Node):
#     def __init__(self, left_node, output_num):
#         '''
#         :param left_node:
#         :param output_num: the number of the output neurons
#         '''
#         super().__init__()
#         self.name = 'Dense'
#         self.last_left = left_node
#         self.out_size = output_num
#         self.input = np.ones([self.last_left.value.shape[0], self.last_left.value.shape[1] + 1])
#         self.input[:, 0:-1] = self.last_left.value
#         self.weight_bias = np.random.normal(0, 0.1, [self.input.shape[1], self.out_size])
#         self.value = np.dot(self.input, self.weight_bias)                        #需要先计算当前节点的输出值，方便下一个Dense节点计算weight_bias的大小
#         left_node.next = self
#
#     def output_val(self):
#         self.value = np.dot(self.input, self.weight_bias)
#
#     def compute_gradient(self):
#         if self.last_left.require_grad and self.last_left.name == 'Dense':
#             self.last_left.sub_grad = self.last_left.weight_bias.T
#         elif self.last_left.require_grad:
#             self.last_left.sub_grad = self.weight_bias[0:-1,:].T
#
#
#     def update_param(self):
#         pass

# class Dense(Node):
#     def __init__(self, left_node, output_num):
#             '''
#             :param left_node:
#             :param output_num: the number of the output neurons
#             '''
#             super().__init__()
#             self.last_left = left_node
#             # self.out_size = output_num
#             self.weight = Variable(np.random.normal(0, 0.1, [left_node.value.shape[1], output_num]))
#             self.W_x = Dot(left_node, self.weight)
#             self.W_x.output_val()
#             self.output = Add(self.W_x, Variable(np.random.normal(0, 0.1, self.W_x.value.shape)))
#             self.output.output_val()
#             self.value = self.output.value
#
#     def output_val(self):
#         self.W_x.output_val()
#         self.output.output_val()
#         self.value = self.output.value
#         return self.value
#
#     def compute_gradient(self):
#         self.output

# def Dense(left_node, output_num):
#     weight = Variable(np.random.normal(0, 0.1, [left_node.value.shape[1], output_num]))
#     # weight = Variable(np.random.normal(0, 0.1, [left_node.value.shape[1], output_num]))
#     W_x = Dot(left_node, weight)
#     W_x.output_val()
#     output = Add(W_x, Variable(np.random.normal(0, 0.1, (1,W_x.value.shape[1]))))
#     output.output_val()
#
#     return output
#
#
#
#
#
# def Forward(root):
#     '''
#     :param root: the last operater
#     :return: output value
#     '''
#     if root == None:
#         return root
#
#     Forward(root.last_left)
#     Forward(root.last_right)
#     root.output_val()
#
#     return root
#
# # #breadth-first search
# # def Backprop(root):
# #     '''
# #     :param root:
# #     :return:
# #     '''
# #     if root == None or (root.last_left == None and root.last_right == None):
# #         return
# #
# #     Q = Queue(20, root)
# #     Q.enqueue(root)
# #
# #     while (Q.isEmpty() == False and not Q.isFull()):
# #         size = Q.queueSize()
# #         for i in range(size):
# #             temp = Q.dequeue()
# #             if temp.grad is None:             # if is the first node
# #                 if temp.name == 'Dense':
# #                     temp.grad = temp.weight_bias.T
# #                     temp.compute_gradient()
# #                     if temp.last_left != None and temp.last_left.require_grad:
# #                         temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
# #                         Q.enqueue(temp.last_left)
# #                 else:
# #                     temp.grad = np.ones(temp.value.shape)
# #                     temp.compute_gradient()
# #                     if temp.last_left != None and temp.last_left.require_grad:
# #                         temp.last_left.grad = temp.last_left.sub_grad
# #                         Q.enqueue(temp.last_left)
# #                     if temp.last_right != None and temp.last_right.require_grad:
# #                         temp.last_right.grad = temp.last_right.sub_grad
# #                         Q.enqueue(temp.last_right)
# #             else:
# #                 temp.compute_gradient()
# #                 if temp.last_left != None and temp.last_left.require_grad:
# #                     temp.last_left.grad = chain_rule(temp.grad, temp.last_left.sub_grad, 'l')
# #                     Q.enqueue(temp.last_left)
# #                 if temp.last_right != None and temp.last_right.require_grad:
# #                     temp.last_right.grad = chain_rule(temp.grad, temp.last_right.sub_grad, 'r')
# #                     Q.enqueue(temp.last_right)
#


