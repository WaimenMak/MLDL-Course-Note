class Node():
    def __init__(self):
        self.last_right = None
        self.last_left = None
        self.next = None
        self.value = 0
        self.grad = 0
        self.require_grad = False

class Input(Node):
    def __init__(self, shape, X):
        super().__init__(self)
        self.require_grad = 1

    def output(self):
        pass

    def gradient(self):
        pass


class Add(Node):
    def __init__(self, left_node, right_node):
        super().__init__(self)
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self


class Mul(Node):
    def __init__(self, left_node, right_node):
        super().__init__(self)
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self

class Dense(Node):
    def __init__(self, left_node, right_node):
        super().__init__(self)
        self.last_left = left_node
        self.last_right = right_node
        left_node.next = self
        right_node.next = self
