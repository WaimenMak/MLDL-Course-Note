# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class queue():
    def __init__(self,size,root):
        self.que = []
        self.maxSize = size + 1
        for i in range(self.maxSize):
            self.que.append(root)
        self.front = 0
        self.rear = self.maxSize - 1

    def enqueue(self,val):
        self.que[(self.rear+1) % self.maxSize] = val
        self.rear = (self.rear+1) % self.maxSize

    def dequeue(self):
        temp = self.que[(self.front) % self.maxSize]
        self.front = (self.front+1) % self.maxSize
        return temp

    def isEmpty(self):
        if ((self.rear+1) % self.maxSize == self.front):
            return True
        else:
            return False

    def isFull(self):
        if ((self.rear+2)% self.maxSize == self.front):
            return True
        else:
            return False
    def queueSize(self):
        if (self.rear >= self.front):
            size = self.rear - self.front + 1
        else:
            size = self.maxSize - self.front + self.rear + 1  
        return size


def levelOrder(root: TreeNode):
    arr = []
    Q = queue(3,root)
    if (root == None):
        return arr
    else:
        Q.enqueue(root)

    while (Q.isEmpty() == False):
        size = Q.queueSize()
        element = []
        for i in range(size):
            it = Q.dequeue()
            element.append(it.val)
            if (it.left != None and not Q.isFull()):
                Q.enqueue(it.left)
            if (it.right != None and not Q.isFull()):
                Q.enqueue(it.right)

        arr.append(element)
    return arr



a = TreeNode()
b = TreeNode()
c = TreeNode()
d = TreeNode()
e = TreeNode()
f = TreeNode()
a.val = 1
a.right = c
a.left = b
b.val = 2
c.val = 3
c.left = f
c.right = d
d.val = 5
d.right = e
e.val = 6
f.val = 4


arr = levelOrder(a)
print(arr)