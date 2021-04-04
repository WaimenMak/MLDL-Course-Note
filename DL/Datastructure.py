class Queue():
    def __init__(self,size,root):
        self.que = []
        self.maxSize = size+1
        for i in range(self.maxSize):      #我的理解队列是数组，事先分配好一定内存的数组,注意数组预定大小是size+1，之前一直跑不通因为用了size
            self.que.append(root)
        
        self.front = 0
        self.rear = self.maxSize-1

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
        
    def queueSize(self):     #计算队列大小
        if (self.rear >= self.front):
            size = self.rear - self.front + 1
        else:
            size = self.maxSize - self.front + self.rear + 1  
        return size