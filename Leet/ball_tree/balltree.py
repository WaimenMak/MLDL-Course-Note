import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Node():
    def __init__(self):
        self.centroid = None
        self.radius = 0
        self.data_size = 0
        self.data = None
        self.left = None
        self.right = None
        self.isleaf = False

class BallTree():
    def __init__(self):
        self.root = Node()


    def create_Tree(self, root, data, N):
        if len(data) <= N:
            # root = Node()
            root.data = np.array(data)
            centroid = np.mean(data, axis=0)
            root.centroid = centroid
            p1 = max_point(data, centroid)
            root.radius = np.linalg.norm(centroid - p1)
            root.isleaf = True
            return root

        if isinstance(data, list):
            root.data = np.array(data)

        centroid = np.mean(data, axis=0)
        root.centroid = centroid
        root.left = Node()
        root.right = Node()

        p1 = max_point(data, centroid)
        root.radius = np.linalg.norm(centroid - p1)
        p2 = max_point(data, p1)
        r_data, l_data = split_data(data, p1, p2)
        del root.data
        self.create_Tree(root.left, l_data, N)
        self.create_Tree(root.right, r_data, N)

        return root

def max_point(data, centroid):
    # mat = np.concatenate(data)
    mat = data
    dist = np.linalg.norm(mat - centroid, axis=1)
    i = np.argmax(dist)
    return data[i]

def split_data(data, p1, p2):
    # mat = np.concatenate(data)
    mat = data
    dist1 = np.linalg.norm(mat - p1, axis=1)
    dist2 = np.linalg.norm(mat - p2, axis=1)
    r_data = list()
    l_data = list()
    for i in range(len(data)):
        if dist1[i] > dist2[i]:
            r_data.append(data[i])
        else:
            l_data.append(data[i])

    # right.data = np.concatenate(right.data, axis=0)
    # left.data = np.concatenate(left.data, axis=0)
    return r_data, l_data

# def create_Tree(root, data, N):
#     if len(data) <= N:
#         # root = Node()
#         root.data = np.array(data)
#         centroid = np.mean(data, axis=0)
#         root.centroid = centroid
#         p1 = max_point(data, centroid)
#         root.radius = np.linalg.norm(centroid - p1)
#         root.isleaf = True
#         return root
#
#     if isinstance(data, list):
#         root.data = np.array(data)
#
#     centroid = np.mean(data, axis = 0)
#     root.centroid = centroid
#     root.left = Node()
#     root.right = Node()
#
#     p1 = max_point(data, centroid)
#     root.radius = np.linalg.norm(centroid - p1)
#     p2 = max_point(data, p1)
#     r_data, l_data = split_data(data, p1, p2)
#     del root.data
#     create_Tree(root.left, l_data, N)
#     create_Tree(root.right, r_data, N)
#
#     return root

def dist(query, R):
    dist = np.linalg.norm((query - R.centroid)) - R.radius
    return max(dist, 0)

def filter_n_refinement(query, rge, tree, set):
    if tree == None:
        return

    if rge >= dist(query, tree):
        if tree.isleaf:
            l = len(tree.data)
            for i in range(l):
                if np.linalg.norm(query - tree.data[i, :]) < rge:
                    set.append(tree.data[i, :])

        filter_n_refinement(query, rge, tree.left, set)
        filter_n_refinement(query, rge, tree.right, set)

    return

def plot(ax, root):
    if root == None:
        return
    if root.isleaf:
        circle = Circle(xy=(root.centroid[0], root.centroid[1]), radius=root.radius, alpha=0.1)
        ax.add_patch(circle)
    # circle = Circle(xy=(root.centroid[0], root.centroid[1]), radius=root.radius, alpha=0.1)
    # ax.add_patch(circle)
    plot(ax, root.left)
    plot(ax, root.right)


if __name__ == "__main__":
    # data = [np.array([2,3]),np.array([5,4]),np.array([9,6]),np.array([4,7]),np.array([8,1]),np.array([7,2])]
    np.random.seed(1)
    data = np.random.rand(100,2)
    balltree = BallTree()
    balltree.create_Tree(balltree.root, data, 5)
    query = np.array([0.5, 0.5])
    s = list()
    rge = 0.3
    filter_n_refinement(query, rge, balltree.root, s)
    print(s)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot(ax, balltree.root)
    circle = Circle(xy=(query[0], query[1]), radius=rge, alpha=0.1, color='r')
    ax.add_patch(circle)

    d = np.array(data)
    plt.plot(d[:,0], d[:,1], 'b.')
    plt.plot(query[0], query[1], 'r.')
    for p in s:
        plt.plot(p[0], p[1], 'r.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

