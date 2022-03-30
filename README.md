# Algorithmic Training

Data structure and algorithm. Coding exercise.

### Ball Tree

![](README.assets/balltree.png)

```python
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
```

