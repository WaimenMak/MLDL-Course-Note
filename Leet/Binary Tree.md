# 二叉树

### 前序遍历

二叉树的前中后遍历使用递归方法非常简单，用迭代法需要用到栈，下面python代码栈是列表 `stack` ，储存了二叉树的节点地址信息。大致思想：每一次遍历到一个节点判断是否为根节点，根节点则输出当前值，储存右孩子地址到栈，指向左孩子，重复以上操作，直到遇到叶子节点。

碰到叶子节点，输出当前内容，指针指向栈顶储存的最近的一个右子树的地址：

```python
else:  
    vec.append(root.val)
    if (len(stack) != 0):
        root = stack[-1]
        stack.pop()
    else:    
        root = None
```

有一种情况，当前为叶子节点，但栈为空，就是没有右子树，直接让指针指向`NULL`，`while` 的判断就会中断，遍历完成。

迭代方法二叉树前序遍历（python ver）：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        vec = []
        if (root == None):
            return vec
        stack = []
        while(root != None or len(stack) != 0):
            if (self.isRoot(root)):
                vec.append(root.val)
                if (root.right != None):
                    stack.append(root.right)
                if (root.left != None):
                    root = root.left
                else:
                    root = stack[-1];
                    stack.pop()
            else: 
                vec.append(root.val)
                if (len(stack) != 0):
                    root = stack[-1]
                    stack.pop()
                else:    
                    root = None
        return vec

    
            
    def isRoot(self, root):
        if (root.left != None or root.right != None):
            return True
        else:
            return False
```

C语言，**注意**传给 `returnSize`的是一个记录数据长度的整形 `int` 数据的地址：

```C
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */


/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
bool rootisroot(struct TreeNode* root){
    if (root ->left != NULL || root ->right != NULL)
    {
        return true;
    }
    return false;
};
//define a stack
struct stack {
    int top;
    struct TreeNode* arr[100];
};

int* preorderTraversal(struct TreeNode* root, int* returnSize){
    int * vec = (int *)malloc(100*sizeof(int));
    int i = 0;
    if (root == NULL){
        *returnSize = i;
        return vec;
    }
    struct stack st;
    st.top = -1;
    //returnSize = &i; //错误写法
    while (st.top!=-1 || root != NULL)
    {
        if (rootisroot(root))
        {
            vec[i] = root->val;
            i++;
            if (root ->right != NULL)
            {
                st.top++;
                st.arr[st.top] = root->right;
            }
            if (root->left != NULL)
                root = root->left;
            else
            {
                root = st.arr[st.top];
                st.top--;
            }

        }
        else 
        {
            if (st.top == -1)
            {
                vec[i] = root->val;
                i++;
                root = NULL;
            }
            else
            {
                vec[i] = root->val;
                i++;
                root = st.arr[st.top];
                st.top--;
            }

        }
    }
    *returnSize = i;
    return vec;
}
```

### 中序遍历

直接放C版本，代码比较冗长，或许不是一个比较好的迭代写法，但是好处在于思想和前序遍历差不多，这次栈内记录根节点的地址，上面是记录根节点右孩子的地址。

```C
bool rootisroot(struct TreeNode* root){
    if (root ->left != NULL || root ->right != NULL)
    {
        return true;
    }
    return false;
};

struct stack {
    int top;
    struct TreeNode* arr[100];
};

int* inorderTraversal(struct TreeNode* root, int* returnSize){
    int * vec = (int *)malloc(100*sizeof(int));
    int i = 0;
    if (root == NULL){
        *returnSize = i;
        return vec;
    }
    struct stack st;
    st.top = -1;

    while (st.top!=-1 || root != NULL)
    {
        if (root == NULL)                      //whether is null or not, if is null and the stack is not empty, the pointer 													//point to stack's top
        {
            root = st.arr[st.top]->right;
            vec[i] = st.arr[st.top]->val;
            i++;
            st.top--;
        }  
        else if (rootisroot(root))   //is root
        {

            if (root ->left!= NULL)
            {
                st.top++;
                st.arr[st.top] = root;  //push
                root = root->left;
            }
            else
            {
            	vec[i] = root->val;
                i++;
                root = root->right;
			}

        }
        else    //is leaf
        {
            if (st.top!=-1)   //the stack is not null.
            {
                vec[i] = root->val;
                i++;
                vec[i] = st.arr[st.top]->val;
                i++;
                root = st.arr[st.top]->right;
                st.top--;        
            }
            else              //this is important, if there is only one node
            {
                vec[i] = root->val;
                i++;
                root = NULL;
            }

        }
    }
    *returnSize = i;
    return vec;
}
```

### 二叉树层序遍历

**示例：**
二叉树：`[3,9,20,null,null,15,7]`,

返回其层序遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```

#### 队列

迭代法加递归法，迭代法需要用到`队列` ，首先回顾队列的特性：1、受限制线性表；2、先进先出（FIFO）；3、两个指针：`rear` 指向队列尾部元素，`front` 指向首部元素；

实际上顺序队列有两个缺点：1. 如果固定前n个位置储存元素，则删除一个元素后面的需要向前移动，操作代价为 $\Theta(n)$ ，2 . 如果不需要维持这个条件，随着`font` 指针增加，队列剩余空间会越来越少，造成假溢出（前面还有空间）。

为解决前两个问题，可以采用循环队列，实际上是假定数组是循环。判断队空或队满会出现矛盾，当`rear ` 在 `font ` 前，既是队空也是队满的表现。因此实现队列的实际大小时可以是默认大小加1，`size+1` 。

这种方式可区分队空队满：1、`(rear+2) % maxsize = front`则 队满，2、`(rear+1) % maxsize = front`为空 `maxsize ` 为队列长度size+1，`rear` 和 `font` 取值从0开始。`front` 初始化为0,即是队首，`rear`初始化为`maxsize-1`即队列最后元素下标

用python实现的循环队列如下：

```python
class queue():
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
```

层序遍历：

- enqueue root

- compute the size of queue

- dequeue the roots in the queue, enqueue the left child and right child of each root.

  解答：

  [1]: https://leetcode-cn.com/problems/binary-tree-level-order-traversal/solution/er-cha-shu-ceng-xu-bian-li-deng-chang-wo-yao-da-sh/	"代码随想录"

```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        arr = []
        Q = queue(140,root)
        if (root == None):
            return arr
        else:
            Q.enqueue(root)

        while (Q.isEmpty() == False):
            element = []
            size = Q.queueSize()
            for i in range(size):
                it = Q.dequeue()
                element.append(it.val)
                if (it.left != None and not Q.isFull()):
                    Q.enqueue(it.left)
                if (it.right != None and not Q.isFull()):
                    Q.enqueue(it.right)

            arr.append(element)
        return arr
```

实际上python的写法可以非常简洁，队列可以用列表实现，出队直接`pop(0)`，上面重复造轮子，算是复习一遍基本数据结构吧。

#### 递归写法

引用某湖南大学老师的总结：

- 递归出口

- 递归体

  貌似需要计算树高度。递归在这里并不是一种很好的做法。

### 二叉树反转

针对二叉树的问题，解题之前一定要想清楚究竟是前中后序遍历，还是层序遍历。

翻转一棵二叉树（226）。

示例：

输入：

     	  4
        /   \
      2     7
     / \   / \
    1   3 6   9

输出：

          4
        /   \
      7     2
     / \   / \
    9   6 3   1
```C
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

bool rootisroot(struct TreeNode* root){
    if (root ->left != NULL || root ->right != NULL)
    {
        return true;
    }
    return false;
};

struct TreeNode* invertTree(struct TreeNode* root){
    struct TreeNode * temp = NULL;
    if (root == NULL)
    {
        return root;
    }
    else if (rootisroot)   //其实不需要判断是否为叶子
    {
        temp = root->left;
        root -> left = root -> right;
        root->right = temp;
        invertTree(root->left);
        invertTree(root->right);
    }
    return root;           //这一步return比较重要，因为最后要返回第一层的根节点
}
```

以上写法为前序遍历，除了中序遍历，其他遍历方法都可以用。

### 二叉树对称

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

```
    1
   / \
  2   2
   \   \
   3    3
```

里外结合，双线操作。

```C
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */

bool compare(struct TreeNode * left, struct TreeNode * right){
    bool outside;
    bool inside;
    if (left == NULL && right == NULL)
        return true;
    else if(left == NULL && right != NULL)
        return false;
    else if (left != NULL && right == NULL)
        return false;
    else if (left-> val != right -> val)
        return false;

    outside = compare(left->left, right->right);
    inside = compare(left->right, right->left);
    return outside && inside;
};

bool isSymmetric(struct TreeNode* root){
    if (root == NULL)
        return true;
    else
    {
        return compare(root->left, root->right);
    }
}
```

关键是传参的地方，之前遍历的递归都是传入一个参数，这里传两个所以有点难理解，相当于并行处理。

迭代法python版本：

```python
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        else:
            Q = queue(50,root)
            Q.enqueue(root.left)
            Q.enqueue(root.right)
            while (Q.isEmpty() == False):
                temp_1 = Q.dequeue()
                temp_2 = Q.dequeue()

                if temp_1 == None and temp_2 == None:
                    continue                          #用continue而不是pass
                elif temp_1 != None and temp_2 == None:
                    return False
                elif temp_1 == None and temp_2 != None:
                    return False
                elif temp_1.val != temp_2.val:
                    return False
                
                Q.enqueue(temp_1.left)
                Q.enqueue(temp_2.right)
                Q.enqueue(temp_1.right)
                Q.enqueue(temp_2.left)

            return True
```

入队的操作和之前层序遍历也不同，一次入队四个节点，每次出队两个，然后比较这两个，所以顺序上是先比较外再比较内。无论递归还是队列，初始情况都是从开头的两个节点开始搞。

### 二叉树最大深度

解法两种：递归+迭代（层序遍历）。

```C
int traverse(struct TreeNode * root, int depth)
{
    if (root == NULL)
    {
        return depth;
    }
    depth += 1;
    
    return fmax(traverse(root->left, depth), traverse(root->right,depth));
}

int maxDepth(struct TreeNode* root){
    int depth = 0;
    depth = traverse(root, depth);
    return depth;
}
```

经典题解：

```CC
class Solution {
public:
    int getDepth(TreeNode* node) {
        if (node == NULL) return 0;
        int leftDepth = getDepth(node->left);       // 左
        int rightDepth = getDepth(node->right);     // 右
        int depth = 1 + max(leftDepth, rightDepth); // 中
        return depth;
    }
    int maxDepth(TreeNode* root) {
        return getDepth(root);
    }
};
```

计算当前节点的左子树和右子树的深度，返回最大深度+1，就是当前节点深度. 同样道理，N叉树可以用同样方法计算最大深度。

[1]: https://mp.weixin.qq.com/s/guKwV-gSNbA1CcbvkMtHBg	"最大深度"

层序遍历可以直接套用模板，在第一个循环里加上`depth+=1`对深度进行记录. `注意` 这里最大深度的求法其实不太符合实际要求，其实这里求的时根节点的高度，所以应该用前序遍历，

```CC
class Solution {
public:
    int result;
    void getDepth(TreeNode* node, int depth) {
        result = depth > result ? depth : result; // 中

        if (node->left == NULL && node->right == NULL) return ;

        if (node->left) { // 左
            depth++;    // 深度+1
            getDepth(node->left, depth);
            depth--;    // 回溯，深度-1
        }
        if (node->right) { // 右
            depth++;    // 深度+1
            getDepth(node->right, depth);
            depth--;    // 回溯，深度-1
        }
        return ;
    }
    int maxDepth(TreeNode* root) {
        result = 0;
        if (root == 0) return result;
        getDepth(root, 1);
        return result;
    }
};
```

**可以看出使用了前序（中左右）的遍历顺序，这才是真正求深度的逻辑！**

### 二叉树平衡

递归里的递归，

```C
int postorder(struct TreeNode* root)
{
    if (root == NULL)
    {
        return 0;
    }
    int a = postorder(root->left);
    int b = postorder(root->right);
    return 1 + fmax(a, b);
}

bool isBalanced(struct TreeNode* root){
    if (root == NULL)
        return true;
    int a = postorder(root->left);
    int b = postorder(root->right);
    if (abs(a - b) > 1)
        return false;
    if (!isBalanced(root->left))
        return false;
    if (!isBalanced(root->right))
        return false;
    return true;
    //return (isBalanced(root->left) && isBalanced(root->right));
}
```

### 二叉树构造

根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
返回如下的二叉树：

```
    3
   / \
  9  20
    /  \
   15   7
```

```c++
class Solution {
private:
    TreeNode* traversal(vector<int>& inorder, vector<int>& postorder){
        if (inorder.size() == 0) return NULL;
        int rootVal = postorder[postorder.size() - 1];
        TreeNode* root = new TreeNode(rootVal);
        int mid;
        for (mid = 0; mid < inorder.size(); mid++){
            if (inorder[mid] == rootVal)
                break;
        }
        if (postorder.size() == 1) return root;
        //inoreder
        vector<int> inorderLeft(inorder.begin(), inorder.begin() + mid);
        vector<int> inorderRight(inorder.begin() + mid + 1, inorder.end());
        //postorder
        //postorder.resize(postorder.size() - 1);
        vector<int> postorderLeft(postorder.begin(), postorder.begin() + mid);
        vector<int> postorderRight (postorder.begin() + mid, postorder.end() - 1);
        //recursive
        root->left = traversal(inorderLeft, postorderLeft);
        root->right = traversal(inorderRight, postorderRight);
        return root;
    }
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        TreeNode* root = traversal(inorder, postorder);
        return root;
    }
};
```

传入traversal的vector的size会一样。

### 构造最大二叉树

给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：

二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。

[654. 最大二叉树 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/maximum-binary-tree/)

```c++
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        if (nums.size() == 0) return NULL;

        if (nums.size() == 1){
            TreeNode* root = new TreeNode(nums[0]);
            return root;
        }

        auto maxPosition = max_element(nums.begin(), nums.end());
        TreeNode* root = new TreeNode(*maxPosition);
        int maxNode;
        for (maxNode = 0; maxNode < nums.size(); maxNode++){
            if (nums[maxNode] == *maxPosition) break;
        }
        vector<int> left(nums.begin(), nums.begin() + maxNode);
        vector<int> right(nums.begin() + maxNode + 1, nums.end());
        root->left = constructMaximumBinaryTree(left);
        root->right = constructMaximumBinaryTree(right);

        return root;
    }
};
```

这题解法与上一题比较相似，但是这种写法有点缺陷，每次重新创造vector会造成额外开销。以上写法是原始写法，在找最大元素时根本不用调用`max_element`,直接遍历找。上面写法造成冗余。

改进如下：

```C++
class Solution {
private:
    // 在左闭右开区间[left, right)，构造二叉树
    TreeNode* traversal(vector<int>& nums, int left, int right) {
        if (left >= right) return nullptr;

        // 分割点下表：maxValueIndex
        int maxValueIndex = left;
        for (int i = left + 1; i < right; ++i) {
            if (nums[i] > nums[maxValueIndex]) maxValueIndex = i;
        }

        TreeNode* root = new TreeNode(nums[maxValueIndex]);

        // 左闭右开：[left, maxValueIndex)
        root->left = traversal(nums, left, maxValueIndex);

        // 左闭右开：[maxValueIndex + 1, right)
        root->right = traversal(nums, maxValueIndex + 1, right);

        return root;
    }
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return traversal(nums, 0, nums.size());
    }
};
```

### 公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

 

示例 1：


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
示例 2：


输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
示例 3：

输入：root = [1,2], p = 1, q = 2
输出：1


提示：

树中节点数目在范围 [2, 105] 内。
-109 <= Node.val <= 109
所有 Node.val 互不相同 。
p != q
p 和 q 均存在于给定的二叉树中。
链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree

```C
//后序遍历
class Solution {
private:
    int find = 0;
    TreeNode * ptr = NULL;
    bool check(TreeNode * cur, TreeNode* p, TreeNode* q, bool ind_l, bool ind_r){
        if (find == 2 && ptr != NULL){   //already found, 不作为
            return true;
        }
        else if (ind_l == true && ind_r == true){  //左右子树分别都找到，则当前节点为祖先。
            ptr = cur;
            return true;
        }
        else if (ind_l == true || ind_r == true){  //其中一颗子树找到，假如当前节点找到则当前节点为祖先
            if (cur->val == p->val || cur->val == q->val)
            {
                find += 1;
                ptr = cur;
            }
            return true;
        }
        else if (cur->val == p->val || cur->val == q->val){  //if not found in the subtree, and current node is the target.
            find += 1;
            return true;
        }
        return false;   //not found the target yet.
    }
    bool traversal(TreeNode * cur, TreeNode* p, TreeNode* q){
        if (find == 2 || cur == NULL) return false;
        bool ind_l = traversal(cur->left, p, q);
        bool ind_r = traversal(cur->right, p, q);
        bool ind = check(cur, p, q, ind_l, ind_r);
        return ind;
        
    }
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        traversal(root, p, q);
        return ptr;
    }
};
```

此类题目，确定好遍历顺序后，分析好题目看看分多少情况处理即可。关键在于逻辑的处理。
