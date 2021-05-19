//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
//#include "Nodes.h"
#include<vector>
#include <iostream>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
    
};
class Solution {
private:
    TreeNode* traversal(vector<int>& inorder, vector<int>& postorder) {
        if (inorder.size() == 0) return NULL;
        int rootVal = postorder[postorder.size() - 1];
        TreeNode* root = new TreeNode(rootVal);
        int mid;
        for (mid = 0; mid < inorder.size(); mid++) {
            if (inorder[mid] == rootVal)
                break;
        }
        //inoreder
        vector<int> inorderLeft(inorder.begin(), inorder.begin() + mid);
        vector<int> inorderRight(inorder.begin() + mid + 1, inorder.end());
        //postorder
        //postorder.resize(postorder.size() - 1);
        vector<int> postorderLeft(postorder.begin(), postorder.begin() + mid);
        vector<int> postorderRight(postorder.begin() + mid, postorder.end() - 1);
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

int main() {
    vector<int> in = {9,3,15,20,7};
    vector<int> po = { 9,15,7,20,3};
    Solution s;
    TreeNode * p = s.buildTree(in, po);

    return 0;
}


