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

class myclass {
public:
    myclass(char* name) {
        myname = new char[5];
        p = myname;
        while (*name != '\0') {
            *myname = *name;
            ++name;
            ++myname;
        }
    }
    char* myname;
    char* p;
};

class Solution2 {
public:
    bool isValid(string s) {
        int len = s.length();
        if (len % 2 != 0) return false;
        vector<char> vec1;
        vector<char> vec2;
        for (int i = 0; i < len; i++) {
            if (i > len / 2 - 1) {
                vec2.push_back(s[i]);
            }
            else {
                vec1.push_back(s[i]);
            }
        }
        for (int j = 0; j < len / 2; j++) {
            if (vec1[j] != vec2[len - j - 1]) {
                return false;
            }
        }
        return true;
    }
};

class Solution3 {
private:
    const string letterMap[10] = { "", " ", "abc", "def", "ghi", "jkl", "mno", "pqrs","tuv","wxyz" };
    vector<string> vec;
    string s;
    void backtrack(int k, int size, int& total_size, string& digits) {
        if (k == total_size) {
            vec.push_back(s);
            return;
        }
        char num = digits[k];
        for (int i = 0; i < size; i++) {
            s.push_back(letterMap[num - '0'][i]);
            backtrack(k + 1, letterMap[digits[k + 1] - '0'].size(), total_size, digits);
            s.pop_back();
        }

    }
public:
    vector<string> letterCombinations(string digits) {
        int total_size = digits.size();
        backtrack(0, letterMap[digits[0] - '0'].size(), total_size, digits);
        return vec;
    }
};

class Solution4 {
public:
    int totalFruit(vector<int>& fruits) {
        int b1 = fruits[0], b2 = fruits[0];
        int pre = 0, next, max = 0;
        for (next = 0; next < fruits.size(); next++) {
            while (fruits[next] != b1 && fruits[next] != b2) {
                if (b1 != b2) {
                    pre++;
                    b1 = fruits[pre];
                }
                b2 = fruits[next];
            }
            if (max < next - pre + 1) max = next - pre + 1;
        }
        // max = max > next - pre + 1 ? max:next - pre + 1;

        return max == 0 ? fruits.size() : max;
    }
};

class Solution5 {
private:
    bool binsearch(int r, int l, int& target, vector<vector<int>>& matrix, int & row) {
        if (r == l)
            return false;
        int mid = r + (r - l) / 2;
        if (matrix[row][mid] == target) return true;
        if (matrix[row][mid] > target) {
            r = mid + 1;
        }
        else {
            l = mid - 1;
        }
        binsearch(r, l, target, matrix, row);

    }
public:
    bool findelement(int m, int n, int& target, vector<vector<int>>& matrix) {
        int row;
        for (row = 0; row < m; row++) {
            if (matrix[row][0] < target && matrix[row][n - 1]) {
                int row_ind = row;
                break;
            }
        }
        if (row > m) return false;

        return binsearch(0, n, target, matrix, row);
    }
};


int main() {
 /*   vector<int> in = {9,3,15,20,7};
    vector<int> po = { 9,15,7,20,3};
    Solution s;
    TreeNode * p = s.buildTree(in, po);*/


    //char m[] = "12345" ;
    //myclass m1(m);
    //cout << m1.p;
    //return 0;
    vector<vector<int>>  matrix = {{1,4,7,11,15},{2,5,8,12,19},{3,6,9,16,22},{10,13,14,17,24}, {18,21,23,26,30} };
    Solution5 s;
    //vector<string> t;
    int target = 5;
    bool tar = s.findelement(5, 5, target, matrix);
        //cout << t;
    cout << tar;
    
}


