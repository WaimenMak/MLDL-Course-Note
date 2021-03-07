#include <stdio.h> 
#include<malloc.h>
#include <stdio.h>
#include<stdbool.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

//struct TreeNode
//bool rootisroot(struct TreeNode* root){
//    if (root ->left != NULL || root ->right != NULL)
//    {
//        return true;
//    }
//    return false;
//};
//
//struct stack {
//    int top;
//    struct TreeNode* arr[100];
//};
//
//int* preorderTraversal(struct TreeNode* root, int* returnSize){
//    if (root == NULL){
//        return NULL;
//    }
//    struct stack st;
//    st.top = -1;
//    int vec[100];
//    vec[0] = 1;
//    int i = 0;
//    returnSize = &i;
//    while (st.top!=-2 && root != NULL)
//    {
//        if (rootisroot(root))
//        {
//            // st.top++;
//            // st.arr[st.top] = root;    //push
//            vec[i] = root->val;
//            i++;
//            // st.top--;   //pop
//            if (root ->right != NULL)
//            {
//                st.top++;
//                st.arr[st.top] = root->right;
//            }
//            // st.top++;
//            // st.arr[st.top] = root->left;
//            if (root->left != NULL)
//                root = root->left;
//            else
//                root = st.arr[st.top];
//                st.top--;
//        }
//        // else if (root == NULL)
//        // {
//        //     root = st.arr[st.top];
//        //     st.top--;
//        // }
//        else 
//        {
//            vec[i] = root->val;
//            i++;
//            root = st.arr[st.top];
//            st.top--;
//        }
//    }
//    root = NULL;
//    return vec;
//}

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
    if (root == NULL){
        return vec;
    }
    struct stack st;
    st.top = -1;
    vec[0] = 1;
    int i = 0;
    while (st.top!=-1 || root != NULL)
    {
        if (rootisroot(root))   //is root
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
            else
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

void main()
{
	struct TreeNode * inpt[100];
	int length;
	int i,len = 5;
	int *p,*p2;
//	int inpt_arr[len];
//	int_arr[] = {1,2,3}
	inpt[0] =  (struct TreeNode *)malloc(sizeof(struct TreeNode));
	inpt[0]->left = NULL;
	inpt[0]->right = NULL;
	inpt[1] = NULL;
	inpt[2] = (struct TreeNode *)malloc(sizeof(struct TreeNode));
	inpt[2]->left = NULL;
	inpt[2]->right = NULL;
	inpt[3] = (struct TreeNode *)malloc(sizeof(struct TreeNode));
	inpt[3]->left = NULL;
	inpt[3]->right = NULL;
	
	inpt[0]->val = 1;
	inpt[0]->right = inpt[2];
	inpt[2]->val = 2;
	inpt[2]->left = inpt[3];
	inpt[3]->val = 3;
	
//	printf("%d", inpt[0]->val);
	//connect
//	length=sizeof(inpt)/sizeof(inpt[0]);
	length = 4;
//	printf("%d",length);
//	for (i = 0; i <= length - 2; i++)
//	{
//		if (inpt[i] != NULL)
//		{
//			inpt[i]->left = inpt[i+1];
//			if (i+2 <= length)
//				inpt[i]->right = inpt[i+2];
//		}
//	}
//	printf("%d", inpt[0]->right->val);
	p2 = inorderTraversal(inpt[0], &i);
	printf("%d\n",p2[2]);
	printf("%d",i);
}
 

