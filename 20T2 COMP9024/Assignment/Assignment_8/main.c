//
// Created by jin_h on 2020/7/26.
//

#include <stdio.h>
#include "RBTree.h"

int main() {
    int i, k,n;
    Tree t = newTree();
    printf("Please enter the number of nodes: ");
    scanf("%d", &n);
    int nodes[n];
    for (i = 0; i < n; i++) {
        scanf("%d", &k);
        nodes[i] = k;
        printf("%d\n", nodes[i]);
    }
    for (i = 0; i < n; i++) {
        t = TreeInsert(t, nodes[i]);
        showTree(t);
    }

    freeTree(t);
}