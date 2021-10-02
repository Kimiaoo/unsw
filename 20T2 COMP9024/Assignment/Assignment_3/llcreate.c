//
// Created by jin_h on 2020/6/17.
//

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

// define struct node
typedef struct node {
    int data;
    struct node *next;
} NodeT;

void freeLL(NodeT *list) {
    NodeT *p, *temp;
    p = list;
    while (p != NULL) {
        temp = p->next;
        free(p);
        p = temp;
    }
}

// Print all elements
void showLL(NodeT *list) {
    NodeT *p;
    for (p = list; p != NULL; p = p->next) {
        if (p->next == NULL) { // if the node is the last one, we do not need to print -->
            printf("%d", p->data);
        } else {
            printf("%d-->", p->data);
        };
    }
}

// append a new element with data v at the end of list
NodeT *joinLL(NodeT *list, int v) {
    // initialise a new node
    NodeT *new = malloc(sizeof(NodeT));
    new->data = v;
    new->next = NULL;

    NodeT *p = malloc(sizeof(NodeT));
    p = list;
    while (p->next) {
        p = p->next;
    }
    p->next = new;
    return list;
}


int main() {
    // initialise a new node
    NodeT *all = malloc(sizeof(NodeT));
    all->next = NULL;

    char *num;
    int n;
    int len;
    // record if it is an integer, 1 means it is an integer, 0 means it is not an integer
    int flag = 1;
    // record if it is the first NUM, 1 means it has a first number, 0 means it does not have a first number
    int check_first = 0;
    num = (char *) malloc(sizeof(char) * 1000000);

    while (flag == 1) {
        printf("Enter an integer: ");
        scanf("%s", num);
        len = strlen(num);

        // if the input is not an integer, set flag = 0 -> showLL
        //for (int i = num[0] == '-' ? 1 : 0; i < len; i++) {
        for (int i = 0; i < len; i++) {
            if (num[i] < '0' || num[i] > '9') {
                flag = 0;
            }
        }
        if (flag == 1) {
            n = atoi(num);
            if (check_first == 0) {
                all->data = n;
                check_first = 1;
            } else {
                all = joinLL(all, n);
            }
        } else {
            if (all->next != NULL || check_first) {
                printf("Done. List is ");
                showLL(all);
            } else {
                printf("Done.");
            }
        }
    }
    freeLL(all);
    return 0;
}