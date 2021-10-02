//
// Created by jin_h on 2020/6/7.
//

#include <stdio.h>
#include "IntStack.h"

#define k 2

int main() {
    int n;
    int res;
    //get the number
    printf("Enter a number: ");
    scanf("%d", &n);
    // set up empty stack
    StackInit();
    // insert integer on top of stack
    while (n > 0) {
        StackPush(n % k);
        n = n / k;
    }
    //if the stack is not empty, then print the result
    while (!StackIsEmpty()) {
        res = StackPop();
        printf("%d", res);
    }
}