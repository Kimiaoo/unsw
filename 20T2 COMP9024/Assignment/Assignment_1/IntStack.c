//
// Created by jin_h on 2020/6/8.
//

#include "IntStack.h"
#include <assert.h>

// define the Data Structure
typedef struct {
    int item[MAXITEMS];
    int top;
} intStackRep;

// define the Data Object
static intStackRep intStackObject;

// set up empty stack
void StackInit() {
    intStackObject.top = -1;
}

// check whether stack is empty
int StackIsEmpty() {
    return (intStackObject.top < 0);
}

// insert int on top of stack
void StackPush(int n) {
    assert(intStackObject.top < MAXITEMS - 1);
    intStackObject.top++;
    int i = intStackObject.top;
    intStackObject.item[i] = n;
}

// remove int from top of stack
int StackPop() {
    assert(intStackObject.top > -1);
    int i = intStackObject.top;
    int n = intStackObject.item[i];
    intStackObject.top--;
    return n;
}