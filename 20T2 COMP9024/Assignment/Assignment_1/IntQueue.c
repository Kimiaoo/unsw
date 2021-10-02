//
// Created by jin_h on 2020/6/8.
//

#include "IntQueue.h"
#include <assert.h>

// define the Data Structure
typedef struct {
    int item[MAXITEMS];
    // the first place in item
    int front;
    // the last place in item
    int rear;
    // the number of integer in queue
    int size;
} intQueueRep;

// define the Data Object
static intQueueRep intQueueObject;

// set up empty queue
void QueueInit() {
    intQueueObject.front = -1;
    intQueueObject.rear = -1;
    intQueueObject.size = 0;
}

// check whether queue is empty
int QueueIsEmpty() {
    return intQueueObject.size ? 0 : 1;
}

// insert int at end of queue
void QueueEnqueue(int n) {
    assert(intQueueObject.rear < MAXITEMS - 1);
    intQueueObject.rear++;
    int i = intQueueObject.rear;
    intQueueObject.item[i] = n;
    intQueueObject.size++;
}

// remove int from front of queue
int QueueDequeue() {
    assert(intQueueObject.front < intQueueObject.rear);
    intQueueObject.front++;
    int i = intQueueObject.front;
    int n = intQueueObject.item[i];
    intQueueObject.size--;
    return n;
}