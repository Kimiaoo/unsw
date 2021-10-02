//
// Created by jin_h on 2020/6/8.
//
#include <stdio.h>
#include <stdlib.h>
#include "IntQueue.h"

int main(void) {
    int i, n;
    char str[BUFSIZ];

    QueueInit();
    printf("Enter a positive number: ");
    scanf("%s", str);

    if ((n = atoi(str)) > 0) {
        for (i = 0; i < n; i++) {
            printf("Enter a number: ");
            scanf("%s", str);
            QueueEnqueue(atoi(str));
        }
    }
    while (!QueueIsEmpty()) {
        printf("%d", QueueDequeue());
    }

    return 0;
}


