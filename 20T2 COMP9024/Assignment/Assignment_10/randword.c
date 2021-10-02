// @file: randword.c
// Created by jin_h on 2020/8/4.

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int random; // the random ASCII of character

    srand(atoi(argv[2]));

    for (int i = 0; i < atoi(argv[1]); i++) {
        random = rand() % 26 + (int) 'a';
        printf("%c", (char) random);
    }

    return 0;
}