//
// Created by jin_h on 2020/6/10.
//

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <malloc.h>

// check if the char array is Palindromep
bool isPalindrome(char A[], int len) {
    int mid_index;
    int i = 0;
    int j = len - 1;
    if (len < 1) {
        return false;
    } else if (len == 1) {
        return true;
    }
    if (len % 2 != 0) {
        mid_index = (len - 1) / 2;
        while (i < mid_index) {
            if (A[i] != A[j]) {
                return false;
            } else {
                i++;
                j--;
            }
        }
        return true;
    } else {
        mid_index = (len - 2) / 2;
        while (i <= mid_index) {
            if (A[i] != A[j]) {
                return false;
            } else {
                i++;
                j--;
            }
        }
        return true;
    }
}

int main() {
//    char *A;
//    A = (char *) malloc(sizeof(char) * 1000);
    char A[]={};
    int len;
    int result;
    printf("Enter a word: ");
//    scanf("%s", A);
    scanf("%c", A);
    len = strlen(A);
    result = isPalindrome(A, len);
    if (result) {
        printf("yes");
    } else {
        printf("no");
    }
}