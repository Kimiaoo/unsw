// @file: popularityRank.py
// @author: Hongxiao Jin
// @creat_time: 2020/7/2 11:41

# include <stdio.h>
# include <malloc.h>
# include <string.h>
# include <stdlib.h>
# include "WGraph.h"

void insertionSort(float array[], int n) {
    int i;
    for (i = 1; i < n; i++) {
        float element = array[i];                 // for this element ...
        int j = i - 1;
        while (j >= 0 && array[j] > element) {  // ... work down the ordered list
            array[j + 1] = array[j];               // ... moving elements up
            j--;
        }
        array[j + 1] = element;                   // and insert in correct position
    }
}

void popularityRank(Graph graph) {
    int nV = numOfVertices(graph);
    float inDegree[nV];
    float outDegree[nV];
    float popularity[nV];
    float rank[nV];

    // count outDegree
    for (int v = 0; v < nV; v++) {
        outDegree[v] = 0;
        inDegree[v] = 0;
        for (int w = 0; w < nV; w++) {
            if (adjacent(graph, v, w)) {
                outDegree[v]++;
            }
            if (adjacent(graph, w, v)) {
                inDegree[v]++;
            }
        }
        if (outDegree[v] == 0) {
            outDegree[v] = 0.5;
        }
    }

    for (int v = 0; v < nV; v++) {
        popularity[v] = inDegree[v] / outDegree[v];
        rank[v] = popularity[v];
    }

    insertionSort(rank, nV);
    if (nV > 0) {
        printf("\nPopularity ranking:\n");
    }
    float temp = -1; // check it the popularity is the same as before
    for (int i = nV - 1; i >= 0; i--) {
        for (int j = 0; j < nV; j++) {
            if (popularity[j] == rank[i] && temp != rank[i]) {
                printf("%d %.1f\n", j, popularity[j]);
            }
        }
        temp = rank[i];
    }
}

// check the input if it is an Integer
int check_input(char *nums) {
    int len = strlen(nums);
    for (int i = nums[0] == '-' ? 1 : 0; i < len; i++) {
        if (nums[i] < '0' || nums[i] > '9') {
            return 0;
        }
    }
    return 1;
}

int main() {
    int n;  // the number of vertices
    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    Graph gra = newGraph(n);
    Edge edge;
    edge.weight = 1;
    int flag = 1;

    char *num;
    num = (char *) malloc(sizeof(char) * 1000000);

    if (n <= 0){
        printf("Done.\n");
    }else{
        printf("Enter an edge (from): ");
        while (flag == 1) {
            scanf("%s", num);
            flag = check_input(num);
            if (flag == 1) {
                edge.v = atoi(num);// turn the input to a digital
                printf("Enter an edge (to): ");
                scanf("%s", num);
                flag = check_input(num); // check if it is an Integer
                if (flag == 1) {
                    edge.w = atoi(num);
                    insertEdge(gra, edge);
                    printf("Enter an edge (from): ");
                } else {
                    printf("Done.\n");
                }
            } else {
                printf("Done.\n");
            }
        }
    }


    popularityRank(gra);
    free(num);
    freeGraph(gra);
}