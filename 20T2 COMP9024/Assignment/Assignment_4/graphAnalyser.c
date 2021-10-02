// @file: graphAnalyser.py
// @author:Hongxiao Jin
// @creat_time: 2020/6/27 23:30

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
# include "Graph.h"


void count_degree(Graph graph) {
    int nV = numOfVertices(graph);
    int count[nV];

    int min_degree = nV;
    int max_degree = 0;

    for (int v = 0; v < nV; v++) {
        count[v] = 0;
        for (int w = 0; w < nV; w++) {
            if (adjacent(graph, v, w)) {
                count[v]++;
            }
        }
    }

    for (int i = 0; i < nV; i++) {
        if (count[i] < min_degree) {
            min_degree = count[i];
        }
        if (count[i] > max_degree) {
            max_degree = count[i];
        }
    }

    printf("Minimum degree: %d\n", min_degree);
    printf("Maximum degree: %d\n", max_degree);

    printf("Nodes of minimum degree:\n");
    for (int i = 0; i < nV; i++) {
        if (count[i] == min_degree) {
            printf("%d\n", i);
        }
    }
    printf("Nodes of maximum degree:\n");
    for (int i = 0; i < nV; i++) {
        if (count[i] == max_degree) {
            printf("%d\n", i);
        }
    }
}

void show_triangles(Graph graph) {
    int nV = numOfVertices(graph);
    printf("Triangles:\n");
    for (int v = 0; v < nV; v++) {
        for (int w = 0; w < nV; w++) {
            for (int z = 0; z < nV; z++) {
                if (v < w && w < z) {
                    if (adjacent(graph, v, w) && adjacent(graph, v, z) && adjacent(graph, z, w)) {
                        printf("%d-%d-%d\n", v, w, z);
                    }
                }
            }
        }
    }
}

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
    int flag = 1;

    char *num;
    num = (char *) malloc(sizeof(char) * 1000000);

    printf("Enter an edge (from): ");
    while (flag == 1) {
        scanf("%s", num);
        flag = check_input(num);
        if (flag == 1) {
            edge.v = atoi(num);
            printf("Enter an edge (to): ");
            scanf("%s", num);
            flag = check_input(num);
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
    count_degree(gra);
    show_triangles(gra);
    freeGraph(gra);
    free(num);
    return 0;
}


