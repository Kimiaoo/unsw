// @file: cycleCheck.py
// @author: Hongxiao Jin
// @creat_time: 2020/6/28 12:47

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "Graph.h"

# define MAXIMUM 1000

int visited[MAXIMUM];

bool dfs_cycle_check(Graph graph, Vertex v) {
    visited[v] = 0;
    for (int w = 0; w < numOfVertices(graph); w++) {
        if (adjacent(graph, v, w)) {
            if (visited[w] == -1) {
                dfs_cycle_check(graph, w);
            } else if (visited[w] == 0) {
                return true;
            }
        }
    }
    visited[v] = 1;
    return false;
}

bool hasCycle(Graph g) {
    Vertex v = 0;
    for (v = 0; v < numOfVertices(g); v++) {
        visited[v] = -1;
    }
    for (v = 0; v < numOfVertices(g); v++) {
        if (dfs_cycle_check(g, v)) {
            return true;
        }
        for (int x = 0; x < numOfVertices(g); x++) {
            if (visited[x] != -1) {
                visited[x] = -1;
            }
        }
    }


    return false;
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

    if (hasCycle(gra)) {
        printf("The graph has a cycle.");
    } else {
        printf("The graph is acyclic.");
    }
    freeGraph(gra);
    free(num);
    return 0;
}