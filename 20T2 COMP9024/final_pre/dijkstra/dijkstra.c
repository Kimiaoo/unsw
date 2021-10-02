// @file: dijkstra.c
// @author: Hongxiao Jin
// @creat_time: 2020/7/2 11:46

// Starting code for Dijkstra's algorithm ... COMP9024 20T2

#include <stdio.h>
#include <stdbool.h>
#include "PQueue.h"

#define VERY_HIGH_VALUE 999999

void display(int pre[], int ver, int start) {
    if (ver != start) {
        display(pre, pre[ver], start);
        printf("-%d", ver);
    } else {
        printf("%d", start);
    }
}

void dijkstraSSSP(Graph g, Vertex source) {
    int dist[MAX_NODES];
    int pred[MAX_NODES];
    bool vSet[MAX_NODES];  // vSet[v] = true <=> v has not been processed
    int s;

    PQueueInit();
    int nV = numOfVertices(g);
    for (s = 0; s < nV; s++) {
        joinPQueue(s);
        dist[s] = VERY_HIGH_VALUE;
        pred[s] = -1;
        vSet[s] = true;
    }
    dist[source] = 0;

    /* NEEDS TO BE COMPLETED */
    int v;
    for (v = 0; v < nV; v++) {
        if (adjacent(g, source, v)) {
            dist[v] = adjacent(g, source, v);
            pred[v] = source;
        }
    }
    dist[source] = 0;
    pred[source] = source;
    vSet[source] = false;
    int min;
    s = leavePQueue(dist);
    while (!PQueueIsEmpty()) {
        for (v = 0; v < nV; v++) {
            min = dist[v];
            if (adjacent(g, s, v) && vSet[v]) {
                if (adjacent(g, s, v) + dist[s] <= min) {
                    min = adjacent(g, s, v) + dist[s];
                    dist[v] = min;
                    pred[v] = s;
                }
            }
        }
        s = leavePQueue(dist);
        vSet[s] = false;

        pred[source] = -1;
        printf("v          ");
        for (v = 0; v < nV; v++) {
            printf("%d      ", v);
        }
        printf("\ndist[v]    ");
        for (v = 0; v < nV; v++) {
            printf("%d      ", dist[v]);
        }
        printf("\npred[v]    ");
        for (v = 0; v < nV; v++) {
            printf("%d      ", pred[v]);
        }
        printf("\n-------------------------\n");
    }

    pred[source] = source;
    for (v = 0; v < nV; v++) {
        if (pred[v] != -1) {
            printf("%d: distance = %d, shortest path: ", v, dist[v]);
            if (v == source) {
                printf("%d", source);
            } else {
                display(pred, v, source);
            }
            printf("\n");
        } else {
            printf("%d: no path\n", v);
        }
    }
}

void reverseEdge(Edge *e) {
    Vertex temp = e->v;
    e->v = e->w;
    e->w = temp;
}

int main(void) {
    Edge e;
    int n, source;

    printf("Enter the number of vertices: ");
    scanf("%d", &n);
    Graph g = newGraph(n);

    printf("Enter the source node: ");
    scanf("%d", &source);
    printf("Enter an edge (from): ");
    while (scanf("%d", &e.v) == 1) {
        printf("Enter an edge (to): ");
        scanf("%d", &e.w);
        printf("Enter the weight: ");
        scanf("%d", &e.weight);
        insertEdge(g, e);
        reverseEdge(&e);               // ensure to add edge in both directions
        insertEdge(g, e);
        printf("Enter an edge (from): ");
    }
    printf("Done.\n");

    dijkstraSSSP(g, source);
    freeGraph(g);
    return 0;
}