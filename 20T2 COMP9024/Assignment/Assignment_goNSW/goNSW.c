// @file: goNSW.c
// @author: Hongxiao Jin
// @creat_time: 2020/7/15 17:53

/*
 * time complexity analysis for my program
 *
 * As we know from Assignment.pdf:
 * 1. n: the number of stops,
 * 2. m: the number of schedules
 * 3. k: the maximum number k of stops on a single train, bus or light rail line.
 *
 * there are mk+2 Vertex in my graph
 * getting all stop names costs O(n)
 * In function get_sor_des_index(), the complexity is O(mk+2)
 * In dijkstra algorithm, the complexity is O((mk+2)^2)
 * dfs function costs O(mk+2)
 *
 * Overall the total complexity is around O((mk)^2)
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <malloc.h>
#include "PQueue.h"
#include "stack.h"

#define VERY_HIGH_VALUE 999999

typedef struct Schedules {
    int line; // line number: show the stop in which schedule
    int stop_index; // record the index in different lines, the numeber of index = number of stops in all schedules
    char stop_name[32]; // name of the stop
    int time; // departure time
    int check_first_or_last; // check if this stop is the first one or the last one in the schedule, 0 = first, 1 = last
} Schedule;

/*
 * initialize schedule
 */
void initial_schedule(Schedule sch) {
    sch.line = -1;
    sch.stop_index = -1;
    sch.stop_name[0] = '\0';
    sch.time = -1;
    sch.check_first_or_last = -1;
}

/*
 * get index of all the start and finish stop
 *
 * time complexity: O(mk+2)
 */
int get_sor_des_index(char name[], Schedule schedule_inform[], int len, int points[], int num, int leave, int n) {
    int i, j;
    if (n == 0) { // n = 0 means find start stops
        for (i = 0, j = 0; i < num && j < len; j++) {
            if (strcmp(schedule_inform[j].stop_name, name) == 0 && schedule_inform[j].time >= leave) {
                points[i] = j;
                i++;
            }
        }
    } else { // n = 1 means find end stops
        for (i = 0, j = 0; i < num && j < len; j++) {
            if (strcmp(schedule_inform[j].stop_name, name) == 0) {
                points[i] = j;
                i++;
            }
        }
    }
    return i;
}

/*
 * get the last index in an array
 *
 * time complexity: O(mk+2)
 */
int get_last_index(const int arr[MAX_NODES]) {
    int i = 0;
    while (arr[i] != -1) {
        i++;
    }
    return i;
}

/*
 * print the result
 *
 * time complexity: O(mk+2)
 */
void display(int **pre, int to, int from, Schedule schedule_inform[], int len_sch) {
    if (to != from) {
        printf("%04d %s\n", schedule_inform[to].time, schedule_inform[to].stop_name);
        if (pre[to][0] < len_sch) {
            if (strcmp(schedule_inform[to].stop_name, schedule_inform[pre[to][0]].stop_name) == 0) {
                printf("Change at %s\n", schedule_inform[to].stop_name);
            }
            display(pre, pre[to][0], from, schedule_inform, len_sch);
        }
    }
}

/*
 * DFS find the latest leave time
 * return the latest leave Vertex
 * reference: algorithm copyright week4 lec non-recursively dfs Pseudocode
 * implement by Hongxiao Jin
 *
 * time complexity: O(mk+2)
 *
 */
int DFS_find_latest_departure_time(Graph gra, const int des[], int len_of_des, int finish, Schedule schedule[]) {
    int visited[numOfVertices(gra)];
    Vertex v, w;
    int latest_time = 0;
    int latest_index = -1;
    for (int i = 0; i < len_of_des; i++) {
        for (v = 0; v < numOfVertices(gra); v++) {
            visited[v] = 0;
        }
        stack view = newStack();
        StackPush(view, finish);
        bool found = false;
        while (found == false && StackIsEmpty(view) == false) {
            v = StackPop(view);
            visited[v] = 1;
            // find latest departure stop
            if (v == des[i]) {
                if (latest_time < schedule[v].time) {
                    latest_time = schedule[v].time;
                    latest_index = v;
                }
                found = true;
            } else {
                for (w = finish - 1; w >= 0; w--) {
                    if (adjacent(gra, v, w) >= 0 && visited[w] == 0) {
                        StackPush(view, w);
                    }
                }
            }
        }
        dropStack(view);
    }
    return latest_index;
}

/*
 * find earliest arrival time
 * return the len of pre array which helps to find pre stop from the last to start
 * refernece: weekly assignment5 dijkstra algorithm
 * update by Hongxiao Jin
 *
 * time complexity: O((mk+2)^2)
 */
int dijkstra_find_shortest_path(Graph g, Vertex source, int **pre) {
    int nV = numOfVertices(g);
    int dist[MAX_NODES]; // distance between two stops
    int pred[MAX_NODES][nV]; // previous matrix
    bool vSet[MAX_NODES];  // vSet[v] = true <=> v has not been processed
    int i, j;
    Vertex s, v;

    PQueueInit();
    // initialize pred
    for (i = 0; i < nV; i++) {
        for (j = 0; j < nV; j++) {
            pred[i][j] = -1;
        }
    }

    // initialize dist and vSet, enqueue stops
    for (s = 0; s < nV; s++) {
        joinPQueue(s);
        vSet[s] = true;
        if (adjacent(g, source, s) >= 0) {
            dist[s] = adjacent(g, source, s);
            pred[s][0] = source;
        } else {
            dist[s] = VERY_HIGH_VALUE;
        }
    }

    // set default value to source
    dist[source] = 0;
    pred[source][0] = source;
    vSet[source] = false;

    // create previous matrix
    int min;
    int num_pre_point = 1;
    s = leavePQueue(dist);
    while (!PQueueIsEmpty()) {
        for (v = 0; v < nV; v++) {
            min = dist[v];
            if (adjacent(g, s, v) >= 0 && vSet[v]) {
                if (adjacent(g, s, v) + dist[s] <= min) {
                    if (adjacent(g, s, v) + dist[s] == min && pred[v][0] != source) {
                        i = get_last_index(pred[v]);
                        pred[v][i] = s;
                        num_pre_point = i + 1;
                    } else {
                        pred[v][0] = s;
                    }
                    min = adjacent(g, s, v) + dist[s];
                    dist[v] = min;
                }
            }
        }
        s = leavePQueue(dist);
        vSet[s] = false;
    }

    // put the results into pre
    pred[source][0] = -1;
    for (i = 0; i < nV; i++) {
        for (j = 0; j < num_pre_point; j++) {
            pre[i][j] = pred[i][j];
        }
    }

    return num_pre_point;
}

int main() {
    int i, j, k; // counter used in circle
    int num_of_network_stop; // the total number of stops on the network
    int num_of_schedule; // the number of schedules
    int num_of_stop; // the total number of stops during one schedule

    // get stops on the network
    // e.g. stop[0] = Wynyard, stop[1] = QVB
    printf("Enter the total number of stops on the network: ");
    scanf("%d", &num_of_network_stop);
    char stop[num_of_network_stop][32]; // stop names
    char temp_name[32];// stop name
    for (i = 0; i < num_of_network_stop; i++) {
        scanf("%s", temp_name);
        strcpy(stop[i], temp_name);
        stop[i][strlen(temp_name)] = '\0';
    }

    printf("Enter the number of schedules: ");
    scanf("%d", &num_of_schedule);

    // get schedules information
    Schedule stop_information; // collect current stop information
    Schedule schedule_inform[num_of_schedule * num_of_network_stop]; // collect all the stop_information

    k = 0; // the first stop index
    for (i = 0; i < num_of_schedule; i++) {
        printf("Enter the number of stops: ");
        scanf("%d", &num_of_stop);
        for (j = 0; j < num_of_stop; j++) {
            initial_schedule(stop_information);
            scanf("%d", &stop_information.time);
            scanf("%s", temp_name);
            strcpy(stop_information.stop_name, temp_name);
            stop_information.stop_name[strlen(temp_name)] = '\0';
            stop_information.line = i;
            stop_information.stop_index = k;
            if (j == 0) {
                stop_information.check_first_or_last = 0;
            } else if (j == num_of_stop - 1) {
                stop_information.check_first_or_last = 1;
            } else {
                stop_information.check_first_or_last = 2;
            }
            schedule_inform[k] = stop_information;
            k++;
        }
    }
    int len_of_schedule = k;

    // make a graph of all stops with different time
    Graph network = newGraph(len_of_schedule + 2);
    Edge edge;

    // create the stop graph
    int duration; // the time between two stops

    // in the same line
    for (i = 0; i < len_of_schedule; i++) {
        if (i + 1 < len_of_schedule) {
            if (schedule_inform[i].line == schedule_inform[i + 1].line) {
                edge.v = schedule_inform[i].stop_index;
                edge.w = schedule_inform[i + 1].stop_index;
                duration = (schedule_inform[i + 1].time / 100 - schedule_inform[i].time / 100) * 60 +
                           (schedule_inform[i + 1].time % 100 - schedule_inform[i].time % 100);
                edge.weight = duration;
                insertEdge(network, edge);
            }
        }
    }
    // check change
    for (i = 0; i < len_of_schedule; i++) {
        for (j = 0; j < len_of_schedule; j++) {
            if (strcmp(schedule_inform[i].stop_name, schedule_inform[j].stop_name) == 0 &&
                schedule_inform[i].line != schedule_inform[j].line) {
                // where we can not change to an other line
                if ((schedule_inform[i].check_first_or_last == 0 && schedule_inform[j].check_first_or_last == 0) ||
                    (schedule_inform[i].check_first_or_last == 1 && schedule_inform[j].check_first_or_last == 1)) {
                    continue;
                } else {
                    if (schedule_inform[i].time >= schedule_inform[j].time) {
                        edge.v = schedule_inform[j].stop_index;
                        edge.w = schedule_inform[i].stop_index;
                        duration = (schedule_inform[i].time / 100 - schedule_inform[j].time / 100) * 60 +
                                   (schedule_inform[i].time % 100 - schedule_inform[j].time % 100);
                        edge.weight = duration;
                        insertEdge(network, edge);
                    }
                }
            }
        }
    }

    char start_name[32]; // the name of start stop
    char end_name[32]; // the name of destination
    int depart_time; // the start time
    printf("\nFrom: ");
    scanf("%s", temp_name);
    strcpy(start_name, temp_name);
    start_name[strlen(temp_name)] = '\0';
    while (strcmp(start_name, "done") != 0) {
        printf("To: ");
        scanf("%s", temp_name);
        strcpy(end_name, temp_name);
        end_name[strlen(temp_name)] = '\0';
        printf("Depart at: ");
        scanf("%d", &depart_time);
        int sources[num_of_schedule]; // the index of start stop
        int destinations[num_of_schedule]; // the index of destination

        int len_of_sou = get_sor_des_index(start_name, schedule_inform, len_of_schedule, sources, num_of_schedule,
                                           depart_time, 0);
        int len_of_des = get_sor_des_index(end_name, schedule_inform, len_of_schedule, destinations, num_of_schedule,
                                           depart_time, 1);

        int set_start = len_of_schedule; // set a common virtual start point
        int set_end = len_of_schedule + 1;// set a common virtual end point

        // connect real start stops to virtual start point
        for (i = 0; i < len_of_sou; i++) {
            edge.v = set_start;
            edge.w = sources[i];
            edge.weight = (schedule_inform[sources[i]].time / 100 - depart_time / 100) * 60 +
                          (schedule_inform[sources[i]].time % 100 - depart_time % 100);
            insertEdge(network, edge);
        }
        // connect real end stops to virtual end point
        for (i = 0; i < len_of_des; i++) {
            edge.v = destinations[i];
            edge.w = set_end;
            edge.weight = 0;
            insertEdge(network, edge);
        }

        // make a matrix to save all stops with their previous stop
        // by dijkstra_find_shortest_path to find earliest arrival stop
        int **pre = malloc(sizeof(int *) * numOfVertices(network));
        for (i = 0; i < numOfVertices(network); i++) {
            pre[i] = malloc(sizeof(int) * numOfVertices(network));
        }

        int len_of_pre = dijkstra_find_shortest_path(network, set_start, pre);

        // creat a new Graph, reverse network through pre
        Graph reverse_network = newGraph(numOfVertices(network));
        Edge new_edge;
        for (i = 0; i < numOfVertices(network); i++) {
            for (j = 0; j < len_of_pre; j++) {
                if (pre[i][j] != -1) {
                    new_edge.v = i;
                    new_edge.w = pre[i][j];
                    if (i == set_end) {
                        new_edge.weight = 0;
                    } else if (pre[i][j] == set_start) {
                        new_edge.weight = (schedule_inform[i].time / 100 - depart_time / 100) * 60 +
                                          (schedule_inform[i].time % 100 - depart_time % 100);
                    } else {
                        new_edge.weight = (schedule_inform[i].time / 100 - schedule_inform[pre[i][j]].time / 100) * 60 +
                                          (schedule_inform[i].time % 100 - schedule_inform[pre[i][j]].time % 100);
                    }
                    insertEdge(reverse_network, new_edge);
                }
            }
        }

        // make a new matrix to save partial stops with their previous stop
        // by dijkstra_find_shortest_path to find latest departure stop
        int **new_pre = malloc(sizeof(int *) * numOfVertices(reverse_network));
        for (i = 0; i < numOfVertices(network); i++) {
            new_pre[i] = malloc(sizeof(int) * numOfVertices(reverse_network));
        }
        dijkstra_find_shortest_path(reverse_network, set_end, new_pre);

        // dfs find the path
        int latest = DFS_find_latest_departure_time(reverse_network, sources, len_of_sou, set_end, schedule_inform);

        if (latest != -1 && new_pre[latest][0] != -1) {
            printf("\n");
            display(new_pre, latest, set_end, schedule_inform, len_of_schedule);
        } else {
            printf("\nNo connection found.\n");
        }

        for (i = 0; i < len_of_sou; i++) {
            edge.v = set_start;
            edge.w = sources[i];
            removeEdge(network, edge);
        }
        for (i = 0; i < len_of_des; i++) {
            edge.v = destinations[i];
            edge.w = set_end;
            removeEdge(network, edge);
        }

        printf("\nFrom: ");
        scanf("%s", temp_name);
        strcpy(start_name, temp_name);
        start_name[strlen(temp_name)] = '\0';
        for (i = 0; i < numOfVertices(network); i++) {
            free(pre[i]);
        }
        free(pre);

        for (i = 0; i < numOfVertices(reverse_network); i++) {
            free(new_pre[i]);
        }
        free(new_pre);

        freeGraph(reverse_network);
    }

    printf("Thank you for using goNSW.\n");
    freeGraph(network);
    return 0;
}