#include <bits/stdc++.h>
#include <omp.h>
#include <sys/time.h>
#define ll long long

using namespace std;

int nthreads;
int V;
vector<vector<int>> weights;
vector<vector<int>> state;
vector<vector<int>> path;

int tspHelper(const vector<vector<int>> &weights, vector<vector<int>> &path, int currPos, int visited, vector<vector<int>> &state);

int tsp(const vector<vector<int>> &weights, vector<vector<int>> &path, int currPos, int visited, vector<vector<int>> &state) {
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int i = 0; i < V; i++) {
        if (i == currPos || (visited & (1 << i))) {
            continue;
        }

        int currWeight = weights[currPos][i] + tspHelper(weights, path, i, visited | (1 << i), state);

        if (currWeight < state[currPos][visited]) {
#pragma omp critical
            {
                if (currWeight < state[currPos][visited]) {
                    state[currPos][visited] = currWeight;
                    path[currPos][visited] = i;
                }
            }
        }
    }
    return state[currPos][visited];
}

int tspHelper(const vector<vector<int>> &weights, vector<vector<int>> &path, int currPos, int visited, vector<vector<int>> &state) {
    if (visited == ((1 << V) - 1)) {
        return weights[currPos][0];
    }

    if (state[currPos][visited] != INT_MAX) {
        return state[currPos][visited];
    }

    for (int i = 0; i < V; i++) {
        if (i == currPos || (visited & (1 << i))) {
            continue;
        }

        int currWeight = weights[currPos][i] + tspHelper(weights, path, i, visited | (1 << i), state);

        if (currWeight < state[currPos][visited]) {
            state[currPos][visited] = currWeight;
            path[currPos][visited] = i;
        }
    }
    return state[currPos][visited];
}

int main(int argc, char *argv[]) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    if (argc < 4) {
        printf("Use ./%s <inp file> <out file> <num threads>\n", __FILE__);
        exit(0);
    }
    FILE *input = freopen(argv[1], "r", stdin);
    FILE *output = freopen(argv[2], "w", stdout);
    nthreads = atoi(argv[3]);

    cin >> V;
    weights.resize(V, vector<int>(V, 0));
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            cin >> weights[i][j];
            weights[j][i] = weights[i][j];
        }
    }

    gettimeofday(&tv0, &tz0);

    state.resize(V, vector<int>(((int)1 << V), INT_MAX));
    path.resize(V, vector<int>(((int)1 << V), INT_MAX));
    int min_weight = tsp(weights, path, 0, 1, state);

    gettimeofday(&tv1, &tz1);

    int currPos = 0;
    int currVisited = 1;
    cout << currPos << " ";
    for (int i = 0; i < V - 1; i++) {
        currPos = path[currPos][currVisited];
        currVisited = currVisited | (1 << currPos);
        cout << currPos << " ";
    }
    cout << endl;
    cout << min_weight << endl;
    // printf("time: %ld microseconds\n", (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec));

    return 0;
}