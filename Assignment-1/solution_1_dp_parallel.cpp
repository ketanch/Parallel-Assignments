#include <bits/stdc++.h>
#include <omp.h>
#include <sys/time.h>
#define ll long long

using namespace std;

ll nthreads;
ll V;
vector<vector<ll>> weight;
vector<vector<ll>> state;
vector<vector<ll>> path;

ll tspHelper(const vector<vector<ll>> &weights, vector<vector<ll>> &path, ll currPos, ll visited, vector<vector<ll>> &state);

ll tsp(const vector<vector<ll>> &weights, vector<vector<ll>> &path, ll currPos, ll visited, vector<vector<ll>> &state) {
    if (visited == ((1 << V) - 1)) {
        return weights[currPos][0];
    }

    if (state[currPos][visited] != LLONG_MAX) {
        return state[currPos][visited];
    }

#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (int i = 0; i < V; i++) {
        if (i == currPos || (visited & (1 << i))) {
            continue;
        }

        ll currWeight = weights[currPos][i] + tspHelper(weights, path, i, visited | (1 << i), state);

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

ll tspHelper(const vector<vector<ll>> &weights, vector<vector<ll>> &path, ll currPos, ll visited, vector<vector<ll>> &state) {
    if (visited == ((1 << V) - 1)) {
        return weights[currPos][0];
    }

    if (state[currPos][visited] != LLONG_MAX) {
        return state[currPos][visited];
    }

    for (int i = 0; i < V; i++) {
        if (i == currPos || (visited & (1 << i))) {
            continue;
        }

        ll currWeight = weights[currPos][i] + tspHelper(weights, path, i, visited | (1 << i), state);

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

int main(int argc, char *argv[]) {
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    if (argc < 2) {
        printf("Use ./%s <inp file> <out file>\n", __FILE__);
        exit(0);
    }
    freopen(argv[1], "r", stdin);
    freopen(argv[2], "w", stdout);
    nthreads = atoi(argv[3]);
    cout << nthreads << endl;

    cin >> V;
    weight.resize(V, vector<ll>(V, 0));
    state.resize(V, vector<ll>(((ll)1 << V), LLONG_MAX));
    path.resize(V, vector<ll>(((ll)1 << V), LLONG_MAX));
    vector<ll> min_weight_path(V);
    ll min_weight = LLONG_MAX;

    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            cin >> weight[i][j];
            weight[j][i] = weight[i][j];
        }
    }
    gettimeofday(&tv0, &tz0);
    min_weight = tsp(weight, path, 0, 1, state);
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
    cout << "Minimum Weight Path has weight: " << min_weight << endl;
    printf("time: %lf seconds\n", ((tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec)) / 1e6);

    return 0;
}