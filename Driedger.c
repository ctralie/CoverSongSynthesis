/*Programmer: Chris Tralie
Purpose: C Implementations of Driedger's NMF updates for the
H matrix.  Includes 3 steps
1) Avoiding repeated activations
2) Restricting number of simultaneous activiations
3) Supporting time-continuous activations
*/
#include <stdio.h>
#include <stdlib.h>
#include "Driedger.h"

int compare(const void* a, const void* b) {
    //Sort in descending order
    double res = (*(double*)b - *(double*)a);
    if (res < 0)
        return -1;
    if (res > 0)
        return 1;
    return 0;
}

void DriedgerUpdates(double* H, int M, int N, int r, int p, int c, float iterfac) {
    int i, j, k;
    int di, dj, dc, dk;
    int rw = (r-1)/2;
    double max;
    double* coltemp = (double*)malloc(M*sizeof(double));
    double* rowtemp = (double*)malloc(N*sizeof(double));
    double* diag = (double*)malloc((N+M+2*c)*sizeof(double));
    for (k = 0; k < N+M+2*c; k++) {
        diag[k] = 0.0;
    }
    //Step 1: Avoid repeated activations
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            max = 0.0;
            for (k = -rw; k <= rw; k++) {
                if (j+k >= 0 && j+k < N) {
                    if (H[i*N + j+k] > max) {
                        max = H[i*N + j+k];
                    }
                }
            }
            if (H[i*N+j] < max) {
                rowtemp[j] = H[i*N+j]*iterfac;
            }
            else {
                rowtemp[j] = H[i*N+j];
            }
        }
        for (j = 0; j < N; j++) {
            H[i*N + j] = rowtemp[j];
        }
    }
    //Step 2: Restrict number of simultaneous activations
    for (j = 0; j < N; j++) {
        //Copy over column
        for (k = 0 ; k < M; k++) {
            coltemp[k] = H[k*N + j];
        }
        //Sort column
        qsort(coltemp, M, sizeof(double), compare);
        for (k = 0; k < M; k++) {
            if (H[k*N + j] < coltemp[p]) {
                H[k*N + j] *= iterfac;
            }
        }
    }
    //Step 3: Support time-continuous activations
    di = M-1;
    dj = 0;
    //Go across all diagonals
    while (dj < N) {
        k = 0;
        //Do cumsum
        while (di+k < M && dj+k < N) {
            diag[c+k] = H[(di+k)*N + dj+k] + diag[c+k-1];
            k++;
        }
        for (dc = 0; dc < c; dc++) {
            diag[c+k+dc] = diag[c+k-1];
        }
        //Update diagonal
        for (dk = 0; dk < k; dk++) {
            //x2 = z[2*c::] - z[0:-2*c]
            H[(di+dk)*N + dj+dk] = diag[2*c+dk] - diag[dk];
        }
        //Go to next diagonal
        if (di == 0) {
            dj++;
        }
        else {
            di--;
        }
    }
    free(rowtemp);
    free(coltemp);
    free(diag);
}


void DiagUpdates(double* H, int M, int N, int c, float iterfac) {
    int k;
    int di, dj, dc, dk;
    double max;
    double* diag = (double*)malloc((N+M+2*c)*sizeof(double));
    for (k = 0; k < N+M+2*c; k++) {
        diag[k] = 0.0;
    }
    //Support time-continuous activations
    di = M-1;
    dj = 0;
    //Go across all diagonals
    while (dj < N) {
        k = 0;
        //Do cumsum
        while (di+k < M && dj+k < N) {
            diag[c+k] = H[(di+k)*N + dj+k] + diag[c+k-1];
            k++;
        }
        for (dc = 0; dc < c; dc++) {
            diag[c+k+dc] = diag[c+k-1];
        }
        //Update diagonal
        for (dk = 0; dk < k; dk++) {
            //x2 = z[2*c::] - z[0:-2*c]
            H[(di+dk)*N + dj+dk] = diag[2*c+dk] - diag[dk];
        }
        //Go to next diagonal
        if (di == 0) {
            dj++;
        }
        else {
            di--;
        }
    }
    free(diag);
}