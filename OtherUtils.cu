__global__ void ZerosToOnes(float* V, int M, int N, float eps) {
    /*
    Turn zeros into ones (for use when something goes in the denomenator)
    :param V: An MxN input matrix
    :param M, N: Dimensions
    :param eps: The value below which to consider things zero
    */
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int idx;
    if (i < M && j < N) {
        idx = i*N + j;
        if (V[idx] < eps) {
            V[idx] = 1;
        }
    }
}

__global__ void TileWDenom(float* WDenomIn, float* WDenomOut, int T, int M, int K) {
    /*
    
    (Since broadcasting doesn't work as advertised in skcuda, I had to revert to this)
    :param WDenomIn: A T x 1 x K array
    :param WDenomOut: A T x M x K array, where each [:, i, :] holds WDenomIn
    :param T, M, K: Dimensions
    */
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int MK = M*K;
    float x;
    int i;
    if (t < T && k < K) {
        x = WDenomIn[t*K + k];
        for (i = 0; i < M; i++) {
            WDenomOut[MK*t + K*i + k] = x;
        }
    }
}


__global__ void TileHDenom(float* HDenomIn, float* HDenomOut, int F, int K, int N) {
    /*
    
    (Since broadcasting doesn't work as advertised in skcuda, I had to revert to this)
    :param HDenomIn: A F x K x 1 array
    :param HDenomOut: A F x K x N array, where each [:, :, j] holds HDenomIn
    :param F, K, N: Dimensions
    */
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int KN = K*N;
    float x;
    int j;
    if (f < F && k < K) {
        x = HDenomIn[f*K + k];
        for (j = 0; j < N; j++) {
            HDenomOut[KN*f + N*k + j] = x;
        }
    }
}


//TODO: Bitonic sort doesn't work yet, and it's slow in global memory
__global__ void bitonicSortNonneg(float* X, float* XPow2, int M, int N, int NPow2) {
    /*
    Do a bitonic sort so that every row of a matrix X is in sorted
    order
    :param X: Pointer to matrix
    :param XPow2: Pointer to matrix holding power of 2 zeropadded X
    :param M: Size of each column of X
    :param N: Size of each row of X
    :param NPow2: Size of each row of X rounded up to nearest power of 2
    */
    extern __shared__ float x[];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int j1, j2;
    float x1, x2;
    float min, max;
    int size = 2;
    int stride;
    int diffPow2 = (NPow2 - N);

    if (i >= M || j >= NPow2) {
        return;
    }

    //Step 2: Perform bitonic sort
    while (size < NPow2 << 1) {
        stride = size >> 1;
        while (stride > 0) {
            j1 = stride*2*(j/stride) + j%stride;
            j2 = j1 + stride;
            x1 = XPow2[i*NPow2 + j1];
            x2 = XPow2[i*NPow2 + j2];
            if (x1 < x2) {
                min = x1;
                max = x2;
            }
            else {
                min = x2;
                max = x1;
            }
            if (j/(size/2)%2 > 0) {
                XPow2[i*NPow2 + j1] = min;
                XPow2[i*NPow2 + j2] = max;
            }
            else {
                XPow2[i*NPow2 + j1] = max;
                XPow2[i*NPow2 + j2] = min;
            }
            stride = stride >> 1;
            __syncthreads();
        }
        size = size << 1;
    }

    //Step 3: Copy Result Back, noting that the first (NPow2-N)
    //values in each row are dummy values
    j = j*2;
    if (j >= diffPow2) {
        X[i*N + j - diffPow2] = XPow2[i*N + j];
    }
    j++;
    if (j >= diffPow2) {
        X[i*N + j - diffPow2] = XPow2[i*N + j];
    }
}