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