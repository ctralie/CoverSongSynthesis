/*Programmer: Chris Tralie
Purpose: To create fast 2D convolutional matrix multiplication code
as parallel CUDA kernels
*/

__global__ void MatMulNaive(float* A, float* B, float* C, int M, int K, int N) {
    /*
    A: MxK matrix
    B: KxN matrix
    C: MxN matrix
    */
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = 0;
    float res = 0.0;
    for (k = 0; k < K; k++) {
        res += A[i*K + k]*B[k*N+j];
    }
    C[i*N+j] = res;
}


__global__ void MatMulConv2D(float* W, float* H, float* Lam, int M, int N, int K, 
        int T, int F, int TBlocks, int FBlocks) {
    /*
    Perform 2D convolutional matrix multiplication
    :param W: An MxKxT input matrix
    :param H: A KxNxF input matrix
    :param Lam: A MxN output matrix
    :param M, N, K, T, F: Dimensions
    :param TBlocks: Number of blocks of T padding to load in per grid block
    :param FBlocks: Number of blocks of F padding to load in per grid block
    */

    /*Shared Memory Layout in x, which holds chunks of W and H that are 
        shared with overlapping convolutions.  For a block size of B:
            1) W goes from i-F:i+B-1, k, 0:T-1.
            2) H goes from k, j-T:j+B-1, 0:F at an offset of (F+B)*T
    */
    extern __shared__ float x[]; 
    int hoff = (F+blockDim.x)*T; //Offset of H chunk in shared memory
    //TODO: Think about row major coalescing with order of access
    int iblock = blockIdx.x*blockDim.x;
    int jblock = blockIdx.y*blockDim.y;
    int i = iblock + threadIdx.x;
    int j = jblock + threadIdx.y;
    int KT = K*T;
    int NF = N*F;
    int k, t, f;
    int thist, thisf;
    int thisi, thisj;
    float res = 0.0;
    //Loop over all K separately
    for (k = 0; k < K; k++) {
        //Step 1: Load chunks of W into shared memory
        //W goes from iblock-F+1:iblock+B-1, k, 0:T-1
        for (f = 0; f < FBlocks+1; f++) {
            if (f == FBlocks) {
                //On the last one, copy over interval from [iblock, iblock+B-1]
                thisi = i; 
                thisf = F+threadIdx.x;
            }
            else{ 
                //For the other chunks, copy over interval from [iblock-F, iblock-1]
                thisi = i-F+f*blockDim.x;
                if (thisi >= iblock) {
                    continue; //Past F boundary for block at iblock-1
                }
                thisf = f*blockDim.x+threadIdx.x;
            }
            for (t = 0; t < TBlocks; t++) {
                thist = t*blockDim.y + threadIdx.y;
                if (thist >= T) {
                    continue;
                }
                //Pull out W[thisi, k, thist]
                if (thisi < 0 || thisi >= M) {
                    x[T*thisf+thist] = 0;
                }
                else {
                    x[T*thisf+thist] = W[thisi*KT+k*T+thist];
                }
            }
        }
        __syncthreads();
        //Step 2: Load chunks of H into shared memory
        //H goes from k, jblock-T+1:jblock+B-1, 0:F at an offset of (F+B)*T
        for (t = 0; t < TBlocks+1; t++) {
            if (t == TBlocks) {
                //On the last one, copy over interval from [jblock:jblock+B-1]
                thisj = j; 
                thist = T+threadIdx.y;
            }
            else {
                //For the other chunks, copy over interval from [jblock-T:jblock-1]
                thisj = j-T+t*blockDim.y;
                if (thisj >= jblock) {
                    continue; //Past T boundary for block at jblock-1
                }
                thist = t*blockDim.y+threadIdx.y;
            }
            for (f = 0; f < FBlocks; f++) {
                thisf = f*blockDim.x + threadIdx.x;
                if (thisf >= F) {
                    continue;
                }
                //Pull out H[k, thisj, f] and put in at an offset
                if (thisj < 0 || thisj >= N) {
                    x[hoff + F*thist+thisf] = 0;
                }
                else{
                    x[hoff + F*thist+thisf] = H[k*NF + thisj*F + thisf];
                }
            }
        }
        __syncthreads();

        //Step 3: Do matrix multiplication
        for (f = 0; f < F; f++) {
            for (t = 0; t < T; t++) {
                //W[i-f, k, t]*H[k, j-t, f]
                res += x[(F+threadIdx.x-f)*T + t]*x[hoff+(T+threadIdx.y-t)*F+f];
            }
        }
        __syncthreads();//The lack of this sync at the end of each k
        // was causing a major bug!!!
    }
    if (i < M && j < N) {
        Lam[i*N+j] = res;
    }
}


__global__ void MatMulConv2DWGrad(float* W, float* H, float* V, float* VLam, 
        int M, int N, int K, int T, int F, int FBlocks) {
    /*
    Perform 2D convolutional matrix multiplicative update of W
    :param W: An MxKxT input matrix
    :param H: A KxNxF input matrix
    :param V: A MxN target matrix
    :param VLam: An MxN matrix which is W*H 2DConv
    :param VLam: The current MxN approximation of V
    :param M, N, K, T, F: Dimensions
    :param FBlocks: Number of blocks of F padding to load in per grid block
    */

    /*Shared Memory Layout in x, which holds chunks of V/VLam and H that are 
        shared with overlapping convolutions.  For a block size of B:
            1) V goes from iblock:iblock+B+F-1, j:j+B-1
            2) VLam goes from iblock:iblock+B+F-1, j:j+B-1 at an offset of
                (B+F)*B
            3) H goes from k, j-T:j+B-1, 0:F at an offset of 2*(B+F)*B

            NOTE: Assuming 1 block of t values
    */
    extern __shared__ float x[]; 
    int vlamoff = (blockDim.x+F)*blockDim.x; //Offset of VLam chunk in shared memory
    int hoff = 2*vlamoff; //Offset of H chunk in shared memory
    int iblock = blockIdx.x*blockDim.x;
    int k = blockIdx.y;
    int tblock = blockIdx.z*blockDim.z;
    int jblock;
    int i = iblock + threadIdx.x;
    int t = tblock + threadIdx.y;
    int NF = N*F;
    int j, f;
    int thist, thisf;
    int thisi, thisj;
    float num = 0.0;
    float denom = 0.0;

    //Loop over chunks of the j dimension
    for (jblock = 0; jblock < N; jblock += blockDim.x) {
        j = jblock+threadIdx.y;
        //Step 1: Copy out sections of V and VLam
        for (f = 0; f < FBlocks+1; f++) {
            if (f == FBlocks) {
                //On the last one, copy over interval from [iblock, iblock+B-1]
                thisi = i; 
                thisf = threadIdx.x;
            }
            else{ 
                //For the other chunks, copy over interval from [iblock+B, iblock+B+F-1]
                thisf = blockDim.x+f*blockDim.x+threadIdx.x;
                if (thisf >= blockDim.x+F) {
                    continue; //Past F boundary
                }
                thisi = i+(f+1)*blockDim.x;
            }
            //Pull out V[thisi, j] and VLam[thisi, j]
            if (thisi < 0 || thisi >= M || j >= N) {
                x[blockDim.x*thisf+threadIdx.y] = 0;
                x[vlamoff+blockDim.x*thisf+threadIdx.y] = 0;
            }
            else {
                x[blockDim.x*thisf+threadIdx.y] = V[thisi*N+j];
                x[vlamoff+blockDim.x*thisf+threadIdx.y] = VLam[thisi*N+j];
            }

        }
        __syncthreads();
        //Step 2: Copy out sections of H
        //H goes from k, j-T:j+B-1, 0:F
        /*if (threadIdx.y <= T) {
            //Copy out [jblock-T, jblock] section
            thisj = jblock - threadIdx.y;
            thist = T - threadIdx.y;
            for (f = 0; f < F; f++) {
                if (thisj < 0 || thisj >= M) {
                    x[hoff+thist*F+f] = 0;
                }
                else {
                    x[hoff+thist*F+f] = H[k*NF + thisj*F + f];
                }
            }
        }
        //Copy out [1, blockdim-1] part
        if (threadIdx.y > 0) {
            thisj = j;
            thist = T + threadIdx.y;
            for (f = 0; f < F; f++) {
                if (thisj >= M) {
                    x[hoff+thist*F+f] = 0;
                }
                else{
                    x[hoff+thist*F+f] = H[k*NF + thisj*F + f];
                }
            }
        }
        __syncthreads();
        */


        //Step 3: Do multiplication
        for (f = 0; f < F; f++) {
            for (thisj = 0; thisj < blockDim.y; thisj++) {
                //V[i+f, j] * H[k, j-t, f]
                //num += x[(threadIdx.x+f)*T+thisj]*x[hoff+(blockDim.y-threadIdx.y+thisj)*F+f];
                num += x[(threadIdx.x+f)*blockDim.x+thisj];
            }
        }
        __syncthreads();
    }
    if (i < M && t < T) {
        //W[i, k, t]
        W[i*K*T + k*T + t] = num;
    }
}