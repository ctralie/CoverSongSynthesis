__global__ void repeatedActivations(float* H, int K, int M, int r, float iterfac) {
    /*
    Avoid repeated activations with a maximum filter
    :param H: An KxM matrix whose repeated activations will be suppressed row-wise
    :param K, M: Dimensions
    :param r: Width of repeated activation filter
    :param iterfac: The shrinkage factor for non-maximum values in a neighborhood
    */
    extern __shared__ float x[];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    //TODO: FINISH THIS
}
