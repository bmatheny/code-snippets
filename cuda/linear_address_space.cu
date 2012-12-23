/**
 * Given some kernel with positional arguments (n -> columns, m -> rows), below
 * is how to calculate the row/col being worked on and the index into a
 * linearized matrix. See also http://en.wikipedia.org/wiki/Row-major_order
 */
__global__ void SomeKernel(float * d_in, float * d_out, int cols, int rows) {
  int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
  int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;

  // ensure we are in a valid row/col, needed because we have generated more
  // threads than needed.
  if ((rowIdx < rows) && (colIdx < cols)) {
    // linearize the index to access d_in/d_out. This is needed because an NxM
    // (colsXrows) matrix is linearized into a continguous address space. This
    // simple calculation gives you the index into a linearized 2 dimensional
    // array. Row*Width+Col is linear index.
    int idx = rowIdx*cols + colIdx;
  }
}
