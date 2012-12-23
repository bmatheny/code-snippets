/**
 * CUDA organizes execution into grids. Each device contains grids. Each grid
 * contains blocks. Each block contains threads.
 * Device[id]->Grid[id]->Block[id]->Thread[id].
 */
__global__ void OrgKernel(void * in, void * out, int size) {
  // block and grid dimensions describe how large the execution grid/block is
  // thread/block indexes specify the index into the block

  // grid dimensions have an x and y coordinate
  // after CUDA 4 can have z coordinate
  int gridDimX = gridDim.x;
  // block dimensions have an x, y and z coordinate
  int blockDimX = blockDim.x;
  // block indexes tell you where in the block you are executing and have an x
  // and y coordinate. Ranges from 0 to gridDim.x - 1
  int blockIdX = blockIdx.x;
  // thread indexes mirror block dimensions and have an x, y and z coordinate
  // x*y*z <= totalNumberOfThreadsAvailable
  int threadIdX = threadIdx.x;

  int xCoord = blockIdX*blockDimX + threadIdX;
  int yCoord = blockIdx.y*blockDim.y + threadIdx.y;
}

int main(void) {
  // grids can't be specified by users, but thread/block sizes can
  // taken from http://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s

  int imageWidth = 512; // we have a 512x512 image
  int imageHeight = 512;

  int desiredThreadsPerBlock = 64;
  int neededBlocks = (imageWidth*imageHeight)/desiredThreadsPerBlock; // 4096 blocks needed

  // 8x8 is == desiredThreadsPerBlock
  dim3 threadsPerBlock(8, 8); // 64 threads per block
  // 64*64 == neededBlocks
  dim3 numBlocks(imageWidth/threadsPerBlock.x, // 512/8 = 64
                 imageHeight/threadsPerBlock.y); // also 64, 64*64 is 4096 total blocks

  // launch kernel with specified blocks, etc
  // first param numBlocks dictates the size of the grid, 64x64 blocks
  // second param threadsPerBlock dictates the size of each block, 8x8 or 64 threads per block
  OrgKernel <<<numBlocks,threadsPerBlock>>>((void*)NULL, (void*)NULL, imageWidth*imageHeight);
}
