#include <iostream>

#include "Queue.cuh"

__global__ void QueueKernel()
{
  // Current algorithm phase, either 0 or 1
  __shared__ int t;

  // TODO move queues into global memory so they can be 2x32x32x4
  __shared__ Queue q[2][10][10];

  // Wipe the queue memory on the first thread
  if (threadIdx.x == 0)
  {
    t = 0;
    memset(&q, 0, sizeof(q));
  }

  // Sync all threads after init
  __syncthreads();

  printf("From within the kernel!!!\n");
  printf("successor of state 10: %i\n", State{10}.successor_generation(3, 0).state);

  Queue *myq = &q[t][threadIdx.x][threadIdx.x];

  myq->push(State{10});
  myq->push(State{11});
  myq->push(State{12});
  myq->push(State{13});
  myq->push(State{14});

  printf("%i\n", myq->empty());

  printf("%i\n", myq->pop()->state);
  printf("%i\n", myq->pop()->state);
  printf("%i\n", myq->pop()->state);
  printf("%i\n", myq->pop()->state);

  printf("%i\n", myq->empty());

  printf("End kernel\n");
}

int main()
{
  cudaError_t err = cudaSuccess;

  QueueKernel<<<1, 3>>>();
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr
        << "Failed to launch kernel! Error code: "
        << cudaGetErrorString(err)
        << "\n";
    return EXIT_FAILURE;
  }

  err = cudaDeviceReset();
  if (err != cudaSuccess)
  {
    std::cerr
        << "Failed to deinitialize the device! Error code: "
        << cudaGetErrorString(err)
        << "\n";
    return EXIT_FAILURE;
  }

  return 0;
}