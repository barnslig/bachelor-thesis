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

  Queue *myq = &q[t][threadIdx.x][threadIdx.x];

  queue_push(myq, State{10});
  queue_push(myq, State{11});
  queue_push(myq, State{12});
  queue_push(myq, State{13});
  queue_push(myq, State{14});

  printf("%i\n", queue_empty(myq));

  printf("%i\n", queue_pop(myq)->state);
  printf("%i\n", queue_pop(myq)->state);
  printf("%i\n", queue_pop(myq)->state);
  printf("%i\n", queue_pop(myq)->state);

  printf("%i\n", queue_empty(myq));

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
