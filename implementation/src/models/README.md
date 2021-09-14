# Models

This folder contains multiple models that are used to test the Grapple model checker.

## Available Models

The following models are available:

| Filename | Description |
| -------- | ----------- |
| [WaypointsState.cuh](./WaypointsState.cuh) | The Swarm Verification Waypoints benchmark |
| [PhilosophersState.cuh](./PhilosophersState.cuh) | The Dining Philosophers Problem, implemented using `switch`-/`if`-conditions |
| [PhilosophersStateV2.cuh](./PhilosophersStateV2.cuh) | The Dining Philosophers Problem, implemented using the SIMT-specific ternary operator |
| [AndersonState.cuh](./AndersonState.cuh) | The Andersons queue lock mutual exclusion algorithm, implemented using the SIMT-specific ternary operator |
| [PetersonState.cuh](./PetersonState.cuh) | The Peterson mutual exclusion protocol, implemented using the  SIMT-specific ternary operator |

## Switching Models

To set the verified model, switch the type alias of `State` within [Grapple.cuh](../Grapple.cuh). Example:

```cpp
#include "models/WaypointsState.cuh"

// ...

using State = WaypointsState;
```

Do not forget to recompile!

## Implementing Models

Start your model by extending the [BaseState](./BaseState.cuh) class. Then, check one of the available models from above as an example on how to implement your own model!
