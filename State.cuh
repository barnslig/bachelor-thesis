/**
 * @file
 * @brief A state successor generator
 */
#ifndef __STATE_H_
#define __STATE_H_

#include <cstdint>

/**
 * Data structure that holds a model's state.
 * The waypoints model only needs an int32.
 */
struct State
{
  int32_t state;
};

#endif
