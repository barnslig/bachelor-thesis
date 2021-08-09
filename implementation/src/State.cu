#include "State.cuh"

__host__ __device__ void State::successor_generation(State *successor, unsigned int process, unsigned int ndc)
{
  successor->state = state | 1 << ((4 * process) + ndc);
}

__host__ __device__ bool State::violates()
{
  /* How to generate 100 random 32 bit integers using numpy:
   *
   *   import numpy as np
   *   rng = np.random.default_rng()
   *   rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, 100, endpoint=True)
   *
   * Here is a little trick to save us from branching. Most probably, it could
   * be improved using bitwise operators or something :)
   */
  return 0 != ((state == -796893842) +
               (state == 779999969) +
               (state == -1168537298) +
               (state == 1200431791) +
               (state == 343263341) +
               (state == -484020923) +
               (state == 152254944) +
               (state == 1580803326) +
               (state == 742905064) +
               (state == -1078233077) +
               (state == -535171611) +
               (state == 3801864) +
               (state == -591946906) +
               (state == -1735204896) +
               (state == -2019895492) +
               (state == -1892053421) +
               (state == 1619348239) +
               (state == -1582071078) +
               (state == 975455264) +
               (state == -1009488133) +
               (state == -1986586248) +
               (state == 1298121607) +
               (state == 591829652) +
               (state == 174446516) +
               (state == 1146941867) +
               (state == -973302145) +
               (state == 1886697991) +
               (state == -242879520) +
               (state == 844493943) +
               (state == -44194297) +
               (state == -2052363332) +
               (state == 2073759503) +
               (state == 1427832783) +
               (state == 1723743020) +
               (state == -1153542822) +
               (state == -311282643) +
               (state == 1951362811) +
               (state == 2021087655) +
               (state == 1852558397) +
               (state == 142999217) +
               (state == 1551312664) +
               (state == 1688262063) +
               (state == -953275708) +
               (state == 1267931653) +
               (state == -424531623) +
               (state == -360759171) +
               (state == -1906136628) +
               (state == -787884137) +
               (state == -369036714) +
               (state == -547100928) +
               (state == 2062351048) +
               (state == -241464430) +
               (state == -330827675) +
               (state == -1338579048) +
               (state == 1000668082) +
               (state == -1024509637) +
               (state == -1157809220) +
               (state == -1223390847) +
               (state == -1226148451) +
               (state == -1525868926) +
               (state == -428231122) +
               (state == 1686987859) +
               (state == 1165889762) +
               (state == -486440368) +
               (state == 107898966) +
               (state == -975841995) +
               (state == -2120763899) +
               (state == -298195237) +
               (state == -480588986) +
               (state == -1895479827) +
               (state == -1480726272) +
               (state == -426296827) +
               (state == 1207040394) +
               (state == 1794600435) +
               (state == 1621471265) +
               (state == -153254338) +
               (state == 1752632236) +
               (state == -1309418877) +
               (state == 558531649) +
               (state == -456823131) +
               (state == 1704887816) +
               (state == 990204918) +
               (state == 1243623110) +
               (state == 1383812845) +
               (state == 1926782188) +
               (state == 2124838199) +
               (state == -721123533) +
               (state == -22990483) +
               (state == 1076265871) +
               (state == 177235009) +
               (state == -1657803211) +
               (state == -206509875) +
               (state == -330928781) +
               (state == -946222203) +
               (state == -305828449) +
               (state == 1262699236) +
               (state == -963772524) +
               (state == -265091042) +
               (state == 1933187888) +
               (state == 1725002572));
}
