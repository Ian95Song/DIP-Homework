/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#ifndef NETWORKARCHITECTURES_H
#define NETWORKARCHITECTURES_H

#include "Network.h"
#include <random>

namespace dip6 {

Network buildSmallNetwork(std::mt19937 &rng);

Network buildBigNetwork(std::mt19937 &rng);
    
}
    
#endif // NETWORKARCHITECTURES_H
