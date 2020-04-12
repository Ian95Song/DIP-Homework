/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "NetworkArchitectures.h"


#include "Dip6.h"

namespace dip6 {


Network buildSmallNetwork(std::mt19937 &rng)
{
    Network network;
    
    float decay = 1e-7f;
    network.appendLayer<dip6::layers::ConvReference>(3, 3, 3, 8)  // 3x3 -  3 -> 8 channels 
                .setOptimizer<dip6::optimizer::MomentumSGD>(decay, 0.0f)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();
    network.appendLayer<dip6::layers::ConvReference>(1, 1, 8, 3)  // 1x1 -  8 -> 3 channels 
                .setOptimizer<dip6::optimizer::MomentumSGD>(decay, 0.0f)
                .initialize(rng);
                
    return network;
}

Network buildBigNetwork(std::mt19937 &rng)
{
    Network network;
    
    float decay = 1e-6f;

    network.appendLayer<dip6::layers::ConvOptimized>(5, 5, 3, 16)  // 5x5 -  3 -> 16 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();

    network.appendLayer<dip6::layers::ConvOptimized>(3, 3, 16, 32)  // 3x3 -  16 -> 32 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();
    
    network.appendLayer<dip6::layers::ConvOptimized>(3, 3, 32, 64)  // 3x3 -  32 -> 64 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();

    network.appendLayer<dip6::layers::ConvOptimized>(3, 3, 64, 64)  // 3x3 -  64 -> 64 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();

    network.appendLayer<dip6::layers::Upsample>(3, 3); // Upsample width and height 3x

    network.appendLayer<dip6::layers::ConvOptimized>(3, 3, 64, 16)  // 3x3 -  64 -> 16 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);
    network.appendLayer<dip6::layers::ReLU>();
    network.appendLayer<dip6::layers::ConvOptimized>(5, 5, 16, 3)  // 5x5 -  16 -> 3 channels 
                .setOptimizer<dip6::optimizer::Adam>(decay)
                .initialize(rng);

     
    return network;
}

    
}
    
