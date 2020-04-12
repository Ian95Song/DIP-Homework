/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>


class StopWatch {
    public:
        StopWatch();
        void reset();
        float getElapsedSeconds();
    protected:
        std::chrono::high_resolution_clock::time_point m_startTime;
};


#endif // STOPWATCH_H
