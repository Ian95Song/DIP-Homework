/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */

#include "StopWatch.h"

StopWatch::StopWatch() {
    reset();
}

void StopWatch::reset() {
    m_startTime = std::chrono::high_resolution_clock::now();
}

float StopWatch::getElapsedSeconds() {
    auto time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - m_startTime);
    return time.count();
}
