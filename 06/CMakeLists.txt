cmake_minimum_required(VERSION 3.1)

project( dip6 LANGUAGES CXX )

find_package( OpenCV REQUIRED )
find_package( Threads REQUIRED )


add_library(code 
    Dip6.cpp
    Dip6.h
    
    Network.cpp
    Network.h
    Tensor.cpp
    Tensor.h

    NetworkArchitectures.cpp
    NetworkArchitectures.h
    DataProvider_DIV2K.cpp
    DataProvider_DIV2K.h

    SIMD.cpp
    SIMD.h
    TaskScheduler.cpp
    TaskScheduler.h
    StopWatch.cpp
    StopWatch.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math ")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG ")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")# -fopt-info-vec-all")
endif()

# TODO: Test this somehow in VS
#if(MSVC)
#    add_definition(/arch:AVX)
#endif()

target_link_libraries(code 
    PUBLIC
        ${OpenCV_LIBS}
        Threads::Threads
)




add_executable(trainSmall 
    trainSmall.cpp 
)

set_target_properties(trainSmall PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(trainSmall 
    PRIVATE
        code
)


add_executable(trainBig 
    trainBig.cpp 
)

set_target_properties(trainBig PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(trainBig 
    PRIVATE
        code
)



add_executable(application 
    application.cpp 
)

set_target_properties(application PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(application 
    PRIVATE
        code
)


add_executable(createDB 
    createDB.cpp 
)

set_target_properties(createDB PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(createDB 
    PRIVATE
        code
)



add_executable(unit_test 
    unit_test.cpp 
)

set_target_properties(unit_test PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(unit_test 
    PRIVATE
        code
)



