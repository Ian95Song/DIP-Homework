cmake_minimum_required(VERSION 3.1)

project( dip5 LANGUAGES CXX )

find_package( OpenCV REQUIRED )


add_library(code 
    Dip5.cpp
    Dip5.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG")
endif()

target_link_libraries(code 
    PUBLIC
        ${OpenCV_LIBS}
)




add_executable(main 
    main.cpp 
)

set_target_properties(main PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(main 
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



