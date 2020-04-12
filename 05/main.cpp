//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================


#include "Dip5.h"

#include <iostream>


using namespace std;
using namespace cv;



#define TEST_FOR_DIP6

#ifdef TEST_FOR_DIP6
#include <immintrin.h>
void testForDip6() {
    
    std::cout << "Testing SSE" << std::endl;
    {
        volatile float a_ = 1.0f;
        volatile float b_ = 2.0f;
        volatile __m128 a = _mm_set1_ps(a_);
        volatile __m128 b = _mm_set1_ps(b_);
        volatile __m128 c = _mm_add_ps(a, b);
        float c_[4];
        _mm_storeu_ps(c_, c);
        a_ = c_[0];
    }
    
    std::cout << "Testing SSE3" << std::endl;
    {
        volatile float a_ = 1.0f;
        volatile float b_ = 2.0f;
        volatile __m128 a = _mm_set1_ps(a_);
        volatile __m128 b = _mm_set1_ps(b_);
        volatile __m128 c = _mm_hadd_ps(a, b);
        float c_[4];
        _mm_storeu_ps(c_, c);
        a_ = c_[0];
    }
    std::cout << "Testing SSE4.1" << std::endl;
    {
        volatile float a_ = 1.0f;
        volatile float b_ = 2.0f;
        volatile __m128 a = _mm_set1_ps(a_);
        volatile __m128 b = _mm_set1_ps(b_);
        volatile __m128 c = _mm_dp_ps(a, b, 0xFF);
        float c_[4];
        _mm_storeu_ps(c_, c);
        a_ = c_[0];
    }
#ifdef __AVX__
    std::cout << "Testing AVX" << std::endl;
    {
        volatile float a_ = 1.0f;
        volatile float b_ = 2.0f;
        volatile __m256 a = _mm256_set1_ps(a_);
        volatile __m256 b = _mm256_set1_ps(b_);
        volatile __m256 c = _mm256_add_ps(a, b);
        float c_[8];
        _mm256_storeu_ps(c_, c);
        a_ = c_[0];
    }
#endif

}
#else
void testForDip6() {
}
#endif


// usage: path to image in argv[1], sigma in argv[2]
// main function. loads image, calls processing routines, shows keypoints
int main(int argc, char** argv) {

    testForDip6();


    // check if enough arguments are defined
    if (argc < 2){
        cout << "Usage:\n\tdip5 path_to_original [sigmaGrad [sigmaNeighborhood]]"  << endl;
        cout << "Press enter to exit"  << endl;
        cin.get();
        return -1;
    }

    // load image, path in argv[1]
    cout << "load image" << endl;
    Mat img = imread(argv[1]);
    if (!img.data){
        cout << "ERROR: original image not specified"  << endl;
        cout << "Press enter to exit"  << endl;
        cin.get();
        return -1;
    }

    Mat imgF;
    // convert U8 to 32F
    cv::cvtColor(img, imgF, cv::COLOR_BGR2GRAY);
    imgF.convertTo(imgF, CV_32FC1);
    cout << " > done" << endl;

    // define standard deviation of directional gradient
    float sigmaBlur;
    if (argc < 3)
        sigmaBlur = 1.5f;
    else
        sigmaBlur = atof(argv[2]);
    // define standard deviation of directional gradient
    float sigmaNeighborhood;
    if (argc < 4)
        sigmaNeighborhood = 2.0f;
    else
        sigmaNeighborhood = atof(argv[3]);


    imshow("originalGrey", img);
    imwrite("originalGrey.png", imgF );


    // calculate interest points
    std::vector<cv::Vec2i> points = dip5::getFoerstnerInterestPoints(imgF, sigmaBlur, sigmaNeighborhood);

    cout << "Number of detected interest points:\t" << points.size() << endl;
    Mat imgRGB;
    // convert to color
    cv::cvtColor(imgF, imgRGB, cv::COLOR_GRAY2BGR);
    imgRGB.convertTo(imgRGB, CV_8UC3);
    
    // plot result
    for (auto p : points) 
        cv::circle(imgRGB, p, 10, Scalar(0,0,255));
    
    imshow("keypoints", imgRGB);
    imwrite("keypoints.png", imgRGB);
    // wait
    waitKey(0);

    
    return 0;
} 
