//============================================================================
// Name        : Dip5.h
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : header file for fifth DIP assignment
//============================================================================

#include <opencv2/opencv.hpp>

#include <iostream>

namespace dip5 {

/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma);   

/**
* @brief Convolution in spatial domain by seperable filters
* @param src Input image
* @param size Size of filter kernel
* @returns Convolution result
*/
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY);

    
/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma);


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY);

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11);
/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy);

/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param minWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float minWeight = 1.5f, float minIsotropy = 0.8f);


/* *****************************
  GIVEN FUNCTIONS
***************************** */

// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma);

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y);

};
