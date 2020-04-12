//============================================================================
// Name        : Dip4.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip4.h"

namespace dip4 {

using namespace std::complex_literals;

/*

===== std::complex cheat sheet =====

Initialization:

std::complex<float> a(1.0f, 2.0f);
std::complex<float> a = 1.0f + 2.0if;

Common Operations:

std::complex<float> a, b, c;

a = b + c;
a = b - c;
a = b * c;
a = b / c;

std::sin, std::cos, std::tan, std::sqrt, std::pow, std::exp, .... all work as expected

Access & Specific Operations:

std::complex<float> a = ...;

float real = a.real();
float imag = a.imag();
float phase = std::arg(a);
float magnitude = std::abs(a);
float squared_magnitude = std::norm(a);

std::complex<float> complex_conjugate_a = std::conj(a);

*/



/**
 * @brief Computes the complex valued forward DFT of a real valued input
 * @param input real valued input
 * @return Complex valued output, each pixel storing real and imaginary parts
 */
cv::Mat_<std::complex<float>> DFTReal2Complex(const cv::Mat_<float>& input)
{
    // TO DO !!!
    cv::Mat out = cv::Mat_<std::complex<float>>(input.rows, input.cols);
    cv::dft(input,out,cv::DFT_COMPLEX_OUTPUT);
    return out;
//    return cv::Mat_<std::complex<float>>(input.rows, input.cols);
}


/**
 * @brief Computes the real valued inverse DFT of a complex valued input
 * @param input Complex valued input, each pixel storing real and imaginary parts
 * @return Real valued output
 */
cv::Mat_<float> IDFTComplex2Real(const cv::Mat_<std::complex<float>>& input)
{
    // TO DO !!!
    cv::Mat out = cv::Mat_<float>(input.rows, input.cols);
    cv::dft(input,out,cv::DFT_SCALE+cv::DFT_REAL_OUTPUT+cv::DFT_INVERSE);
    return out;
//    return cv::Mat_<float>(input.rows, input.cols);
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @return Circular shifted matrix
*/
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy)
{
    // TO DO !!!
   cv::Mat out = in.clone();
   int R = in.rows; //y
   int C = in.cols; //x
   int i,j,outx,outy;
   float result;


   for (i=0; i<R; i++){

      if ((i+dy)<0){
         outy = i+dy+R;
      }
      else if ((i+dy)>(R-1)){
         outy = i+dy-R;
      }
      else {
         outy = i+dy;
      }

      for (j=0; j<C; j++){
         if ((j+dx)<0){
            outx = j+dx+C;
         }
         else if ((j+dx)>(C-1)){
            outx = j+dx-C;
         }
         else {
            outx = j+dx;
         }
         result = in.at<float>(i, j);
         out.at<float>(outy,outx) = result;
      }
   }

   return out;
}


/**
 * @brief Computes the thresholded inverse filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return The inverse filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeInverseFilter(const cv::Mat_<std::complex<float>>& input, const float eps)
{
    // TO DO !!!
//    std::vector<cv::Mat> compinput;
//    split(input,compinput);
//
//    cv::Mat abso = input.clone();
//    cv::Mat out = input.clone();
//
//    int i,j;
//    float maxi =0.0;
//    float x;
//    for (i=0; i<input.rows; i++){
//       for (j=0; j<input.cols; j++){
//          x = std::sqrt(std::pow(compinput[0].at<float>(i,j),2)+std::pow(compinput[1].at<float>(i,j),2));
//          abso.at<float>(i,j) = x;
//          if (x>maxi){
//             maxi = x;
//          }
//       }
//    }
//
//    float thr = maxi*eps;
//
//    for (i=0; i<input.rows; i++){
//       for (j=0; j<input.cols; j++){
//          if (abso.at<float>(i,j)>=thr){
//             out(i,j) = (1.0f+0.0if) / (abso.at<float>(i,j));
//          }
//          else {
//             out(i,j) = (1.0f+0.0if)  / thr ;
//          }
//       }
//    }



    int i, j;
    float maxi = 0.0;
//    cv::Mat out = cv::Mat_<std::complex<float>>(input.rows, input.cols);
    cv::Mat_<std::complex<float>> out (input.rows, input.cols,CV_32FC1);
//    cv::Mat out = input.clone();

    for (i=0; i<input.rows; i++){
       for (j=0; j<input.cols; j++){
          if (std::abs(input.at<float>(i,j))>maxi){
             maxi = std::abs(input.at<float>(i,j));
          }
       }
    }

    float thr = eps*maxi;

    for (i=0; i<input.rows; i++){
       for (j=0; j<input.cols; j++){
          if (std::abs(input(i,j))>=thr){
             out.at<std::complex<float>>(i,j) = (1.0f+0.0if) / (input.at<std::complex<float>>(i,j));
          }
          else {
             out.at<std::complex<float>>(i,j) = 1 / thr ;
          }
       }
    }

    return out;
}


/**
 * @brief Applies a filter (in frequency domain)
 * @param input Image in frequency domain (complex valued)
 * @param filter Filter in frequency domain (complex valued), same size as input
 * @return The filtered image, complex valued, in frequency domain
 */
cv::Mat_<std::complex<float>> applyFilter(const cv::Mat_<std::complex<float>>& input, const cv::Mat_<std::complex<float>>& filter)
{
    // TO DO !!!
    cv::Mat out = input.clone();
//    cv::mulSpectrums(input,filter,out,cv::DFT_COMPLEX_OUTPUT,false);
    cv::mulSpectrums(input,filter,out,0);
    return out;
}


/**
 * @brief Function applies the inverse filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param eps Factor to compute the threshold (relative to the max amplitude)
 * @return Restorated output image
 */
cv::Mat_<float> inverseFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, const float eps)
{
    // TO DO !!!
//    cv::Mat dftfilter = filter.clone();
//    cv::Mat invfilter = filter.clone();
//    dftfilter = DFTReal2Complex(filter);
////    dft(filter,dftfilter);
//    invfilter = computeInverseFilter(dftfilter,eps);
//    cv::Mat kernelshift = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
//    cv::Mat out = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
//    cv::Mat out2 = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
//    cv::Mat out3 = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
////    out = DFTReal2Complex(degraded);
//    cv::dft(degraded,out);
//
//
//    int i, j,dx,dy;
//    for (i=0; i<filter.rows; i++){
//       for (j=0; j<filter.cols; j++){
//          kernelshift.at<float>(i,j)=invfilter.at<float>(i,j);
//       }
//     }
//
//    if (1==(1&filter.cols)){
//         dx = -(filter.cols-1)/2;
//         dy = -(filter.rows-1)/2;
//    }
//    else {
//         dx = -filter.cols/2;
//         dy = -filter.rows/2;
//    }
//
//    kernelshift = circShift(kernelshift,dx,dy);
//    out = applyFilter(out,kernelshift);
//    cv::dft(out,out,cv::DFT_INVERSE+cv::DFT_SCALE);
////    out3 = IDFTComplex2Real(out2);

    cv::Mat dftfilter = filter.clone();
    cv::Mat invfilter = filter.clone();

    cv::Mat kernelshift = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
    cv::Mat out = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
//    cv::Mat out2 = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
//    cv::Mat out3 = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
    out = DFTReal2Complex(degraded);
//    cv::dft(degraded,out);


    int i, j,dx,dy;
    for (i=0; i<filter.rows; i++){
       for (j=0; j<filter.cols; j++){
          kernelshift.at<float>(i,j)=filter.at<float>(i,j);
       }
     }

    if (1==(1&filter.cols)){
         dx = -(filter.cols-1)/2;
         dy = -(filter.rows-1)/2;
    }
    else {
         dx = -filter.cols/2;
         dy = -filter.rows/2;
    }

    kernelshift = circShift(kernelshift,dx,dy);
    dftfilter = DFTReal2Complex(kernelshift);
//    dft(filter,dftfilter);
    invfilter = computeInverseFilter(dftfilter,eps);


    out = applyFilter(out,invfilter);
//    cv::dft(out,out,cv::DFT_INVERSE+cv::DFT_SCALE);
    out = IDFTComplex2Real(out);

    return out;
}


/**
 * @brief Computes the Wiener filter
 * @param input Blur filter in frequency domain (complex valued)
 * @param snr Signal to noise ratio
 * @return The wiener filter in frequency domain (complex valued)
 */
cv::Mat_<std::complex<float>> computeWienerFilter(const cv::Mat_<std::complex<float>>& input, const float snr)
{
    // TO DO !!!
    cv::Mat_<std::complex<float>> out (input.rows, input.cols,CV_32FC1);
    int i,j;

    for (i=0; i<input.rows; i++){
       for (j=0; j<input.cols; j++){
          std::complex<float> a = input.at<std::complex<float>>(i,j);
          float real = a.real();
          float imag = a.imag();
          std::complex<float> conjpk (real, -imag);
          std::complex<float> deno = std::pow(std::abs(a),2) + (1/(snr*snr));
          std::complex<float> result = conjpk/deno;
          out.at<std::complex<float>>(i,j) = result;
       }
    }




//    std::vector<cv::Mat> compinput;
//    split(input,compinput);
//
//    cv::Mat_<std::complex<float>> conjpk (input.rows, input.cols,CV_32FC1);
//    cv::Mat_<std::complex<float>> squpk (input.rows, input.cols,CV_32FC1);
//    cv::Mat_<std::complex<float>> out (input.rows, input.cols,CV_32FC1);
//
//    int i,j;
//    std::complex<float> x,y,result;
//
//    for (i=0; i<input.rows; i++){
//       for (j=0; j<input.cols; j++){
//          y = compinput[0].at<std::complex<float>>(i,j)-compinput[1].at<std::complex<float>>(i,j);
//          conjpk.at<std::complex<float>>(i,j) = y;
//          x = std::pow(compinput[0].at<std::complex<float>>(i,j),2)+std::pow(compinput[1].at<std::complex<float>>(i,j),2);
//          squpk.at<std::complex<float>>(i,j) = x;
//       }
//    }
//
//    std::complex<float> z = 1/std::pow(snr,2);
//    for (i=0; i<input.rows; i++){
//       for (j=0; j<input.cols; j++){
//          result = conjpk.at<std::complex<float>>(i,j)/(squpk.at<std::complex<float>>(i,j)+z);
//          out.at<std::complex<float>>(i,j) = result ;
//       }
//    }
    return out;
}

/**
 * @brief Function applies the wiener filter to restorate a degraded image
 * @param degraded Degraded input image
 * @param filter Filter which caused degradation
 * @param snr Signal to noise ratio of the input image
 * @return Restorated output image
 */
cv::Mat_<float> wienerFilter(const cv::Mat_<float>& degraded, const cv::Mat_<float>& filter, float snr)
{
    // TO DO !!!
    cv::Mat dftfilter = filter.clone();
    cv::Mat invfilter = filter.clone();

    cv::Mat kernelshift = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
    cv::Mat out = cv::Mat::zeros(degraded.rows,degraded.cols,CV_32FC1);
    out = DFTReal2Complex(degraded);


    int i, j,dx,dy;
    for (i=0; i<filter.rows; i++){
       for (j=0; j<filter.cols; j++){
          kernelshift.at<float>(i,j)=filter.at<float>(i,j);
       }
     }

    if (1==(1&filter.cols)){
         dx = -(filter.cols-1)/2;
         dy = -(filter.rows-1)/2;
    }
    else {
         dx = -filter.cols/2;
         dy = -filter.rows/2;
    }

    kernelshift = circShift(kernelshift,dx,dy);
    dftfilter = DFTReal2Complex(kernelshift);
    invfilter = computeWienerFilter(dftfilter,snr);


    out = applyFilter(out,invfilter);
    out = IDFTComplex2Real(out);

    return out;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * function degrades the given image with gaussian blur and additive gaussian noise
 * @param img Input image
 * @param degradedImg Degraded output image
 * @param filterDev Standard deviation of kernel for gaussian blur
 * @param snr Signal to noise ratio for additive gaussian noise
 * @return The used gaussian kernel
 */
cv::Mat_<float> degradeImage(const cv::Mat_<float>& img, cv::Mat_<float>& degradedImg, float filterDev, float snr)
{

    int kSize = round(filterDev*3)*2 - 1;

    cv::Mat gaussKernel = cv::getGaussianKernel(kSize, filterDev, CV_32FC1);
    gaussKernel = gaussKernel * gaussKernel.t();

    cv::Mat imgs = img.clone();
    cv::dft( imgs, imgs, CV_DXT_FORWARD, img.rows);
    cv::Mat kernels = cv::Mat::zeros( img.rows, img.cols, CV_32FC1);
    int dx, dy; dx = dy = (kSize-1)/2.;
    for(int i=0; i<kSize; i++)
        for(int j=0; j<kSize; j++)
            kernels.at<float>((i - dy + img.rows) % img.rows,(j - dx + img.cols) % img.cols) = gaussKernel.at<float>(i,j);
	cv::dft( kernels, kernels, CV_DXT_FORWARD );
	cv::mulSpectrums( imgs, kernels, imgs, 0 );
	cv::dft( imgs, degradedImg, CV_DXT_INV_SCALE, img.rows );

    cv::Mat mean, stddev;
    cv::meanStdDev(img, mean, stddev);

    cv::Mat noise = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0, stddev.at<double>(0)/snr);
    degradedImg = degradedImg + noise;
    cv::threshold(degradedImg, degradedImg, 255, 255, CV_THRESH_TRUNC);
    cv::threshold(degradedImg, degradedImg, 0, 0, CV_THRESH_TOZERO);

    return gaussKernel;
}


}
