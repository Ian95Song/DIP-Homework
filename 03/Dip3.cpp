//============================================================================
// Name    : Dip3.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip3.h"

#include <stdexcept>

namespace dip3 {

const char * const filterModeNames[NUM_FILTER_MODES] = {
    "FM_SPATIAL_CONVOLUTION",
    "FM_FREQUENCY_CONVOLUTION",
    "FM_SEPERABLE_FILTER",
    "FM_INTEGRAL_IMAGE",
};



/**
 * @brief Generates 1D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel1D(int kSize){

    // TO DO !!!
    cv::Mat G1D = cv::Mat::zeros(1,kSize,CV_32FC1);
    float sig = kSize / 5;

    int i;
    float value;
    float sum =0;

    for (i=0; i<kSize; i++){
       value = exp(-(cv::pow(i-(kSize-1)/2,2)/cv::pow(sig,2))/2);
       G1D.at<float>(0, i) = value;
       sum += value;
    }

    G1D = G1D/sum;


    return G1D;
}

/**
 * @brief Generates 2D gaussian filter kernel of given size
 * @param kSize Kernel size (used to calculate standard deviation)
 * @returns The generated filter kernel
 */
cv::Mat_<float> createGaussianKernel2D(int kSize){

    // TO DO !!!
    cv::Mat G2D = cv::Mat::zeros(kSize,kSize,CV_32FC1);
    float sig = kSize / 5;

    int i, j;
    float value;
    float sum =0.0;

    for (i=0; i<kSize; i++){
       for (j=0; j<kSize; j++){
          value = exp(-((cv::pow(i-(kSize-1)/2,2)/cv::pow(sig,2))+(cv::pow(j-(kSize-1)/2,2)/cv::pow(sig,2)))/2);
          G2D.at<float>(i, j) = value;
          sum += value;

       }
    }

    G2D = G2D/sum;


    return G2D;
}

/**
 * @brief Performes a circular shift in (dx,dy) direction
 * @param in Input matrix
 * @param dx Shift in x-direction
 * @param dy Shift in y-direction
 * @returns Circular shifted matrix
 */
cv::Mat_<float> circShift(const cv::Mat_<float>& in, int dx, int dy){

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
 * @brief Performes convolution by multiplication in frequency domain
 * @param in Input image
 * @param kernel Filter kernel
 * @returns Output image
 */
cv::Mat_<float> frequencyConvolution(const cv::Mat_<float>& in, const cv::Mat_<float>& kernel){

   // TO DO !!!

   cv::Mat kernelshift = cv::Mat::zeros(in.rows,in.cols,CV_32FC1);

   int i, j,dx,dy;

   for (i=0; i<kernel.rows; i++){
      for (j=0; j<kernel.cols; j++){
         kernelshift.at<float>(i,j)=kernel.at<float>(i,j);
      }
   }

   if (1==(1&kernel.cols)){
         dx = -(kernel.cols-1)/2;
         dy = -(kernel.rows-1)/2;
   }
   else {
         dx = -kernel.cols/2;
         dy = -kernel.rows/2;
   }

   kernelshift = circShift(kernelshift,dx,dy);
   cv::dft(kernelshift,kernelshift,0);

   cv::Mat out = in.clone();
   cv::Mat infre = in.clone();
   cv::dft(infre,infre,0);

   cv::mulSpectrums(infre,kernelshift,out,0);

   cv::dft(out,out,cv::DFT_INVERSE+cv::DFT_SCALE);

   return out;
}


/**
 * @brief  Performs UnSharp Masking to enhance fine image structures
 * @param in The input image
 * @param filterMode How convolution for smoothing operation is done
 * @param size Size of used smoothing kernel
 * @param thresh Minimal intensity difference to perform operation
 * @param scale Scaling of edge enhancement
 * @returns Enhanced image
 */
cv::Mat_<float> usm(const cv::Mat_<float>& in, FilterMode filterMode, int size, float thresh, float scale)
{
   // TO DO !!!

   // use smoothImage(...) for smoothing
   cv::Mat out = in.clone();
   cv::Mat i1 = in.clone();
   cv::Mat i2 = in.clone();
   cv::Mat i3 = in.clone();
   int i,j;

   i1 = smoothImage(in,size,filterMode);
   i2 = in-i1;
   for (i=0; i<i2.rows; i++){
      for (j=0; j<i2.cols; j++){
         if(abs(i2.at<float>(i,j))<thresh){
            i2.at<float>(i,j)=0;
         }
         else if(i2.at<float>(i,j)>=thresh){
            i2.at<float>(i,j)=i2.at<float>(i,j)-thresh;
         }
         else{
            i2.at<float>(i,j)=i2.at<float>(i,j)+thresh;
         }
      }
   }
   i3 = scale*i2;
   out = in+i3;

   return out;
}


/**
 * @brief Convolution in spatial domain
 * @param src Input image
 * @param kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{

   // Hopefully already DONE, copy from last homework

   int R1 = src.rows;//y
   int C1 = src.cols;//x
   int R2 = kernel.rows;//y
   int C2 = kernel.cols;//x
   int i, j, m, n;

   cv::Mat big = cv::Mat::zeros(R1+R2-1,C1+C2-1,CV_32FC1);

   for (i=0;i<R1;i++){
      for (j=0;j<C1;j++){
         float copi = 0.0;
         copi = src.at<float>(i,j);
         big.at<float>(i+R2/2-1/2,j+C2/2-1/2) = copi;
      }
   }

   cv::Mat out = src.clone();

   for (i= 0;i<R1;i++){
       for (j=0;j<C1;j++){
           float result = 0.0;
           for (m=0;m<R2;m++){
               for (n=0;n<C2;n++){
                   result += (big.at<float>(i+m,j+n) * kernel.at<float>(R2-m-1,C2-n-1));
               }
           }
           out.at<float>(i,j)=result;

       }
   }

   return out;

}


/**
 * @brief Convolution in spatial domain by seperable filters
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel){

   // TO DO !!!
   int R = src.rows;//y
   int C = src.cols;//x

   cv::Mat tempog = cv::Mat(R,C,CV_32FC1);
   cv::Mat temp = cv::Mat(C,R,CV_32FC1);
   cv::Mat outog = cv::Mat(C,R,CV_32FC1);
   cv::Mat out = cv::Mat(R,C,CV_32FC1);
   int i,j;
   float a;

   tempog = spatialConvolution(src,kernel);

   for (i= 0;i<R;i++){
       for (j=0;j<C;j++){
           a= tempog.at<float>(i,j);
           temp.at<float>(j,i)=a;
       }
   }

   outog = spatialConvolution(temp,kernel);

   for (i= 0;i<C;i++){
       for (j=0;j<R;j++){
          out.at<float>(j,i) = outog.at<float>(i,j);
       }
   }

   return out;
//   return src;
}


/**
 * @brief Convolution in spatial domain by integral images
 * @param src Input image
 * @param size Size of filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> satFilter(const cv::Mat_<float>& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

/**
 * @brief Performs a smoothing operation but allows the algorithm to be chosen
 * @param in Input image
 * @param size Size of filter kernel
 * @param type How is smoothing performed?
 * @returns Smoothed image
 */
cv::Mat_<float> smoothImage(const cv::Mat_<float>& in, int size, FilterMode filterMode)
{
    switch(filterMode) {
        case FM_SPATIAL_CONVOLUTION: return spatialConvolution(in, createGaussianKernel2D(size));	// 2D spatial convolution
        case FM_FREQUENCY_CONVOLUTION: return frequencyConvolution(in, createGaussianKernel2D(size));	// 2D convolution via multiplication in frequency domain
        case FM_SEPERABLE_FILTER: return separableFilter(in, createGaussianKernel1D(size));	// seperable filter
        case FM_INTEGRAL_IMAGE: return satFilter(in, size);		// integral image
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}



}

