//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip2.h"

namespace dip2 {


/**
 * @brief Convolution in spatial domain.
 * @details Performs spatial convolution of image and filter kernel.
 * @params src Input image
 * @params kernel Filter kernel
 * @returns Convolution result
 */
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
   // TO DO !!

   int R1 = src.rows;
   int C1 = src.cols;
   int R2 = kernel.rows;
   int C2 = kernel.cols;
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
 * @brief Moving average filter (aka box filter)
 * @note: you might want to use Dip2::spatialConvolution(...) within this function
 * @param src Input image
 * @param kSize Window size used by local average
 * @returns Filtered image
 */
cv::Mat_<float> averageFilter(const cv::Mat_<float>& src, int kSize)
{
   // TO DO !!

   cv::Mat kernal = cv::Mat(kSize,kSize,CV_32FC1,1./float(kSize*kSize));
   cv::Mat out = spatialConvolution(src,kernal);
   return out;
}

/**
 * @brief Median filter
 * @param src Input image
 * @param kSize Window size used by median operation
 * @returns Filtered image
 */
cv::Mat_<float> medianFilter(const cv::Mat_<float>& src, int kSize)
{
//    TO DO !!
//   cv::Mat out = src.clone();
   cv::Mat out = cv::Mat::ones(src.size(),CV_32FC1)*120.;
//   kSize = 3;

   int R1 = src.rows;
   int C1 = src.cols;
   int i, j, m, n;

   for (i= (kSize-1)/2;i<R1-(kSize-1)/2;i++){
       for (j=(kSize-1)/2;j<C1-(kSize-1)/2; j++){
           float result = 0.0;
           std::vector<float> vec;
           for (m=0;m<kSize;m++){
               for (n=0;n<kSize;n++){
                   vec.push_back(src.at<float>(i-(kSize-1)/2+m,j-(kSize-1)/2+n));
               }
           }
           std::sort(vec.begin(),vec.end());
           out.at<float>(i,j)= vec[(kSize*kSize-1)/2];
       }
   }


   return out;
}

/**
 * @brief Bilateral filer
 * @param src Input image
 * @param kSize Size of the kernel
 * @param sigma_spatial Standard-deviation of the spatial kernel
 * @param sigma_radiometric Standard-deviation of the radiometric kernel
 * @returns Filtered image
 */
cv::Mat_<float> bilateralFilter(const cv::Mat_<float>& src, int kSize, float sigma_spatial, float sigma_radiometric)
{
    // TO DO !!

   int R1 = src.rows;
   int C1 = src.cols;
   int i, j, m, n;
   float spatdis,raddis;

   cv::Mat big = cv::Mat::zeros(R1+kSize-1,C1+kSize-1,CV_32FC1);

   for (i=0;i<R1;i++){
      for (j=0;j<C1;j++){
         float copi = 0.0;
         copi = src.at<float>(i,j);
         big.at<float>(i+kSize/2-1/2,j+kSize/2-1/2) = copi;
      }
   }

   cv::Mat out = src.clone();

   cv::Mat spatkernel = cv::Mat::zeros(kSize,kSize,CV_32FC1);
   cv::Mat radkernel = cv::Mat::zeros(kSize,kSize,CV_32FC1);
   cv::Mat kernel = cv::Mat::zeros(kSize,kSize,CV_32FC1);




   for (i= 0;i<R1;i++){
       for (j=0;j<C1; j++){

           for (m=0;m<kSize;m++){
               for (n=0;n<kSize;n++){
                   spatdis = cv::pow(m-(kSize-1)/2,2)+cv::pow(n-(kSize-1)/2,2);
                   spatkernel.at<float>(m,n)=exp(-(spatdis/(2*cv::pow(sigma_spatial,2))));
                   raddis=cv::pow((big.at<float>(i+m,j+n)-src.at<float>(i,j)),2);
                   radkernel.at<float>(m,n)=exp(-(raddis/(2*cv::pow(sigma_radiometric,2))));
               }
           }

           cv::multiply(spatkernel,radkernel,kernel);
           kernel = kernel/cv::sum(kernel)[0];

           float result = 0.0;
           for (m=0;m<kSize;m++){
               for (n=0;n<kSize;n++){
                   result += (big.at<float>(i+m,j+n) * kernel.at<float>(m,n));
                }
           }

           out.at<float>(i,j)=result;
       }
   }


    return out;
}

/**
 * @brief Non-local means filter
 * @note: This one is optional!
 * @param src Input image
 * @param searchSize Size of search region
 * @param sigma Optional parameter for weighting function
 * @returns Filtered image
 */
cv::Mat_<float> nlmFilter(const cv::Mat_<float>& src, int searchSize, double sigma)
{
    return src.clone();
}



/**
 * @brief Chooses the right algorithm for the given noise type
 * @note: Figure out what kind of noise NOISE_TYPE_1 and NOISE_TYPE_2 are and select the respective "right" algorithms.
 */
NoiseReductionAlgorithm chooseBestAlgorithm(NoiseType noiseType)
{
    // TO DO !!
    switch (noiseType) {
        case dip2::NOISE_TYPE_1: {
            return dip2::NR_MEDIAN_FILTER;
        } break;
        case dip2::NOISE_TYPE_2: {
            return dip2::NR_BILATERAL_FILTER;
        } break;
    }
}



cv::Mat_<float> denoiseImage(const cv::Mat_<float> &src, NoiseType noiseType, dip2::NoiseReductionAlgorithm noiseReductionAlgorithm)
{
    // TO DO !!

    // for each combination find reasonable filter parameters

    switch (noiseReductionAlgorithm) {
        case dip2::NR_MOVING_AVERAGE_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::averageFilter(src, 5);
                case NOISE_TYPE_2:
                    return dip2::averageFilter(src, 5);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_MEDIAN_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::medianFilter(src, 5);
                case NOISE_TYPE_2:
                    return dip2::medianFilter(src, 3);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        case dip2::NR_BILATERAL_FILTER:
            switch (noiseType) {
                case NOISE_TYPE_1:
                    return dip2::bilateralFilter(src, 5, 20.0f, 250.0f);
                case NOISE_TYPE_2:
                    return dip2::bilateralFilter(src, 5, 1.0f, 300.0f);
                default:
                    throw std::runtime_error("Unhandled noise type!");
            }
        default:
            throw std::runtime_error("Unhandled filter type!");
    }
}





// Helpers, don't mind these

const char *noiseTypeNames[NUM_NOISE_TYPES] = {
    "NOISE_TYPE_1",
    "NOISE_TYPE_2",
};

const char *noiseReductionAlgorithmNames[NUM_FILTERS] = {
    "NR_MOVING_AVERAGE_FILTER",
    "NR_MEDIAN_FILTER",
    "NR_BILATERAL_FILTER",
};


}
