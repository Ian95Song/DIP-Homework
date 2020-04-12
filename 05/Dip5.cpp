//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description :
//============================================================================

#include "Dip5.h"


namespace dip5 {


/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // Hopefully already DONE, copy from last homework, just make sure you compute the kernel size from the given sigma (and not the other way around)
    cv::Mat G1D = cv::Mat::zeros(1,kSize,CV_32FC1);

    int i;
    float value;
    float sum =0.0;

    for (i=0; i<kSize; i++){
       value = exp(-(cv::pow(i-(kSize-1.0)/2.0,2.0)/cv::pow(sigma,2.0))/2.0);
       G1D.at<float>(0, i) = value;
       sum += value;
    }

    G1D = G1D/sum;

    return G1D;
//    return cv::Mat_<float>::zeros(1, kSize);
}

// Copy from last homework to use separableFilter
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{

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
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    // Hopefully already DONE, copy from last homework
    // But do mind that this one gets two different kernels for horizontal and vertical convolutions.

   int R = src.rows;//y
   int C = src.cols;//x

   cv::Mat tempog = cv::Mat(R,C,CV_32FC1);
   cv::Mat temp = cv::Mat(C,R,CV_32FC1);
   cv::Mat outog = cv::Mat(C,R,CV_32FC1);
   cv::Mat out = cv::Mat(R,C,CV_32FC1);
   float a;
   int i,j;

   tempog = spatialConvolution(src,kernelX);

   for (i= 0;i<R;i++){
       for (j=0;j<C;j++){
           a= tempog.at<float>(i,j);
           temp.at<float>(j,i)=a;
       }
   }

   outog = spatialConvolution(temp,kernelY);

   for (i= 0;i<C;i++){
       for (j=0;j<R;j++){
          out.at<float>(j,i) = outog.at<float>(i,j);
       }
   }

    return out;
}


/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // TO DO !!!

    cv::Mat G1D = createGaussianKernel1D(sigma);
    cv::Mat Dev1D = G1D.clone();


    int i,x;
    int c = (kSize-1)/2;

    for (i=0; i<kSize; i++){
       x = i-c;
       Dev1D.at<float>(0, i) = -x*G1D.at<float>(0,i)/cv::pow(sigma,2.0);
    }

    return Dev1D;

}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
    // TO DO !!!

    gradX.create(img.rows, img.cols);
    gradY.create(img.rows, img.cols);

    cv::Mat G1D = createGaussianKernel1D(sigmaGrad);
    cv::Mat Dev1D = createFstDevKernel1D(sigmaGrad);
    gradX = separableFilter(img,Dev1D,G1D);
    gradY = separableFilter(img,G1D,Dev1D);

}

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
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
    A00.create(gradX.rows, gradX.cols);
    A01.create(gradX.rows, gradX.cols);
    A11.create(gradX.rows, gradX.cols);

    cv::multiply(gradX,gradX,A00);
    cv::multiply(gradX,gradY,A01);
    cv::multiply(gradY,gradY,A11);

//    unsigned kSize = getOddKernelSizeForSigma(sigmaNeighborhood);

//    cv::GaussianBlur(A00,A00,cv::Size(kSize,kSize),sigmaNeighborhood,0,cv::BORDER_DEFAULT);
//    cv::GaussianBlur(A01,A01,cv::Size(kSize,kSize),sigmaNeighborhood,0,cv::BORDER_DEFAULT);
//    cv::GaussianBlur(A11,A11,cv::Size(kSize,kSize),sigmaNeighborhood,0,cv::BORDER_DEFAULT);


    cv::Mat G1D = createGaussianKernel1D(sigmaNeighborhood);

    A00 = separableFilter(A00,G1D,G1D);
    A01 = separableFilter(A01,G1D,G1D);
    A11 = separableFilter(A11,G1D,G1D);


//    cv::Mat Gauskernel = G1D * G1D.t();
//    cv::dft( A00, A00, CV_DXT_FORWARD, A00.rows);
//    cv::Mat kernels = cv::Mat::zeros( A00.rows, A00.cols, CV_32FC1);
//    int dx, dy; dx = dy = (kSize-1)/2.;
//    for(int i=0; i<kSize; i++)
//        for(int j=0; j<kSize; j++)
//            kernels.at<float>((i - dy + A00.rows) % A00.rows,(j - dx + A00.cols) % A00.cols) = Gauskernel.at<float>(i,j);
//	cv::dft( kernels, kernels, CV_DXT_FORWARD );
//	cv::mulSpectrums( A00, kernels, A00, 0 );
//	cv::dft( A00, A00, CV_DXT_INV_SCALE, A00.rows );


    // TO DO !!!
}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);

    cv::Mat tr, det,a,b;
    
	cv::add(A00,A11,tr); //tr
    cv::multiply(A00,A11,a);
    cv::multiply(A01,A01,b);
    det = a-b; //det

    cv::divide(det,tr,weight);

    isotropy = 4*det/(tr.mul(tr));



    // TO DO !!!
}


/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param minWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float minWeight, float minIsotropy)
{
    // TO DO !!!
    cv::Mat_<float> gradX, gradY,A00,A01,A11,weight,isotropy;
    calculateDirectionalGradients(img, sigmaGrad,gradX, gradY);
    calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);
    calculateFoerstnerWeightIsotropy(A00, A01, A11,weight, isotropy);

    float meanw = cv::mean(weight)[0];
    cv::threshold(weight,weight,minWeight*meanw,10000,cv::THRESH_TOZERO);
    cv::threshold(isotropy,isotropy,minIsotropy,1,cv::THRESH_TOZERO);


    int i,j;
    std::vector<cv::Vec2i> points;
    for (i= 0;i<weight.rows;i++){
       for (j=0;j<weight.cols;j++){
           if (weight.at<float>(i,j)* isotropy.at<float>(i,j)!=0 && isLocalMaximum(weight, j, i)==true){
              points.push_back(cv::Vec2i(j,i));
           }
       }
    }

    return points;
}



/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
