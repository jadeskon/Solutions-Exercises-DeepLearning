// mnist_and_filter_bank.cpp
//
// Small program that ...
//    ... reads in the MNIST image data set (digits 0,..,9) as 28x28 pixel images
//    ... visualizes some random examples MNIST digits of the training data set
//    ... generates a small filter bank
//    ... filters some random example images and visualizes the filter results
//
// ---
// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <conio.h>
#include <random>                 // for random numbers & 1D normal distribution
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI
#include <time.h>

#include "mnist_dataset_reader.h"


using namespace cv;
using namespace std;

int main()
{
   srand(time(NULL));

   // 1. read in MNIST data
   string path_to_extracted_mnist_files = "V:\\01_job\\00_vorlesungen_meine\\17_deep_learning_ws1617\\07_datasets\\01_mnist\\extracted";
   mnist_dataset_reader my_reader(path_to_extracted_mnist_files);


   /* 2. example code to retrieve and display a single MNIST image
   Mat* img = my_reader.get_mnist_image_as_cvmat( my_reader.get_train_images(), 1 );
   Mat img_resized;
   resize( *img, img_resized, cv::Size(0,0), 4,4);
   imshow("A sample digit", img_resized);
   */


   // 3. show me some example images of the training data from the MNIST data set
   Mat* samples_as_image = my_reader.get_board_of_sample_images(my_reader.get_train_images(), my_reader.get_train_labels(), 60000);
   imshow("Some sample MNIST training images", *samples_as_image);


   // 4. generate a filter bank
   vector<Mat*> filter_bank;
   Mat* kernel;

   float filter_values_kernel1[25] = { -1, -1, 0, 1, 1,
                                       -1, -1, 0, 1, 1,
                                       -1, -1, 0, 1, 1,
                                       -1, -1, 0, 1, 1,
                                       -1, -1, 0, 1, 1 };
   kernel = new Mat(5, 5, CV_32F, filter_values_kernel1);   
   filter_bank.push_back(kernel);

   float filter_values_kernel2[25] = { -1, -1, -1, -1, -1,
                                       -1, -1, -1, -1, -1,
                                        0,  0,  0,  0,  0,
                                        1,  1,  1,  1,  1,
                                        1,  1,  1,  1,  1 };
   kernel = new Mat(5, 5, CV_32F, filter_values_kernel2);
   filter_bank.push_back(kernel);

   float filter_values_kernel3[25] = { 1, 1, 0, -1, -1,
                                       1, 1, 0, -1, -1,
                                       1, 1, 0, -1, -1,
                                       1, 1, 0, -1, -1,
                                       1, 1, 0, -1, -1 };
   kernel = new Mat(5, 5, CV_32F, filter_values_kernel3);
   filter_bank.push_back(kernel);

   float a= 1.0f/ 9.0f;
   float b=-1.0f/16.0f;
   float filter_values_kernel4[25] = { b, b, b, b, b,
                                       b, a, a, a, b,
                                       b, a, a, a, b,
                                       b, a, a, a, b,
                                       b, b, b, b, b };
   kernel = new Mat(5, 5, CV_32F, filter_values_kernel4);
   filter_bank.push_back(kernel);



   // generate some random filters
   for (int further_filter = 1; further_filter <= 4; further_filter++)
   {
      kernel = new Mat(5, 5, CV_32F);
      for (int y=0; y<5; y++)
      {
         for (int x=0; x<5; x++)
         {
            float rnd_val = ((float)(rand() % 2001)/1000.0f) - 1.0f; // rnd number in [-1,1]
            kernel->at<float>(y,x) = rnd_val;
         }
      }      
      cout << *kernel;
      filter_bank.push_back(kernel);
   }

   // 5. filter 10 example MNIST images and visualize the filter results
   const int w = 800;
   const int h = 800;
   const int some_space = 10; // pixels
   const int N = 10; // number of example images to filter   

   Point anchor = Point(-1, -1);

   // A value to be added to each pixel during the convolution. By default it is 0
   double delta = 0;

   // depth of filter response matrix will be the same as that of source matrix 
   //int ddepth = -1;
   int ddepth = CV_32FC1;
   

   
   // we will filter N MNIST example images
   while (1)
   {
      Mat* big_image = new Mat(h, w, CV_8UC3);
      int x = 0;
      int y = 0;

      for (int img_nr = 0; img_nr < N; img_nr++)
      {
         // 5.1 "guess" a random image index
         int rnd_idx = rand() % 60000;

         // 5.2 get the corresponding example image as a cv::Mat
         Mat* example_img_gray = my_reader.get_mnist_image_as_cvmat(
            my_reader.get_train_images(), rnd_idx);
         resize(*example_img_gray, *example_img_gray, Size(56,56));

         // 5.3 draw it into the big image
         Mat example_image_3_channels;
         cvtColor(*example_img_gray, example_image_3_channels, CV_GRAY2RGB);
         Mat* roi = new Mat(*big_image, Rect(x, y, example_image_3_channels.cols, example_image_3_channels.rows));
         example_image_3_channels.copyTo(*roi);

         // 5.5 now filter the current example MNIST digit with all filters in the filter bank
         for (uint filter_nr = 0; filter_nr < filter_bank.size(); filter_nr++)
         {
            // compute next draw position (x,y)
            x += example_img_gray->cols + some_space;

            // get the filter
            Mat* next_filter_kernel = filter_bank.at(filter_nr);

            // The function filter2D uses the DFT - based algorithm in case of sufficiently
            // large kernels (11 x 11 or larger) and the direct algorithm
            // (that uses the engine retrieved by createLinearFilter()) for small kernels
            cv::Mat filter_result_gray;
            cv::filter2D(*example_img_gray, filter_result_gray,
                         ddepth, *next_filter_kernel,
                         anchor, delta, BORDER_DEFAULT);
            double minval,maxval;
            minMaxLoc(filter_result_gray, &minval, &maxval);
            printf("\nmin=%.1f max=%.1f", minval, maxval);

            // see http://stackoverflow.com/questions/12023958/what-does-cvnormalize-src-dst-0-255-norm-minmax-cv-8uc1 
            normalize(filter_result_gray, filter_result_gray, 0, 255, NORM_MINMAX, CV_8UC1);
            minMaxLoc(filter_result_gray, &minval, &maxval);
            printf(" --> min=%.1f max=%.1f", minval, maxval);
         
            // now convert the result to a 3 channel matrix
            Mat filter_result_gray_3C;
            cvtColor(filter_result_gray, filter_result_gray_3C, CV_GRAY2RGB);

            // copy the filter result to the big image as well
            Mat* roi = new Mat(*big_image, Rect(x, y, filter_result_gray_3C.cols, filter_result_gray_3C.rows));
            filter_result_gray_3C.copyTo(*roi);
         }

         // 5.5 go one line down
         x = 0;
         y += example_img_gray->rows + some_space;
      
      } // for (10 MNIST example images to filter)
      imshow("some MNIST digits and corresponding filter responses", *big_image);


      // 6. waitKey is important in order to make imshow() really display all images
      waitKey(0);

   }

   cout << "Good bye!\n";   

} // main