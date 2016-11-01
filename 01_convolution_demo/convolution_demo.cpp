// convolution_demo.cpp
//
// Small program that retrieves some image frames
// from your webcam or a specified video file
// and uses OpenCV's cv::filter2D() function to
// filter your images.
//
// Visualizing the filter results is a little bit tricky.
// Note that the filter response values can be negative as well!
//
// ---
// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org


#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <conio.h>
#define _USE_MATH_DEFINES
#include <math.h>                 // for M_PI
#include <time.h>


using namespace cv;
using namespace std;

// use web cam images or images from a video file?
const int USE_WEB_CAM=0;

// visualize negative filter values with blue and
// positive filter values with red?
const bool VISUALIZE_FILTER_RESULTS_WITH_COLORS = false;

// turn this on, to compute the absolute value of
// the filter response values and normalize all filter
// responses to the interval [-1,+1]
const bool DO_IT_RIGHT = false;




int main()
{
   //
   // 1. open webcam or video file as an image source
   //
   VideoCapture cap;
   if (USE_WEB_CAM)
   {
      // 1.1 use default webcam as image source
      cap = VideoCapture(0); // open the default camera
   }
   else
   {
      // 1.2 use a video file as image source
      // TODO! CHANGE THIS TO YOUR VIDEO FILE LOCATION! NOTE: \\ gives you one '\'
      string videoname =
       "V:\\01_job\\00_vorlesungen_meine\\17_deep_learning_ws1617\\14_exercises\\01_building_opencv_and_experimenting_with_convolutions\\test_video\\testpattern_with_clear_vertical_and_horizontal_edges.avi";
      cap = VideoCapture(videoname);
   }
   
   

   //
   // 2. could we really open that image source without problems?
   //
   if (!cap.isOpened())  // check if we succeeded
   {
      printf("\nCould not open video source!");
      _getch();
      return -1;
   }



   //
   // 3. Define a filter matrix (also called "filter kernel")
   //
   #define FilterType 1

   // 3x3 filter for detecting vertical edges
   #if FilterType == 1
      float filter_values[9] = { -1, 0, +1,
                                 -1, 0, +1,
                                 -1, 0, +1 };
   #endif

   // 3x3 filter for detecting horizontal edges
   #if FilterType == 2
      float filter_values[9] = { -1, -1, -1,
                                  0,  0,  0,
                                 +1, +1, +1 };
   #endif

   // 3x3 Center-Surround
   #if FilterType == 3
      float filter_values[9] = { -1, -1, -1,
                                 -1,  8, -1,
                                 -1, -1, -1 };
   #endif

   // 3x3 Glättungsfilter (Box-blur)
   #if FilterType == 4
      float filter_values[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                                 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                                 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
   #endif

   // NxN Glättungsfilter (Box-blur)
   #if FilterType == 5
      const int N = 10;
      float filter_values[N*N];
      for (int i = 0; i<N*N; i++)
         filter_values[i] = 1.0f / (float)(N*N);
   #else
      const int N = 3;
   #endif

   Mat kernel = Mat(N, N, CV_32F, filter_values);   


   // 4. filter the image frame by frame and
   //    visualize the filter result
   int frame_counter = 0;
   while (true)
   {
      // 4.1 try to get a new frame from camera or video file
      Mat frame;
      bool success = cap.read(frame); 

      // 4.2 did it work?
      if (!success)
         break; // no!
         
      // 4.3 convert input image to gray scale image
      Mat frame_gray;
      cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

      // 4.4 show camera input image as converted previously to gray-scale
      cv::imshow("camera frame", frame_gray);

      // 4.5 filter the image
      Mat result;

      // also see http://stackoverflow.com/questions/31604056/opencv-filter2d-is-there-any-use-of-kernels-not-anchored-at-center 
      //
      // the anchor point specifies how the filter matrix is layed over the pixel region to be filtered
      // actually only interesting for non-symmetric filter kernels, e.g. a 2x3 filter kernel
      //
      // the anchor point is actually a pixel offset used in the convolution, see 
      // http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#filter2d
      Point anchor = Point(-1, -1); 

      // A value to be added to each pixel during the convolution. By default it is 0
      double delta = 0;
      
      // depth of filter response matrix will be the same as that of source matrix 
      int ddepth = -1;
      if (VISUALIZE_FILTER_RESULTS_WITH_COLORS || DO_IT_RIGHT)
         ddepth = CV_32FC1;

      // The function filter2D uses the DFT - based algorithm in case of sufficiently
      // large kernels(11 x 11 or larger) and the direct algorithm
      // (that uses the engine retrieved by createLinearFilter()) for small kernels
      cv::filter2D(frame_gray, result,
                   ddepth, kernel,
                   anchor, delta, BORDER_DEFAULT);
      
      // 4.6 visualization of the filter result

      // activate this to do it right!
      
      if (DO_IT_RIGHT)
      {         
         double minval, maxval;
         cv::minMaxLoc(result, &minval, &maxval);
         printf("\nmin=%.1f max=%.1f", minval, maxval);
         result = abs(result)*(1.0/765.0); // adapt the normalization factor to the maximum response of your filter!
      }

      cv::imshow("filter result matrix", result);
      char filename[500];
      sprintf_s(filename, "V:\\05_tmp\\output\\%04d.png", frame_counter );
      imwrite( filename, result);
      sprintf_s(filename, "V:\\05_tmp\\input\\%04d.png", frame_counter);
      imwrite(filename, frame_gray);


      // 4.7 filter result visualization using colors      
      if (VISUALIZE_FILTER_RESULTS_WITH_COLORS)
      {
         // what is the minimum and maximum value?
         double minval, maxval;
         cv::minMaxLoc(result, &minval, &maxval);
         printf("\nmin=%.1f max=%.1f", minval, maxval);

         // normalize the values
         double normalization_factor = 1.0 / max(minval,maxval);
         result = result * normalization_factor;
         //cv::minMaxLoc(result, &minval, &maxval);
         //printf("\nmin=%.1f max=%.1f", minval, maxval);

         // create a 3 channel matrix (RGB) for visualizing
         // large filter values with red and small filter values
         // values with blue
         Mat visu = Mat(result.rows, result.cols, CV_8UC3);
         for (int y=0; y<result.rows; y++)
         {
            for (int x=0; x<result.cols; x++)
            {  
               float val = result.at<float>(y,x);
               int gray = (int)(255.0f*val);

               Vec3b col;
               if (val<0)
                  col = Vec3b(-gray,0,0);
               else
                  col = Vec3b(0,0,+gray);
            
               visu.at<Vec3b>(y, x) = col;
            }
         }

         // show result image
         cv::imshow("visu", visu);

      } // if (visualize by colors)


      // 4.8 ESC key pressed? --> exit!
      char c = waitKey(0);
      if (c=='q')
         break;

      frame_counter++;

   } // while
   

   // 5. release capture device, e.g., WebCam, such that other apps can use it
   cap.release();

} // main