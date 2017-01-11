// hebbian_learning_test.cpp
//
// Test of unsupervised learning of features with the Hebbian learning rule.
//
// 1.
// Download a video, e.g. https://www.youtube.com/watch?v=ccdzdqAjpGs
//
// 2.
// Then use the video or a part of the video as input for the learner.
// For the specified video use e.g. the first 1300 frames.
//
// NOTE: For OpenCV to be able to read the video make sure,
//       opencv_ffmpeg310.dll can be found
//       by adding the path to your PATH environment variable,
//       or just copy the file into the project path
//
// 3.
// Visualize the feature during learning.
// Is it meaningful? What's the problem?
//
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
#include <fstream>

using namespace cv;
using namespace std;

string VIDEOFILENAME = "V:\\01_job\\12_datasets\\plane_spotting\\1_lowres.mp4";
int FS = 100; // filter size
float LEARN_RATE = 1e-6f;


int main()
{
   printf("Hebbian learning test.\n");

   VideoCapture cap( VIDEOFILENAME );
   if (!cap.isOpened())
   {
      printf("Sorry! Could not open the file %s\n", VIDEOFILENAME.c_str());
      _getch();
      return -1;
   }

   Mat F(FS, FS, CV_32FC1);
   for (int y=0; y<FS; y++)
      for (int x=0; x<FS; x++)
         F.at<float>(y,x) = (float)(rand() % 101) / 100.0f;
   //printf("Here is the initial matrix F:\n");
   //cout << F << "\n";


   namedWindow("roi");
   moveWindow("roi", 10,10);
   namedWindow("F");
   moveWindow("F", 10+FS+100, 10);

   unsigned int frame_counter = 0;
   for (;;)
   {
      // get next frame from video capture device
      Mat frame_col, frame_gray;
      cap >> frame_col;

      // convert color frame to gray-scale frame
      cvtColor(frame_col, frame_gray, COLOR_BGR2GRAY);

      // map values from [0,255] to [0,1]
      frame_gray.convertTo(frame_gray, CV_32FC1, 1.0 / 255.5);
      double minval, maxval;
      minMaxLoc(frame_gray, &minval, &maxval);
      //printf("min=%f max=%f\n", minval, maxval);

      // get a part of the image     
      int mx = frame_gray.size().width/2;
      int my = frame_gray.size().height/2;
      Rect roi_rect(mx - FS / 2, my - FS / 2, FS, FS);
      Mat roi(frame_gray, roi_rect);
      imshow("roi", roi);
      //printf("roi has shape %d x %d:\n", roi.size().width, roi.size().height);

      // From: http://docs.opencv.org/3.1.0/d3/d63/classcv_1_1Mat.html#afd5159655a12555b5c2725c750893a46
      // double cv::Mat::dot 	( 	InputArray  	m	) 	const
      //    Computes a dot - product of two vectors.
      //    The method computes a dot - product of two matrices.
      //    If the matrices are not single - column or single - row vectors,
      //    the top - to - bottom left - to - right scan ordering is used to
      //    treat them as 1D vectors.
      //    The vectors must have the same size and type.
      //    If the matrices have more than one channel,
      //    the dot products from all the channels are summed together.

      // compute response of current filter F
      // to input patch x (roi) by computing the scalar product: o_j = F*x
      double o_j = roi.dot( F );
      imshow("F", F);

      // adapt the filter according to Hebb's rule:
      // Delta w_ij = eta * o_i * o_j
      for (int y = 0; y<FS; y++)
         for (int x = 0; x<FS; x++)
         {
            // get old filter value w_ij
            float old_filter_value = F.at<float>(y, x);

            // get input o_i
            float o_i = roi.at<float>(y,x);
            
            float new_filter_value;

            // Hebbs' rule
            //new_filter_value = old_filter_value + LEARN_RATE * o_i * (float)o_j;

            float N;
            N = (float) frame_counter;
            //N = 100.0f;
            new_filter_value = (N*old_filter_value + o_i) / (N+1.0f);
            
            if (new_filter_value>1.0f)
               new_filter_value = 1.0f;

            // store new filter value
            F.at<float>(y, x) = new_filter_value;
         }


      // show ROI using a rectangle
      rectangle(frame_gray, roi_rect, CV_RGB(0,0,0));
            
      // show video frame
      imshow("video frame", frame_gray);
      if (waitKey(30) >= 0) break;
      
      // show video frame number from time to time
      frame_counter++;
      if (frame_counter % 100 == 0)
         printf("frames processed: %d\n", frame_counter);
   }
   
   printf("End of Hebbian learning test.\n");
   _getch();
}