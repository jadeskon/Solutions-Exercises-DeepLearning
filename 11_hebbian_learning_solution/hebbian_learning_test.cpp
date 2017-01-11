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
// Visualize the features after learning.
// Are they meaningful?
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
int FS = 40; // filter size

const int NF = 100; // number of filters
Mat* all_filters[NF];
int filter_adapt_counter[NF];

const int VISU_WIDTH  = 500;
const int VISU_HEIGHT = 700;
Mat* visu = new Mat(VISU_HEIGHT, VISU_WIDTH, CV_32FC1);

void show_filter();


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
      
   for (int i=0; i<NF; i++)
   {
      Mat* F = new Mat(FS, FS, CV_32FC1);
      for (int y=0; y<FS; y++)
         for (int x=0; x<FS; x++)
            F->at<float>(y,x) = (float)(rand() % 101) / 100.0f;

      all_filters[i] = F;
      filter_adapt_counter[i] = 0;
   }
   

   namedWindow("roi");
   moveWindow("roi", 10,10);

   vector<Rect> rois_to_analyze;
   
   unsigned int frame_counter = 0;
   for (;;)
   {
      // 1. get next frame from video capture device
      Mat frame_col, frame_gray;
      cap >> frame_col;

      // 2. do we already have set up the list of ROIs to analyze?
      //    note: we can init this list only after getting the first
      //          video frame since we need to know the frame dimensions
      if (rois_to_analyze.size() == 0)
      {
         for (int y = 0; y + FS < frame_col.size().height; y+=FS)
         {
            for (int x = 0; x + FS < frame_col.size().width; x+=FS)
            {
               Rect r(x,y,FS,FS);
               rois_to_analyze.push_back( r );
            }
         }
         printf("generated list of ROIs to analyze. Nr of ROIs: %d\n",
            rois_to_analyze.size());
      }

      // 3. convert color frame to gray-scale frame
      cvtColor(frame_col, frame_gray, COLOR_BGR2GRAY);

      // 4. map values from [0,255] to [0,1]
      frame_gray.convertTo(frame_gray, CV_32FC1, 1.0 / 255.5);
      double minval, maxval;
      minMaxLoc(frame_gray, &minval, &maxval);
      //printf("min=%f max=%f\n", minval, maxval);


      // 5. for all ROIs to analyze
      for (unsigned int roi_idx = 0; roi_idx<rois_to_analyze.size(); roi_idx++)
      {
         Mat roi(frame_gray, rois_to_analyze[roi_idx]);
         if (roi_idx==50)
            imshow("roi", roi);
         //printf("roi has shape %d x %d:\n", roi.size().width, roi.size().height);

         // 5.1 now compute filter responses for all filters
         //     and determine the filter that best matches for that ROI
         int winner_idx = 0;
         double min_dist = 0.0;

         for (int i = 0; i < NF; i++)
         {
            Mat F = *all_filters[i];
            double dist = norm(F, roi, NORM_L2);
            if ((i == 0) || (dist<min_dist))
            {
               min_dist = dist;
               winner_idx = i;
            }
         }

         // at start of filter learning
         // move all filters towards natural filters!
         bool init_phase = frame_counter < 300;
         if (init_phase)
            winner_idx=rand() % NF;
           
         // 5.2 adapt the winning filter a little bit towards the input
         //printf("adapting filter %d\n", winner_idx);
         Mat* F = all_filters[winner_idx];
         for (int y = 0; y<FS; y++)
         {
            for (int x = 0; x<FS; x++)
            {
               // get old filter value
               float old_filter_value =F->at<float>(y, x);

               float o_i = roi.at<float>(y,x);
            
               float new_filter_value;
               float N;
            
               if (init_phase)
                  N = 10.0f;
               else
                  //N = (float)frame_counter; // first try this
                  N = (float)filter_adapt_counter[winner_idx]; // then try this
               new_filter_value = (old_filter_value*N + o_i) / (N+1.0f);

               if (new_filter_value>1.0f)
                  new_filter_value = 1.0f;

               // store new filter value
               F->at<float>(y, x) = new_filter_value;
            }
         }
         if (!init_phase)
            filter_adapt_counter[winner_idx]++;

      } // for (all rois to analyze ...)

      // 6. show all filters
      show_filter();


      // 7. show ROIs using rectangles
      for (unsigned int i=0; i<rois_to_analyze.size(); i++)
      {         
         rectangle(frame_gray, rois_to_analyze[i], CV_RGB(0,0,0));
      }
            
      // 8. show video frame
      imshow("video frame", frame_gray);
      if (waitKey(1) >= 0) break;
      
      // 9. show video frame number from time to time
      frame_counter++;
      if (frame_counter % 100 == 0)
         printf("frames processed: %d\n", frame_counter);
   }
   
   printf("End of Hebbian learning test.\n");
   _getch();

} // main


void show_filter()
{
   *visu = 0;

   bool still_space_for_drawing = true;

   int y = 0;
   int x = 0;
   const int some_space = 7;
   int filter_idx = -1;
   while (still_space_for_drawing && filter_idx<NF-1)
   {
      // get next filter index
      filter_idx++;

      // get pointer to filter
      Mat* F = all_filters[filter_idx];

      // copy visualization of image of digit to board      
      Mat* roi = new Mat(*visu, Rect(x, y, FS, FS));
      F->copyTo(*roi);
      
      // draw filter index on visualization board as well
      char txt[100];
      sprintf(txt, "%d (%d)", filter_idx, filter_adapt_counter[filter_idx]);
      putText(*visu,
         txt,
         Point(x, y + FS + 8),
         FONT_HERSHEY_SIMPLEX, 0.25, // font face and scale
         CV_RGB(255, 255, 255), // white
         1); // line thickness and type

      // compute next draw x position
      x += FS + some_space;
      
      // line break before next visualization of filter?
      if (x + 2 * FS + some_space >= VISU_WIDTH)
      {
         // yes! line break
         x = 0;
         y += FS + 2 * some_space;

         // is there still enough space for rendering another filter?
         if (y + FS + some_space >= VISU_HEIGHT)
         {
            // no space for rendering left
            still_space_for_drawing = false;
         }

      } // if

   } // while (still_space_for_drawing)

   imshow("filter bank visualization", *visu);

} // show_filter()