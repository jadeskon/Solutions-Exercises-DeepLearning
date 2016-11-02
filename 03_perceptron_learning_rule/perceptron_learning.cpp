// perceptron_learning.cpp
//
// Small program that learns a Perceptron classifier for MNIST test data
// images using the MNIST training data images and the Perceptron learning rule.
//
// As input we will use the output of a filter bank based on handcrafted
// (not learned yet! will come later in the section CNN...) filter matrices
// or randomly generated filters.
//
// Remember the Perceptron learning rule from the lecture:
// ∆𝑤_𝑖 = 𝛼(𝑡−𝑜𝑢𝑡)𝑥_𝑖
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

#include "mnist_dataset_reader.h"
#include "perceptron.h"

using namespace cv;
using namespace std;

#define USE_FILTER_BANK 0
#define USE_VALUES_FROM_RANGE_0_TO_255 0

int NR_INPUTS;  // depends whether we use the filter bank as input or not

#define LEARNRATE0 0.00001f
#define LEARNRATE1 0.0001f
#define LEARNRATE2 0.001f
#define LEARNRATE3 0.01f
#define LEARNRATE4 0.1f

Perceptron* my_pct;
mnist_dataset_reader* my_reader;
vector<Mat*> filter_bank;
Mat* input_img_gray;

Point anchor = Point(-1, -1);
// A value to be added to each pixel during the convolution. By default it is 0
double delta = 0;

// depth of filter response matrix will be the same as that of source matrix 
//int ddepth = -1;
int ddepth = CV_32FC1;





void feed_perceptron(int img_idx, unsigned char** img_set)
{
   if (USE_FILTER_BANK == 0)
   {
      unsigned char* input_img = img_set[img_idx];
      for (int i = 0; i < 28 * 28; i++)
      {
         // set neuron output according to input image pixel value
         float gray_value;
         if (USE_VALUES_FROM_RANGE_0_TO_255)
            gray_value = (float)input_img[i];          // will be in [0,255]
         else
            gray_value = (float)input_img[i] / 255.0f; // will be in [0,1]
         

         my_pct->input_values[i] = gray_value;

      } // for (all input neurons to feed with values)
   }
   else
   {

      // get the input image to feed into the Perceptron as a cv::Mat
      // (such that we can filter it in the next step using cv::filter2D())
      my_reader->get_mnist_image_as_cvmat(input_img_gray, img_set, img_idx);
      
      // Now filter the current example MNIST digit with all filters in the filter bank      
      int input_nr = 0;
      for (uint filter_nr = 0; filter_nr < filter_bank.size(); filter_nr++)
      {
         // get the filter
         Mat* next_filter_kernel = filter_bank.at(filter_nr);

         // The function filter2D uses the DFT - based algorithm in case of sufficiently
         // large kernels (11 x 11 or larger) and the direct algorithm
         // (that uses the engine retrieved by createLinearFilter()) for small kernels
         cv::Mat filter_result_gray;
         cv::filter2D(*input_img_gray, filter_result_gray,
            ddepth, *next_filter_kernel,
            anchor, delta, BORDER_DEFAULT);

         // see http://stackoverflow.com/questions/12023958/what-does-cvnormalize-src-dst-0-255-norm-minmax-cv-8uc1 
         normalize(filter_result_gray, filter_result_gray, 0.0, 1.0, NORM_MINMAX, -1);
         //minMaxLoc(filter_result_gray, &minval, &maxval);
         //printf("min=%.3f, max=%.3f\n", minval, maxval);

         /*
         printf("filter result matrix size is %d x %d\n",
         filter_result_gray.size().width,
         filter_result_gray.size().height );
         */

         // now feed the Perceptron with the filter matrix values
         for (int y = 0; y < 28; y++)
         {
            for (int x = 0; x < 28; x++)
            {
               // set neuron output according to input image pixel value
               float filter_response_value = filter_result_gray.at<float>(y, x); // will be in [0,1]
               my_pct->input_values[input_nr] = filter_response_value;
               input_nr++;
            }
         }

      } // for (all filters in filter bank)

   } // if (USE_FILTER_BANK==0)

} // feed_perceptron




void compute_classification_rate_on_mnist_test_data_set(      
      ofstream* my_file,
      int learn_step_nr)
{
   unsigned char** test_imgs = my_reader->get_test_images();
   int classified_correct = 0;
   for (int sample_nr = 0; sample_nr < my_reader->nr_test_images_read; sample_nr++)
   {
      // 1. get pointer to next test vector
      unsigned char* test_img = test_imgs[sample_nr];

      // 2. get ground truth label information
      int ground_truth_class = (int)my_reader->get_test_labels()[sample_nr];

      // 3. feed Perceptron with input values
      feed_perceptron(sample_nr, test_imgs);

      // 4. forward pass
      my_pct->compute_outputs();

      // 5. identify output neuron with highest output value
      int winner_index = my_pct->get_classification();
      //printf("predicted vs. is: %d vs. %d\n", winner_index, ground_truth_class);

      // 6. did we correctly classify?
      if (winner_index == ground_truth_class)
         classified_correct++;

      if (0)
      {
         my_pct->show_debug_info();
         _getch();
      }

   } // for (all test vectors)

   // show classification rate
   float classification_rate = (float)classified_correct / (float)my_reader->nr_test_images_read;
   printf("\n\tClassifcation rate on %d test images.\n\tCorrectly classified %d images --> rate = %.3f\n",
      my_reader->nr_test_images_read, classified_correct, classification_rate);

   *my_file << learn_step_nr << " " << classified_correct << " " << classification_rate << endl;

} // compute_classification_rate_on_mnist_test_data_set







int main()
{
   // for storing a MNIST digit as a 28x28 pixel cv::Mat
   input_img_gray = new cv::Mat(28,28, CV_8UC1);

   ofstream* my_file = new ofstream("V:\\tmp\\perceptron_learning.txt");

   srand((unsigned int)time(NULL));
   //srand(140376);

   // 1. read in MNIST data
   string path_to_extracted_mnist_files = "V:\\01_job\\00_vorlesungen_meine\\17_deep_learning_ws1617\\07_datasets\\01_mnist\\extracted";
   my_reader = new mnist_dataset_reader(path_to_extracted_mnist_files);

   // 2. show me some example images of the training data from the MNIST data set
   Mat* samples_as_image = my_reader->get_board_of_sample_images(
                              my_reader->get_train_images(),
                              my_reader->get_train_labels(),
                              60000);
   imshow("Some sample MNIST training images", *samples_as_image);
   cv::waitKey(100);


   // 3. generate a filter bank   
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
      //cout << *kernel;
      filter_bank.push_back(kernel);
   }

      
   // 4. create a Perceptron   
   if (USE_FILTER_BANK==0)
   {
      NR_INPUTS = 28*28;      
   }
   else
   {
      int nr_filters = filter_bank.size();
      NR_INPUTS = nr_filters * 28 * 28;
   }
   my_pct = new Perceptron(NR_INPUTS, 10, LEARNRATE1);
   

   // 5. show classification rate on MNIST test data set (10.000 images)
   //    of randomly generated Perceptron before learning
   compute_classification_rate_on_mnist_test_data_set(my_file, 0);


   // 6. Perceptron training
   //
   // in each loop we will select randomly one training image,
   // feed it into the Perceptron and then adapt the weights
   printf("\nStarting Perceptron training ...");
   unsigned char** train_imgs = my_reader->get_train_images();
   unsigned char* train_labels = my_reader->get_train_labels();
   int nr_images_trained = 0;
   float teacher_vec[10];
   while (1)
   {      

      // 6.1 "guess" a random image index
      int rnd_idx = rand() % 60000;
            
      // 6.2 feed the image directly into the Perceptron or
      //     use filter bank responses as input?
      feed_perceptron(rnd_idx, train_imgs);
      
      // 6.3 forward pass
      my_pct->compute_outputs();
      //my_pct->show_debug_info();
      //_getch();

      // 6.4 prepare teacher vector and do a single Perceptron learning step
      int ground_truth_label = train_labels[rnd_idx];
      for (int i = 0; i < 10; i++)
      {
         if (i==ground_truth_label)
            teacher_vec[i] = 1.0;
         else
            teacher_vec[i] = 0.0;
      }
      my_pct->learn( teacher_vec );


      // 6.5 keep track of number of images trained
      nr_images_trained++;
      if (nr_images_trained % 10000==0)
      {
         printf("\nNumber of images trained: %d", nr_images_trained);
      }

      // 6.6 from time to time recompute classification rate on test dataset
      //     (not training dataset!), in order to see whether we are learning something
      if (nr_images_trained % 50000 == 0)
      {
         compute_classification_rate_on_mnist_test_data_set(my_file, nr_images_trained);
      }
            
   } // while (training)



   cout << "Good bye! Press a key to exit\n";
   _getch();

} // main