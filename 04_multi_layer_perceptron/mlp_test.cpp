// mlp_test.cpp
//
// In this exercise we will prepare two different implementations of
// a MLP (only the feedforward-step, no backpropagation step!).
//
// The first implementation follows an OOP approach.
// The second implementation follows tries to avoid OOP overhead in order
// to come to a faster implementation.
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

#include "../MNIST/mnist_dataset_reader.h"
#include "mlp_oop/mlp_oop.h"
#include "mlp_fast/mlp_fast.h"

using namespace cv;
using namespace std;


#define NR_OF_IMAGES_TO_TRAIN 10000
#define DO_SPEED_TEST_MLP_OOP 1
#define DO_SPEED_TEST_MLP_FAST 1

int NR_NEURONS_PER_LAYER[] = {784,40,10};
#define HORIZONTAL_LINE printf("\n######################################################");


int main()
{
   srand((unsigned int)time(NULL));
   //srand(140376);
      
   // 1. read in MNIST data   
   string path_to_extracted_mnist_files = "V:\\01_job\\00_vorlesungen_meine\\17_deep_learning_ws1617\\07_datasets\\01_mnist\\extracted";
   mnist_dataset_reader* my_reader = new mnist_dataset_reader(path_to_extracted_mnist_files);

   // 2. show me some example images of the training data from the MNIST data set
   Mat* samples_as_image = my_reader->get_board_of_sample_images(
      my_reader->get_train_images(),
      my_reader->get_train_labels(),
      60000);
   imshow("Some sample MNIST training images", *samples_as_image);
   cv::waitKey(100);


   // 3. speed test of two MLP implementations   
   time_t starttime, endtime;
   int elapsedtime_mlp_oop, elapsedtime_mlp_fast;
   unsigned char** train_imgs = my_reader->get_train_images();
   unsigned char* train_labels = my_reader->get_train_labels();


   // 3.1 test mlp_oop speed?
   if (DO_SPEED_TEST_MLP_OOP)
   {
      HORIZONTAL_LINE
      printf("\nFeedforward Speed Test MLP implementation #1: mlp_oop");
      HORIZONTAL_LINE
      mlp_oop my_mlp_oop;
      my_mlp_oop.add_neuron_layer(NR_NEURONS_PER_LAYER[0]);
      my_mlp_oop.add_neuron_layer(NR_NEURONS_PER_LAYER[1]);
      my_mlp_oop.add_neuron_layer(NR_NEURONS_PER_LAYER[2]);
      
            
      starttime = clock();            
      for (int train_img_nr=1; train_img_nr<= NR_OF_IMAGES_TO_TRAIN; train_img_nr++)
      {
         // "guess" a random image index
         int rnd_idx = rand() % 60000;

         // feed the gray scale values into the MLP
         unsigned char* input_img = train_imgs[rnd_idx];
         for (int i = 0; i < 28 * 28; i++)
         {
            // set neuron output according to input image pixel value
            float gray_value = gray_value = (float)input_img[i] / 255.0f; // will be in [0,1]

            my_mlp_oop.all_layers[0]->all_neurons_in_this_layer[i]->output = gray_value;
         }

         // forward pass
         my_mlp_oop.forward_pass();

         // from time to time show the progress
         if (train_img_nr % (NR_OF_IMAGES_TO_TRAIN/10) == 0)
         {
            printf("\nNumber of images forwarded: %d", train_img_nr);
         }

      } // for (traing_img_nr)
      endtime = clock();
      elapsedtime_mlp_oop = (int) (endtime-starttime);
      printf("\n-->Time needed: %dms\n", elapsedtime_mlp_oop);
   } // if (speed test mlp_oop?)


   // 3.2 test mlp_fast speed?
   if (DO_SPEED_TEST_MLP_FAST)
   {
      HORIZONTAL_LINE
      printf("\nFeedforward Speed Test MLP implementation #2: mlp_fast");
      HORIZONTAL_LINE
      mlp_fast my_mlp_fast(3,
                           NR_NEURONS_PER_LAYER[0],
                           NR_NEURONS_PER_LAYER[1],
                           NR_NEURONS_PER_LAYER[2]
                           );
      
      starttime = clock();
      for (int train_img_nr = 1; train_img_nr <= NR_OF_IMAGES_TO_TRAIN; train_img_nr++)
      {
         // "guess" a random image index
         int rnd_idx = rand() % 60000;

         // feed the gray scale values into the MLP
         unsigned char* input_img = train_imgs[rnd_idx];
         for (int i = 0; i < 28 * 28; i++)
         {
            // set neuron output according to input image pixel value
            float gray_value = (float)input_img[i] / 255.0f; // will be in [0,1]

            my_mlp_fast.out[0][i] = gray_value;

            //my_mlp_oop.all_layers[0]->all_neurons_in_this_layer[i]->output = gray_value;
         }

         // forward pass
         my_mlp_fast.forward_pass();

         // from time to time show the progress
         if (train_img_nr % (NR_OF_IMAGES_TO_TRAIN/10) == 0)
         {
            printf("\nNumber of images forwarded: %d", train_img_nr);
         }

      } // for (traing_img_nr)
      endtime = clock();
      elapsedtime_mlp_fast = (int)(endtime - starttime);
      printf("\n-->Time needed: %dms\n", elapsedtime_mlp_fast);
   } // if (speed test mlp_fast?)


   // 4. if both speed tests were executed,
   //    we can compute how much faster mlp_fast is compared to mlp_oop
   if (DO_SPEED_TEST_MLP_OOP && DO_SPEED_TEST_MLP_FAST)
   {
      float ratio = 1.0f / ((float)elapsedtime_mlp_fast / (float)elapsedtime_mlp_oop);
      printf("\nThe implementation mlp_fast is %.2f times faster than mlp_oop", ratio);
   }


   cout << "\n\nGood bye! Press a key to exit\n";
   _getch();

} // main