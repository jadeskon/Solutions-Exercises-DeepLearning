// backprop_test.cpp
//
// In this exercise we focus on implementing the
// Backpropagation algorithm, which allows to adapt the weights
// in a Multi Layer Perceptron (MLP) even if the weights are
// targeting hidden neurons.
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


using namespace cv;
using namespace std;


#define NR_OF_IMAGES_TO_TRAIN 10000
#define NR_INPUT_NEURONS 784
#define NR_HIDDEN_NEURONS_LAYER1 10
#define NR_OUTPUT_NEURONS 10

#define DO_UNIT_TEST_IDENTITY 1
#define DO_UNIT_REGRESSION 0
#define DID_SOME_UNIT_TEST DO_UNIT_TEST_IDENTITY||DO_UNIT_REGRESSION


int main()
{
   srand((unsigned int)time(NULL));
   //srand(140376);

   // do some unit tests to check wether my MLP & Backprop implementation (still) works?
   if (DO_UNIT_TEST_IDENTITY)
   {
      mlp_oop mlp;
      mlp.unit_test_identity();
   }

   if (DO_UNIT_REGRESSION)
   {
      mlp_oop mlp;
      mlp.unit_test_regression();
   }
   
   if (DID_SOME_UNIT_TEST)
      exit(0);



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


   // 3. create MLP
   mlp_oop my_mlp_oop;
   my_mlp_oop.add_neuron_layer(0,   NR_INPUT_NEURONS         , true,  transferfunc_type::tf_type_identity );
   my_mlp_oop.add_neuron_layer(1,   NR_HIDDEN_NEURONS_LAYER1 , true,  transferfunc_type::tf_type_logistic );
   my_mlp_oop.add_neuron_layer(2,   NR_OUTPUT_NEURONS        , false, transferfunc_type::tf_type_logistic );
   my_mlp_oop.show_mlp_structure_information();


   // 4. prepare access to training images & ground truth labels
   unsigned char** train_imgs = my_reader->get_train_images();
   unsigned char* train_labels = my_reader->get_train_labels();

   // 5. prepare a teacher vector which can be filled with the
   //    desired output values for each output neuron
   float* teacher_vector = new float[NR_OUTPUT_NEURONS];
     

   // 6. now randomly choose training pairs,
   //    feedforward the input vector
   //    backpropagate the error signals for the neurons,
   //    and adapt the weights 
   for (int train_img_nr = 1; train_img_nr <= NR_OF_IMAGES_TO_TRAIN; train_img_nr++)
   {
      // 6.1 "guess" a random image index
      int rnd_idx = rand() % 60000;

      // 6.2 feed the gray scale values into the MLP
      unsigned char* input_img = train_imgs[rnd_idx];
      for (int i = 0; i < 28 * 28; i++)
      {
         // set neuron output according to input image pixel value
         float gray_value = gray_value = (float)input_img[i] / 255.0f; // will be in [0,1]

         my_mlp_oop.all_layers[0]->all_neurons_in_this_layer[i]->out = gray_value;
      }

      // 6.3 forward pass
      my_mlp_oop.forward_pass();

      // 6.4 prepare teacher vector and do a single Perceptron learning step
      int ground_truth_label = train_labels[rnd_idx];
      for (int i = 0; i < 10; i++)
      {
         if (i == ground_truth_label)
            teacher_vector[i] = 1.0;
         else
            teacher_vector[i] = 0.0;
      }

      // 6.5 do a single backpropagation step
      my_mlp_oop.backpropagation_compute_error_signals( teacher_vector );
      my_mlp_oop.backpropagation_change_weights();

      // 6.6 from time to time show the progress
      if (train_img_nr % (NR_OF_IMAGES_TO_TRAIN / 10) == 0)
      {
         printf("\nNumber of images forwarded: %d", train_img_nr);
      }

   } // for (traing_img_nr)
     
     
   cout << "\n\nGood bye! Press a key to exit\n";
   _getch();

} // main