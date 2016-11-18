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


#define DO_UNIT_TEST_IDENTITY 0
#define DO_UNIT_REGRESSION 1


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
   
   cout << "\n\nGood bye! Press a key to exit\n";
   _getch();

} // main