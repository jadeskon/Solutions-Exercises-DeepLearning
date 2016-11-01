#include "Perceptron.h"
#include <stdio.h>
#include <stdlib.h> // for rand()

Perceptron::Perceptron(int nr_inputs, int nr_outputs, float learning_rate)
{
   // 1. store the information about the number
   //    of input values & output neurons
   //    and desired learning rate
   this->nr_inputs     = nr_inputs;
   this->nr_outputs    = nr_outputs;
   this->learning_rate = learning_rate;

   // 2. create array for storing input values
   //    why nr_inputs+1? because we need an "on"-neuron
   //    that always gives input 1.0 for the bias weight
   input_values = new float[nr_inputs+1];
   input_values[nr_inputs] = 1.0f; // set input value of on-neuron

   // 3. create array for storing output values
   output_values = new float[nr_outputs];

   // 4. create 2D array for storing weights
   //    for each output neuron we store a 1D array
   //    with a weight for each input AND a weight for the bias
   weights = new float*[nr_outputs];
   for (int out=0; out<nr_outputs; out++)
      weights[out] = new float[nr_inputs+1];

   // 5. initialize all the weights
   for (int out = 0; out<nr_outputs; out++)
   {
      for (int in = 0; in < nr_inputs+1; in++)
      {
         // initialize with a random weight from [-1,1]
         //weights[out][in] = ((float)(rand() % 201) / 100.0f) - 1.0f;

         // initialize with a random weight from [0,1]
         weights[out][in] = (float)(rand() % 101) / 100.0f;
      }
   }

} // constructor for class Perceptron


///
/// assuming that the input_values array has been set
/// from outside, we will compute for each output neuron
/// its current value
///
void Perceptron::compute_outputs()
{
   for (int out = 0; out<nr_outputs; out++)
   {
      // 1. compute weighted sum of inputs = netto input
      float sum = 0.0f;
      for (int in = 0; in < nr_inputs+1; in++)
      {         
         float contrib = weights[out][in] * input_values[in];
         sum += contrib;
      }

      // 2. store output value
      output_values[out] = sum;
   }
    
} // compute_outputs


///
/// assuming we use an 1-out-N encoding for
/// the output neurons, the Perceptron's classification
/// for the current input values is the output
/// neuron with the largest output value
///
int Perceptron::get_classification()
{
   int max_out_idx = 0;
   for (int out = 1; out<nr_outputs; out++)
   {
      if (output_values[out]>output_values[max_out_idx])
         max_out_idx = out;
   }

   return max_out_idx;

} // get_classification


void Perceptron::show_debug_info()
{
   printf("\nOutput values of neurons: ");
   for (int out = 0; out<nr_outputs; out++)
   {
      printf("\noutput #%d = %f", out, output_values[out]);
   }
      
} // show_debug_info


 /// Remember the Perceptron learning rule from the lecture:
 /// ∆𝑤_𝑖 = 𝛼(𝑡−𝑜𝑢𝑡)𝑥_𝑖
 ///
void Perceptron::learn(float* teacher_vector)
{
   // for all output neurons do the training independently ...
   for (int o=0; o<nr_outputs; o++)
   {
      // actual output of neuron?
      float out = output_values[o];

      // desired output of neuron?
      float t = teacher_vector[o];

      // difference between teacher and actual output?
      float diff = t-out;


      // adapt each input weight according to Perceptron learning rule
      // ∆𝑤_𝑖= 𝛼(𝑡−𝑜𝑢𝑡)𝑥_𝑖
      for (int i = 0; i<nr_inputs+1; i++)
      {
         // input value x_i?
         float x_i = input_values[i];

         // compute weight change
         float delta_w_i = learning_rate * diff * x_i;

         // do weight change
         weights[o][i] += delta_w_i;

         // restrict weight to be in [0,1]?
         /*
         if (weights[o][i]<0.0f)
            weights[o][i] = 0.0f;
         if (weights[o][i]>1.0f)
            weights[o][i] = 1.0f;
         */


      } // for (all inputs i)

   } // for (all output neurons o)

} // learn