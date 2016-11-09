#include "mlp_fast.h"
#include <omp.h>

mlp_fast::mlp_fast(int nr_layers, ...)
{
   // 1. initialize argument pointer
   // note: typedef char* va_list
   va_list argPtr;
   va_start(argPtr, nr_layers);
   this->nr_layers = nr_layers;

   // 2. create array to store number of neurons per layer
   nr_neurons_per_layer = new int[nr_layers];
   
   for (int layer_nr = 0; layer_nr < nr_layers; layer_nr++)
   {
      // get next argument that describes desired number of neurons in next layer
      nr_neurons_per_layer[layer_nr] = va_arg(argPtr, int);
      printf("\nOk, in layer %d you want %d neurons.", layer_nr, nr_neurons_per_layer[layer_nr]);
   }

   // 3. create data structures for the MLP:
   //    for each layer l we need 
   //    ... a 1D array in order to store the netto input value of each neuron in layer l
   //    ... a 1D array in order to store the output value of each neuron in layer l
   //    ... a 2D array in order to store the weights from previous layer l-1 to layer l

   net = new float*[nr_layers];
   out = new float*[nr_layers];
   for (int layer_nr = 0; layer_nr < nr_layers; layer_nr++)
   {
      int nr_neurons = nr_neurons_per_layer[layer_nr];

      net[layer_nr] = new float[nr_neurons];
      out[layer_nr] = new float[nr_neurons];
   }

   weights = new float**[nr_layers];
   for (int l = 0; l < nr_layers-1; l++)
   {
      int nr_neurons_this_layer = nr_neurons_per_layer[l];
      int nr_neurons_next_layer = nr_neurons_per_layer[l];
      weights[l] = new float*[nr_neurons_this_layer];

      for (int i = 0; i < nr_neurons_this_layer; i++)
      {
         weights[l][i] = new float[nr_neurons_next_layer];

         // initialize weights
         for (int j = 0; j < nr_neurons_next_layer; j++)
         {
            // now weights[l][i][j] is the weight
            // from neuron i in layer l
            // to neuron j in layer +1

            // random initialization of weight drawn from a
            // uniform distribution over [-1,1]
            weights[l][i][j] = -1 + ((float)rand() / (float)RAND_MAX)*2.0f;
         }
      }
   }
   


} // mlp_fast


///
/// computes neuron activities in ascending order
/// of layers
///
void mlp_fast::forward_pass()
{
   // 1. for all but the first layer
   //    in increasing order
   for (int l = 1; l < nr_layers; l++)
   {
      // 1.1 how many neurons are there in this layer l
      //     and previous layer l-1?
      int nr_neurons_this_layer = nr_neurons_per_layer[l];
      int nr_neurons_prev_layer = nr_neurons_per_layer[l-1];

      // 1.2 for each neuron in layer l we will now
      //     compute its netto input and its output

      //#pragma omp parallel for
      for (int j=0; j<nr_neurons_this_layer; j++)
      {

         // 1.3 in order to compute the netto input for neuron j
         //     in layer l we have to compute the weighted sum of
         //     all inputs from neurons i in layer l-1
         int prevl = l-1;
         float netj = 0.0f;
         for (int i = 0; i < nr_neurons_prev_layer; i++)
         {
            // get output of neuron i
            float o = out[prevl][i];

            // get weight from neuron i to neuron j
            float w = weights[prevl][i][j];

            // update netto input of neuron j
            netj += w*o;

         } // for (all neurons i in previous layer)

         // 1.4 identity as transfer function
         out[l][j] = netj;

      } // for (all neurons j in this layer)
   } // for (all layers)

} // forward_pass


float mlp_fast::get_sum_of_output_values()
{
   float sum_out = 0.0f;
   for (int j = 0; j<nr_neurons_per_layer[nr_layers-1]; j++)
      sum_out += out[nr_layers-1][j];

   return sum_out;
}


