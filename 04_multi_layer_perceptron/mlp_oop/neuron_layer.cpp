#include "neuron_layer.h"
#include "neuron.h"

///
/// create a new neuron layer by generating
/// the desired number of neurons
/// and store a pointer to each newly generated neuron
///
neuron_layer::neuron_layer(int nr_neurons)
{
   // 1. store nr of neurons in this layer
   this->nr_neurons = nr_neurons+1;

   // 2. create the desired number of neurons
   for (int i = 0; i < nr_neurons; i++)
   {
      // create a new neuron
      neuron* n = new neuron();

      // add the pointer to the newly generated neuron
      // to the list of neurons in this layer
      all_neurons_in_this_layer.push_back( n );
   }

   // 3. 
   // Do not forget to create an extra neuron
   // the "on" neuron, that always has output 1.0
   // in order to avoid extra treetment of bias parameter learning   
   neuron* n = new neuron();
   n->output = 1.0;
   all_neurons_in_this_layer.push_back(n);
}


///
/// let all neurons in this layer compute their
/// output values
/// but not the last neuron, which is the "always on" neuron!
///
void neuron_layer::compute_outputs()
{
   for (int i = 0; i < nr_neurons-1; i++)
   {
      all_neurons_in_this_layer[i]->compute_output();
   }

} // compute_outputs



///
/// helper function for development
/// shows e.g. the output value of each neuron in this layer
void neuron_layer::show_debug_infos()
{
   for (int i = 0; i < nr_neurons; i++)
   {
      neuron* n = all_neurons_in_this_layer[i];
      printf("%d: out=%.2f   ", i, n->output);
   }
   printf("\n");

} // show_debug_infos