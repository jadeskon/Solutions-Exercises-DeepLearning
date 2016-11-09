#include "neuron.h"


neuron::neuron()
{
}


///
/// first computes the activity of this neuron
/// (= weighted sum of its inputs), then maps
/// the activity using some transfer function to
/// an output value
///
void neuron::compute_output()
{
   // 1. compute activity of this neuron
   activation = 0.0f;

   // for all input connections ...
   for (unsigned int i = 0; i < inputs.size(); i++)
   {
      // 1.1 get the next neuron connection
      neuron_connection* c = inputs[i];

      // 1.2 get output of sending neuron
      float o = c->input_neuron->output;

      // 1.3 compute contribution of the sending neuron
      //     to this neuron's activity
      float contrib = o * c->weight;

      // 1.4 add that contribution to the overall new activation
      //     of this neuron
      activation += contrib;
   }

   // only identity as transfer function so far...
   output = activation;

} // compute_output