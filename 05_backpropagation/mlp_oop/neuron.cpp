#include "neuron.h"



neuron::neuron(transferfunc_type transferfunc_to_use)
{
   this->transferfunc_to_use = transferfunc_to_use;
}


///
/// depending on the transfer function type set for
/// this neuron, f(x) will return the transfer function
/// value of x
///
float neuron::f(float x)
{
   switch (transferfunc_to_use)
   {
      case tf_type_identity : return x; break;
      case tf_type_logistic : return logistic_func(x); break;
      case tf_type_tanh     : return tangens_hyperbolicus(x); break;
      case tf_type_relu     : return relu(x); break;
      default               : return x; break;
   }
}


///
/// depending on the transfer function type set for
/// this neuron, f_derive(x) will return the value of 
/// the derivative function at x
///
float neuron::f_derived(float x)
{
   switch (transferfunc_to_use)
   {
       case tf_type_identity : return 1.0f; break;
       case tf_type_logistic : return logistic_func_derived(x); break;
       case tf_type_tanh     : return tangens_hyperbolicus_derived(x); break;
       case tf_type_relu     : return relu_derived(x); break;
       default               : return 1.0f; break;
   }
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
   net = 0.0f;

   // for all input connections ...
   for (unsigned int i = 0; i < inputs.size(); i++)
   {
      // 1.1 get the next neuron connection
      neuron_connection* c = inputs[i];

      // 1.2 get output of sending neuron
      float o = c->input_neuron->out;

      // 1.3 compute contribution of the sending neuron
      //     to this neuron's activity
      float contrib = o * c->weight;

      // 1.4 add that contribution to the overall new activation
      //     of this neuron
      net += contrib;
   }

   // map netto input using the transfer function to some output value
   out = f( net );

} // compute_output


///
/// run through all incoming connections of this neuron
/// and adapt the weights
///
void neuron::change_weights()
{
   // for all input connections ...
   for (unsigned int i = 0; i < inputs.size(); i++)
   {
      //get the next neuron connection
      neuron_connection* c = inputs[i];

      // TATARATAAA! NOW THE ACTUAL LEARNING HAPPENS! :)
      c->weight += c->delta_w;
   }
   
} // change_weights