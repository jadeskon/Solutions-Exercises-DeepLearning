#include "neuron_layer.h"
#include "neuron.h"

///
/// create a new neuron layer by generating
/// the desired number of neurons
/// and store a pointer to each newly generated neuron
///
neuron_layer::neuron_layer(int id, int nr_neurons, bool create_bias_neuron, transferfunc_type transfer_func_to_use)
{
   this->layer_id = id;

   // 1. store nr of neurons in this layer
   if (create_bias_neuron)
   {
      this->nr_neurons = nr_neurons+1;
      this->has_bias_neuron = true;
   }
   else
   {
      this->nr_neurons = nr_neurons;
      this->has_bias_neuron = false;
   }

   // 2. create the desired number of neurons
   for (int i = 0; i < nr_neurons; i++)
   {
      // create a new neuron
      neuron* n = new neuron( transfer_func_to_use );

      // add the pointer to the newly generated neuron
      // to the list of neurons in this layer
      all_neurons_in_this_layer.push_back( n );
   }

   // 3. 
   // Do not forget to create an extra neuron
   // the "on" neuron, that always has output 1.0
   // in order to avoid extra treetment of bias parameter learning   
   if (create_bias_neuron)
   {
      neuron* n = new neuron( transferfunc_type::tf_type_identity );
      n->out = 1.0;
      all_neurons_in_this_layer.push_back(n);
   }
   
} // ctor neuron_layer



///
/// let all neurons in this layer compute their
/// output values
/// (!) but not the last neuron, which is the "always on" neuron (!)
///     (we should not overwrite the output of this on-neuron
///      which was preset to 1.0 in order to represent a neuron
///      that always has output 1.0)
///
void neuron_layer::compute_outputs()
{
   int N;
   if (has_bias_neuron)
      N = nr_neurons-1;
   else
      N = nr_neurons;
   for (int i = 0; i < N; i++)
   {
      all_neurons_in_this_layer[i]->compute_output();
   }

} // compute_outputs



///
/// helper function for code development
/// shows e.g. the output value of each neuron in this layer
///
void neuron_layer::show_debug_infos()
{
   for (int i = 0; i < nr_neurons; i++)
   {
      neuron* n = all_neurons_in_this_layer[i];
      printf("%d: out=%.2f   ", i, n->out);
   }
   printf("\n");

} // show_debug_infos



/// 
/// we will compute one error signal 𝛿_𝑗 for each neuron in this layer
/// and already compute the weight change Δ𝑤_𝑘𝑗 (but yet not adapt the weight!)
///
/// Backprop weight update formula for output neurons j:
/// Δ𝑤_𝑘𝑗 = −𝛼 (𝑦_𝑗−𝑡_𝑗) ∗ 𝑓′(𝑛𝑒𝑡𝑗) ∗ 𝑦_𝑘
///       = 𝛼*𝛿_𝑗∗𝑦_𝑘
///  --> 𝛿_𝑗 = -(𝑦_𝑗−𝑡_𝑗) ∗ 𝑓′(𝑛𝑒𝑡𝑗)
///
/// Backprop weight update formula for hidden neurons j:
/// Δ𝑤_𝑘𝑗 = −𝛼 (−∑_(𝑖 = 1)^𝑁 (δ_𝑖 𝑤_𝑗𝑖)) ∗ 𝑓′(𝑛𝑒𝑡𝑗) ∗ 𝑦_𝑘
///       = 𝛼*𝛿_𝑗∗𝑦_𝑘
///  --> 𝛿_𝑗 = ∑_(𝑖 = 1)^𝑁 (δ_𝑖 𝑤_𝑗𝑖)) ∗ 𝑓′(𝑛𝑒𝑡𝑗)
///
void neuron_layer::compute_error_signals_and_weight_change(float* teacher_vector,
                                                           neuron_layer* next_layer,
                                                           float learn_rate)
{
   // 1. is this layer an ouput layer?
   //    it is one, if there is no next layer
   bool is_output_layer;
   if (next_layer==NULL)
      is_output_layer = true;
   else
      is_output_layer = false;
   
   // 2. for all neurons in this layer   
   for (int j = 0; j < nr_neurons; j++)
   {
      // 2.1 get next neuron
      neuron* neuron_j = all_neurons_in_this_layer[j];

      // 2.2 compute error signal for this neuron j
      if (is_output_layer)
      {
         // neuron j is output neuron:

         // 𝛿_𝑗 = -(𝑦_𝑗−𝑡_𝑗)∗ 𝑓′(𝑛𝑒𝑡𝑗)
         neuron_j->error_signal = -(neuron_j->out - teacher_vector[j]) * neuron_j->f_derived(neuron_j->net);

      }
      else
      {
         // neuron j is a hidden neuron:

         // 𝛿_𝑗 = ∑_(𝑖 = 1)^𝑁 (δ_𝑖 𝑤_𝑗𝑖)) ∗ 𝑓′(𝑛𝑒𝑡𝑗)
         float sum_of_weighted_error_signals = 0.0f;
         int N = next_layer->nr_neurons;
         for (int i = 0; i < N; i++)
         {
            // get next neuron i in following layer
            neuron* neuron_i = next_layer->all_neurons_in_this_layer[i];

            // is it the always-on neuron (which has no input connections)?
            // if yes, ignore it!
            if (neuron_i->inputs.size() == 0)
               continue;

            // get current value of weight w_ji from neuron j to neuron i
            float w_ji = neuron_i->inputs[j]->weight;

            sum_of_weighted_error_signals += neuron_i->error_signal * w_ji;
         }

         // 𝛿_𝑗 = ∑_(𝑖 = 1)^𝑁 (δ_𝑖 𝑤_𝑗𝑖)) ∗ 𝑓′(𝑛𝑒𝑡𝑗)
         neuron_j->error_signal = sum_of_weighted_error_signals * 
                                  neuron_j->f_derived(neuron_j->net);

      } // if (is neuron j output or hidden neuron?)

      // 2.3 for all incoming connections k->j of this neuron
      for (unsigned int k = 0; k < neuron_j->inputs.size(); k++)
      {
         // get next incoming connection
         neuron_connection* con = neuron_j->inputs[k];

         // compute next weight change for this weight Δ𝑤_𝑘𝑗
         con->delta_w = learn_rate * neuron_j->error_signal * con->input_neuron->out;

      } // for (all incoming connections)

      
   } // for (all neurons j in current layer)
        
} // compute_error_signals_and_weight_change


///
/// In compute_error_signals_and_weight_change() we have computed
/// a "error signal" for each neuron (even for non-output neurons)
/// and a weight change delta_w
///
/// Now we will really change the weights, i.e., add delta_w to each
/// weight
///
void neuron_layer::change_weights()
{
   for (int j = 0; j < nr_neurons; j++)
   {
      // get next neuron
      neuron* neuron_j = all_neurons_in_this_layer[j];

      // tell the neuron to adapt its incoming weights

      neuron_j->change_weights();
   }
    
} // change_weights



///
/// for debugging: show minimum & maximum values
///                of all incoming connection weights
///                and neuron netto input / output values
///
void neuron_layer::show_min_max_values()
{
   float min_weight, max_weight;
   float min_net, max_net;
   float min_out, max_out;
   min_weight=max_weight=min_net=max_net=min_out=max_out=0.0f;

   for (int j = 0; j < nr_neurons; j++)
   {
      // get next neuron
      neuron* neuron_j = all_neurons_in_this_layer[j];

      // new net extreme?
      if (neuron_j->net < min_net)
         min_net = neuron_j->net;
      if (neuron_j->net > max_net)
         max_net = neuron_j->net;

      // new out extreme?
      if (neuron_j->out < min_out)
         min_out = neuron_j->out;
      if (neuron_j->out > max_out)
         max_out = neuron_j->out;

      // check wether there is new weight value extreme?
      for (unsigned int k = 0; k < neuron_j->inputs.size(); k++)
      {
         float w = neuron_j->inputs[k]->weight;

         if (w < min_weight)
            min_weight = w;
         if (w > max_weight)
            max_weight = w;

      } // for (all input connections of current neuron)
      
   } // for (all neurons in this layer)

   printf("layer #%d: min_weight=%.2f, max_weight=%.2f, min_net=%.2f, max_net=%.2f, min_out=%.2f, max_out=%.2f\n",
           layer_id,
           min_weight, max_weight,
           min_net, max_net,
           min_out, max_out
         );

} // show_min_max_values



///
/// this function runs through all neurons in this layer
/// and for each neuron all incoming weight values will be printed
///
void neuron_layer::show_incoming_weights()
{
   printf("\nInput weights for all neurons in layer %d\n", layer_id);
   for (int neuron_nr = 0; neuron_nr < nr_neurons; neuron_nr++)
   {
      neuron* n = all_neurons_in_this_layer[neuron_nr];

      printf("neuron %d: ", neuron_nr);

      for (unsigned int input_nr = 0; input_nr < n->inputs.size(); input_nr++)
      {
         neuron_connection* c = n->inputs[input_nr];

         printf("%5.2f ", c->weight);

      } // for (all input connections of the current neuron n)

      printf("\n");

   } // for (all neurons in this layer)

} // show_incoming_weights