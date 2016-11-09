#include "mlp_oop.h"

#include "neuron_layer.h"


mlp_oop::mlp_oop()
{
   nr_layers = 0;
}

///
/// creates a new neuron layer with the desired
/// number of neurons
///    and
/// connects each neuron in this new layer
/// to each neuron in the previous layer
///
void mlp_oop::add_neuron_layer(int nr_neurons)
{
   // 1. create the new layer
   neuron_layer* new_layer = new neuron_layer(nr_neurons);

   // 2. add pointer to new layer to array of layers
   all_layers.push_back( new_layer );
   nr_layers++;

   // 3. connect each neuron in the new layer
   //    to all neurons from the previous layer
   //    - if this is not the first layer ...
   if (all_layers.size() != 1)
   {
      // get pointer to previous layer
      neuron_layer* prev_layer = all_layers[ all_layers.size() - 2 ];

      // for all neurons in the current new layer ...
      // but the last neuron which is the "always on" neuron!
      for (int i = 0; i < new_layer->nr_neurons-1; i++)
      {
         // get pointer to newly created neuron
         neuron* n = new_layer->all_neurons_in_this_layer[i];

         // for all neurons in previous layer ...
         // (including the bias "always on" neuron)
         for (int j = 0; j < prev_layer->nr_neurons; j++)
         {
            // get next neuron from previous layer
            neuron* m = prev_layer->all_neurons_in_this_layer[j];

            // create new connection object
            neuron_connection* c = new neuron_connection();

            // setup connection object information
            c->input_neuron = m;

            // now store connection information in neuron m
            n->inputs.push_back( c );

         } // for (all neurons in previous layer)

      } // for (all neurons new layer)

   } // if (this is not the first layer being added)

   cout << endl << "generated neuron layer with index #" << nr_layers-1 << " with " << nr_neurons << " neurons.";

} // add_neuron_layer


///
/// computes neuron activities in ascending order
/// of layers
///
void mlp_oop::forward_pass()
{
   // for all layers but the first ...
   for (unsigned int layer_nr = 1; layer_nr < all_layers.size(); layer_nr++)
   {
      // let all neurons in that layer compute their activity
      all_layers[layer_nr]->compute_outputs();   

   } // for (all layers but the first)

} // forward_pass



float mlp_oop::get_sum_of_output_values()
{
   neuron_layer* last_layer = all_layers[all_layers.size() - 1];
   
   float sum_out = 0.0f;   
   for (int j=0; j<last_layer->nr_neurons; j++)
      sum_out += last_layer->all_neurons_in_this_layer[j]->output;

   return sum_out;
}
