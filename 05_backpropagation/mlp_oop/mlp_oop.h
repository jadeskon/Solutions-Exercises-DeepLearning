/// mlp_oop.h
/// 
/// Implementation of a Multi Layer Perceptron (MLP),
/// i.a. a feedword network with neurons organized in layers
/// and each two neighbored layers are fully connected into
/// upward direction.
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#pragma once


#include <vector>

#include "neuron_layer.h"

using namespace std;


class mlp_oop
{

   public:

                                  mlp_oop();                                                          // constructs a new empty MLP

      void                        add_neuron_layer(int id,
                                                   int nr_neurons,
                                                   bool create_bias_neuron,
                                                   transferfunc_type tf_type);                        // adds a new neuron layer and connects all previous layer neurons to each neuron in this new layer

      void                        forward_pass();                                                     // compute neuron activities in ascending layer order

      void                        set_learn_rate(float new_learn_rate);                               // allows to set how "fast" the MLP adjusts its weights in the Backprop step

      void                        backpropagation_compute_error_signals(float* teacher_vector);
      
      void                        backpropagation_change_weights();



      void                        save_mlp_visualization_as_image(string infotxt);                    // allows to use automatically save Graphviz visualizations of the current state of the MLP

      void                        show_mlp_structure_information();                                   // outputs some information about the MLP (nr of layers, nr of neurons per layer)

      void                        show_output_values_of_neurons_from_layer(int l);

      void                        show_min_max_values_per_layer();                                    // for debugging whether weights or neuron output values degenerate

      void                        unit_test_identity();                                               
      void                        unit_test_regression();
      
      float                       get_sum_of_output_values();


      vector<neuron_layer*>       all_layers;                                                         // array of pointers to all neuron layers

      int                         nr_layers;                                                          // how many neuron layers are there currently in this MLP?

      float                       learn_rate;                                                         // how fast will the MLP adjust its weight in the Backprop step?

      int                         img_save_counter;                                                   // which is the next image number to save?

}; // class mlp_oop
