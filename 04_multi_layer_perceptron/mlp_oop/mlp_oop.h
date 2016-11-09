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

      void                        add_neuron_layer(int nr_neurons);                                   // adds a new neuron layer and connects all previous layer neurons to each neuron in this new layer

      void                        forward_pass();                                                     // compute neuron activities in ascending layer order

      float                       get_sum_of_output_values();

      vector<neuron_layer*>       all_layers;                                                         // array of pointers to all neuron layers

      int                         nr_layers;                                                          // how many neuron layers are there currently in this MLP?

}; // class mlp_oop
