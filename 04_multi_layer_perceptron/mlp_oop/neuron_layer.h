// neuron_layer.h
// 
// Groups a set of neurons together to form one layer
//
// In order to avoid the overhead of calling functions
// due to speed reasons, all neuron layer properties are made public.
//
// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#pragma once

#include <iostream>

#include "neuron.h"

using namespace std;


class neuron_layer
{
   public:

                        neuron_layer(int nr_neurons);                                                   // constructs a new layer with the specified number of neurons

      void              compute_outputs();                                                              // all neurons in this layer will compute their outputs

      void              show_debug_infos();                                                             // shows some helpful debug infos about this neuron layer



      vector<neuron*>   all_neurons_in_this_layer;                                                      // array of pointers to all neurons that live in this layer

      int               nr_neurons;                                                                     // how many neurons does this layer have?
};
