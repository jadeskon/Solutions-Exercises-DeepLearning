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

                        neuron_layer(int id,
                                     int nr_neurons,
                                     bool create_bias_neuron,
                                     transferfunc_type transfer_func_to_use);                           // constructs a new layer with the specified number of neurons

      void              compute_outputs();                                                              // all neurons in this layer will compute their outputs

      void              compute_error_signals_and_weight_change(float* teacher_vector,
                                                                neuron_layer* next_layer,
                                                                float learn_rate);                      // let all neurons in this layer compute their error signals

      void              change_weights();                                                               // uses the pre-comuted weight changes to really adapt the weights


      void              show_debug_infos();                                                             // shows some helpful debug infos about this neuron layer

      void              show_min_max_values();                                                          // will print min/max values of all incoming weights and neuron output values

      void              show_incoming_weights();                                                        // will output current values of incoming neuron weights (of all neurons in that layer!)



      vector<neuron*>   all_neurons_in_this_layer;                                                      // array of pointers to all neurons that live in this layer

      int               nr_neurons;                                                                     // how many neurons does this layer have?

      int               layer_id;                                                                       // each layer will get an id, this will help for debugging

      bool              has_bias_neuron;                                                                // is the last neuron in this layer a bias / always-on neuron?
};
