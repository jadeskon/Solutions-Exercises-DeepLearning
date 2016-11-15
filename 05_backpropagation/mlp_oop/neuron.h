/// neuron.h
/// 
/// Implements the generalized perceptron neuron model
///
/// In order to avoid the overhead of calling set()/get() functions
/// due to speed reasons, all neuron properties are made public.
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#pragma once

#include "neuron_connection.h"

#include "transfer_functions.h" // for transferfunc_type

#include <vector>

using namespace std;

class neuron_connection;

class neuron
{
   public:
                                        neuron(transferfunc_type transferfunc_to_use);

      void                              compute_output();

      float                             f(float x);                        // returns the transfer function value at x

      float                             f_derived(float x);                // returns the derivative of the transfer function value at x

      void                              change_weights();                  // adds pre-computed weight change value delta_w for each incoming connection
 
      vector<neuron_connection*>        inputs;
      float                             net;
      transferfunc_type                 transferfunc_to_use;
      float                             out;

      float                             error_signal;                      // the "error signal" we compute for each neuron
                                                                           // in the Backpropagation algorithm
      

}; // class neuron