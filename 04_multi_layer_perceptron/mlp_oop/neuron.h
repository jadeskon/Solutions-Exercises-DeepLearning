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

#include <vector>

using namespace std;

class neuron_connection;

class neuron
{
   public:
                                        neuron();

      void                              compute_output();
 
      float                             activation;
      float                             output;
      vector<neuron_connection*>        inputs;

}; // class neuron