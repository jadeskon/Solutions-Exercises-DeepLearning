/// neuron_connection.h
/// 
/// Implements a connection between two neurons.
/// We store the information of the input neuron
/// and the weight value.
/// The connection object is stored by the post-synaptic
/// neuron itself.
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org


#pragma once

#include "neuron.h"

class neuron;  // forward declaration need, since neuron.h also includes neuron_connection.h

class neuron_connection
{
   public:

                  neuron_connection();       // default constructor
      
      neuron*     input_neuron;
      float       weight;

};
