/// mlp_fast.h
/// 
/// Implementation of a Multi Layer Perceptron (MLP),
/// i.a. a feedword network with neurons organized in layers
/// and each two neighbored layers are fully connected into
/// upward direction.
///
/// Here we try to avoid OOP overhead in order to come to
/// a faster MLP implementation.
///
/// Compare this implementation with mlp_oop (strongly OOP
/// approach used there).
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#include <stdio.h>  // for printf()
#include <stdarg.h> // for working with variadic functions (usage of ellipse ...)
#include <random>

using namespace std;


class mlp_fast
{
   public:
                                  mlp_fast(int nr_layers, ...);                                       // constructs a new empty MLP

      void                        forward_pass();                                                     // compute neuron activities in ascending layer order

      float                       get_sum_of_output_values();

      int                         nr_layers;                                                          // how many neuron layers are there currently in this MLP?
      
      int*                        nr_neurons_per_layer;                                               // array that stores nr of neurons present in each layer

      float**                     net;                                                                // netto input values of neurons

      float**                     out;                                                                // output values of neurons

      float***                    weights;                                                            // weights from one layer to next

}; // mlp_fast
