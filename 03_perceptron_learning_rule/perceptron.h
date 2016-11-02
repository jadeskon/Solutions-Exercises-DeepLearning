#pragma once

/// perceptron.h
///
/// Header file for a simple Perceptron class
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#define USE_BINARY_NEURON_OUTPUTS 0

class Perceptron
{
   public:

                     Perceptron(int nr_inputs, int nr_outputs, float learning_rate);

      void           compute_outputs();

      void           show_debug_info();

      int            get_classification();

      void           learn(float* teacher_vector);


      int            nr_inputs;

      int            nr_outputs;

      float          learning_rate;

      float*         input_values;

      float*         output_values;

      float**        weights;

}; // class Perceptron
