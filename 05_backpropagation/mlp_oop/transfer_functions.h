#pragma once

/// transfer_functions.h
/// 
/// Provides function definitions for
/// some important transfer functions used in artifical
/// neuron models and their corresponding derivative.
///
/// ---
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org

#include <math.h>

enum transferfunc_type { tf_type_identity, tf_type_logistic, tf_type_tanh, tf_type_relu };


float identity(float x);

float identity_derivative(float x);


float logistic_func(float x);

float logistic_func_derived(float x);


float tangens_hyperbolicus(float x);

float tangens_hyperbolicus_derived(float x);


float relu(float x);

float relu_derived(float x);
