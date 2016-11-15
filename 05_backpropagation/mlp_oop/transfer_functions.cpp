#pragma once

#include "transfer_functions.h"

float identity(float x)
{
   return x;
}

float identity_derivative(float x)
{
   return 1.0f;
}


///
/// _standard_ logistic function
///
/// see https://en.wikipedia.org/wiki/Logistic_function#Mathematical_properties
///
float logistic_func(float x)
{
   float y = 1.0f / (1.0f + (float)exp(-x));
   return y;
}



///
/// derivative of standard logistic function f(x) is f'(x) = f(x)*(1-f(x))
///
/// see https://en.wikipedia.org/wiki/Logistic_function#Derivative 
///
float logistic_func_derived(float x)
{
   float y = logistic_func(x) * (1 - logistic_func(x));
   return y;
}


float tangens_hyperbolicus(float x)
{
   float y = (float)tanh( x );
   return y;
}



float tangens_hyperbolicus_derived(float x)
{
   float r = tangens_hyperbolicus(x);
   float y = 1.0f - r*r;
   return y;
}



float relu(float x)
{
   if (x<0)
      return 0.0f;
   else
      return x;
}


float relu_derived(float x)
{
   if (x<0)
      return 0.0f;
   else
      return 1.0f;
}