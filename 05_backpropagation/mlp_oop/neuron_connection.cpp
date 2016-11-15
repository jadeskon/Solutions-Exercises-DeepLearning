#include "neuron_connection.h"

neuron_connection::neuron_connection()
{
   // random initialization of weight drawn from a
   // uniform distribution over [-1,1]
   weight = -1 + ((float)rand() / (float) RAND_MAX)*2.0f;
}