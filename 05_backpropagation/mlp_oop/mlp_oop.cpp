#include "mlp_oop.h"
#include <conio.h>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>

#include "neuron_layer.h"

#define FOLDER_TO_STORE_MLP_VISUALIZATION_IMAGES "V:\\01_job\\00_vorlesungen_meine\\17_deep_learning_ws1617\\21_visualizations_of_mlps"


mlp_oop::mlp_oop()
{
   nr_layers = 0;

   // set default learn rate: 1.0/1000.0
   // can be set also with set_learn_rate()
   learn_rate = 0.001f; 

   // for debugging we can store images of the MLP
   // including weight information
   img_save_counter = 0;
}



///
/// creates a new neuron layer with the desired
/// number of neurons
///    and
/// connects each neuron in this new layer
/// to each neuron in the previous layer
///
void mlp_oop::add_neuron_layer(int id, int nr_neurons, bool create_bias_neuron, transferfunc_type tf_type)
{
   // 1. create the new layer
   neuron_layer* new_layer = new neuron_layer( id, nr_neurons, create_bias_neuron, tf_type );


   // 2. add pointer to new layer to array of layers
   all_layers.push_back( new_layer );
   nr_layers++;


   // 3. connect each neuron in the new layer
   //    to all neurons from the previous layer
   //    - if this is not the first layer ...
   if (all_layers.size() != 1)
   {
      // get pointer to previous layer
      neuron_layer* prev_layer = all_layers[ all_layers.size() - 2 ];

      // for all neurons in the current new layer ...
      // (but not the "always-on" neuron
      int N;
      if (create_bias_neuron)
         N = new_layer->nr_neurons - 1; // this layer has a bias neuron
      else
         N = new_layer->nr_neurons;     // this layer has no bias neuron
      for (int i = 0; i < N; i++)
      {
         // get pointer to newly created neuron
         neuron* n = new_layer->all_neurons_in_this_layer[i];

         // for all neurons in previous layer ...
         // (including a possible bias "always on" neuron)
         for (int j = 0; j < prev_layer->nr_neurons; j++)
         {
            // get next neuron from previous layer
            neuron* m = prev_layer->all_neurons_in_this_layer[j];

            // create new connection object
            neuron_connection* c = new neuron_connection();

            // setup connection object information
            c->input_neuron = m;

            // now store connection information in neuron m
            n->inputs.push_back( c );

         } // for (all neurons in previous layer)

      } // for (all neurons new layer)

   } // if (this is not the first layer being added)

   cout << endl << "generated neuron layer with index #" << nr_layers-1 << " with " << nr_neurons << " neurons.";

} // add_neuron_layer



///
/// show MLP structure information
///
void mlp_oop::show_mlp_structure_information()
{
   printf("\n\n");
   printf("MLP structure information:\n");
   printf("--------------------------\n");
   printf("Number of layers : %d\n", nr_layers);
   for (unsigned int layer_idx = 0; layer_idx < all_layers.size(); layer_idx++)
   {
      printf("layer #%d: %d neurons\n", layer_idx, all_layers[layer_idx]->nr_neurons);
   }

} // show_mlp_structure_information



///
/// computes neuron activities in ascending order
/// of layers
///
void mlp_oop::forward_pass()
{
   // for all layers but the first ...
   for (unsigned int layer_idx = 1; layer_idx < all_layers.size(); layer_idx++)
   {
      // let all neurons in that layer compute their activity
      all_layers[layer_idx]->compute_outputs();

   } // for (all layers but the first)

} // forward_pass



float mlp_oop::get_sum_of_output_values()
{
   neuron_layer* last_layer = all_layers[all_layers.size() - 1];
   
   float sum_out = 0.0f;   
   for (int j=0; j<last_layer->nr_neurons; j++)
      sum_out += last_layer->all_neurons_in_this_layer[j]->out;

   return sum_out;
}



void mlp_oop::set_learn_rate(float new_learn_rate)
{
   this->learn_rate = new_learn_rate;
}



///
/// implements the "Backpropagation of Error" algorithm
/// for Multi Layer Perceptrons
///
/// see, e.g., https://de.wikipedia.org/wiki/Backpropagation
/// for the weight update formulas for output and hidden neurons
///
void mlp_oop::backpropagation_compute_error_signals(float* teacher_vector)
{
   // starting from the last layer
   // we will first compute error signals layer by layer
   // by "propagating" the neuron error signals back to the first layer
   // then we will use the pre-computed weight changes to really adapt the weights

   int last_layer_idx = all_layers.size()-1;

   // 1. pre-compute weight changes, but do not yet adapt the weights
   for (unsigned int layer_idx = last_layer_idx; layer_idx >= 1; layer_idx--)
   {
      // 1.1 is there a next layer?
      // if yes, prepare a pointer to the next layer,
      // since for computing the error signals for hidden neurons,
      // we need to access the error signals of the neurons in the next layer
      int next_layer_idx = layer_idx+1;
      neuron_layer* ptr_next_layer = NULL;
      if (layer_idx != last_layer_idx)
         ptr_next_layer = all_layers[next_layer_idx];

      // 1.2 let all neurons in the current layer compute their error signal and weight change
      all_layers[layer_idx]->compute_error_signals_and_weight_change( teacher_vector, ptr_next_layer, learn_rate);

   } // for (all layers but the first)

} // backpropagation_compute_error_signals



///
/// use the pre-computed weight changes from
/// backpropagation_compute_error_signals()
/// to really adapt the weights now
///
void mlp_oop::backpropagation_change_weights()
{
   
   for (int layer_idx = 1; layer_idx < nr_layers; layer_idx++)
   {
      all_layers[layer_idx]->change_weights();
   }

} // backpropagation_change_weights



void mlp_oop::show_output_values_of_neurons_from_layer(int l)
{
   neuron_layer* desired_layer = all_layers[l];

   printf("\nOutput values of neurons from layer #%d : ", l);
   for (int j = 0; j<desired_layer->nr_neurons; j++)
      printf("\t#%d: %4.2f ", j, desired_layer->all_neurons_in_this_layer[j]->out);
}


///
/// for debugging whether weights or neuron output values degenerate
///
void mlp_oop::show_min_max_values_per_layer()
{
   for (int layer_idx = 0; layer_idx < nr_layers; layer_idx++)
   {
      all_layers[layer_idx]->show_min_max_values();
   }
   printf("\n");

} // show_min_max_values_per_layer


///
/// I have implemented a MLP with Backpropagation! Yes!!!
/// But does it work correctly?
/// Here is a simple unit test to check whether the MLP really
/// learns something.
/// 
/// We will generate a N->N MLP that shall learn the identity,
/// i.e., map each N-dimensional input vector x to the same output vector x
/// 
/// Two variants can be tested: using unit vectors or non-unit vectors
///

void mlp_oop::unit_test_identity()
{
   #define TEST_WITH_UNIT_VECTORS 0
   
   printf("\n\n");
   printf("Unit Test 'Identity'\n");

   // 1. generate a N-N MLP
   const int N = 3; // nr of neurons in input and output layer each
   set_learn_rate(0.1f);
   add_neuron_layer(0, N, true,  transferfunc_type::tf_type_identity);
   add_neuron_layer(1, N, false, transferfunc_type::tf_type_identity);
   show_mlp_structure_information();
   this->save_mlp_visualization_as_image("initial graph");

   // 2. train the network
   const int NR_TRAIN_STEPS = 100000;
   float x[N];
   printf("Training with %d training pairs...\n", NR_TRAIN_STEPS);
   for (int trainstep = 0; trainstep < NR_TRAIN_STEPS; trainstep++)
   {
      // 2.1 prepare training pair (x,x)
      if (TEST_WITH_UNIT_VECTORS)
      { 
         int rnd_idx = rand() % N; // will be in {0,...,N-1}
         for (int i = 0; i < N; i++)
         {
            if (i == rnd_idx)
               x[i] = 1.0f;
            else
               x[i] = 0.0f;
         }
      }
      else
      {
         for (int i = 0; i < N; i++)
         {
            x[i] = -1.0f + (rand() % 2001)/1000.0f;
         }
      }

      // 2.2 feed the MLP with the input vector
      for (int i = 0; i < N; i++)
         this->all_layers[0]->all_neurons_in_this_layer[i]->out = x[i];

      // 2.3 forward pass
      this->forward_pass();
      //this->save_mlp_visualization_as_image("feedforward step");

      // 2.4 backprop pass
      this->backpropagation_compute_error_signals(x);
      this->backpropagation_change_weights();
      //this->save_mlp_visualization_as_image("backprop step");

      if ((0) && (trainstep % (NR_TRAIN_STEPS/10) == 0))
      {
         printf("Trained %d of %d steps\n", trainstep, NR_TRAIN_STEPS);
         this->show_min_max_values_per_layer();
         _getch();
      }

   } // for (all training vectors)
   printf("Training finished.\n");


   // 3. show learned values of all incoming weights of output neurons
   this->all_layers[1]->show_incoming_weights();

   this->save_mlp_visualization_as_image("final weights");

   // 4. test the network
   for (int test = 0; test < 10; test++)
   {
      // 4.1 prepare test input vector x
      //     for the i-th test input vector x, we set only the i-th argument
      //     of vector x to 1.0 and all other arguments to 0.0
      if (TEST_WITH_UNIT_VECTORS)
      {
         int rnd_idx = rand() % N; // will be in {0,...,N-1}
         for (int i = 0; i < N; i++)
         {
            if (i == rnd_idx)
               x[i] = 1.0f;
            else
               x[i] = 0.0f;
         }
      }
      else
      {
         for (int i = 0; i < N; i++)
         {
            x[i] = -1.0f + (rand() % 2001) / 1000.0f;
         }
      }

      // 4.2 feed the MLP with the input vector
      for (int i = 0; i < N; i++)
         this->all_layers[0]->all_neurons_in_this_layer[i]->out = x[i];

      // 4.3 forward pass
      this->forward_pass();
      
      // 4.4 show current values of input and output neurons
      //     in order to see whether the MLP really learned
      //     to map input vectors x
      //     to the same output vectors x
      show_output_values_of_neurons_from_layer(0);
      show_output_values_of_neurons_from_layer(1);

      printf("\nPress a key to test the MLP with another random input vector x\n");
      _getch();
      
   } // for (all tests)

   printf("\n\nUnit Test 'Learn identity' finished.\n");
   printf("Press a key.\n");
   _getch();

} // unit_test_identity



///
/// I have implemented a MLP with Backpropagation! Yes!!!
/// But does it work correctly?
/// Here is a simple unit test to check whether the MLP really
/// learns something.
///
/// There are two inputs x0,x1.
/// There are three outputs y0,y1,y2,y3,y4
/// The net shall learn the functions y0=x0+x1, y1=x0-x1, y2=3*x0, y3=sin(x0), y4=x0*x1
///
/// Try out different hyperparameters!
///  - number of learn steps 10000, 1000000
///  - learn rates 0.001f, 0.00001f
///  - different transfer function types
///    (especially identity vs. non-linear variants as tanh and relu)
///  - number of neurons per layer (2, 5, 20)
///  - different topologies of the net (3 layers, 4 layers, 5 layers, ...)

void mlp_oop::unit_test_regression()
{
   const int NR_TRAIN_STEPS = 1000000;

   printf("\n\n");
   printf("Unit Test 'Regression'\n");

   // 1. generate a MLP
   set_learn_rate(0.001f);
   add_neuron_layer(0,   2,  true,  transferfunc_type::tf_type_identity);
   add_neuron_layer(1,   5,  true,  transferfunc_type::tf_type_relu);
   add_neuron_layer(4,   5, false,  transferfunc_type::tf_type_identity);
   show_mlp_structure_information();
   //this->save_mlp_visualization_as_image("initial graph");


   // 2. generate training & test data
   const int NR_TRAIN_PATTERNS = 1000;
   vector<vector<float>> train_data;
   for (int i = 0; i < NR_TRAIN_PATTERNS; i++)
   {
      vector<float> inp_out_vec;
      float x0 = -5.0f + ((float)(rand() % 1001) / 1000.0f)*10.0f;
      float x1 = -5.0f + ((float)(rand() % 1001) / 1000.0f)*10.0f;
      inp_out_vec.push_back( x0      );
      inp_out_vec.push_back( x1      );
      inp_out_vec.push_back( x0 + x1 );
      inp_out_vec.push_back( x0 - x1 );
      inp_out_vec.push_back( 3 * x0  );
      inp_out_vec.push_back( sin(x0) );
      inp_out_vec.push_back( x0 * x1 );

      train_data.push_back( inp_out_vec );
   }
   const int NR_TEST_PATTERNS = 1000;
   vector<vector<float>> test_data;
   for (int i = 0; i < NR_TEST_PATTERNS; i++)
   {
      vector<float> inp_out_vec;
      float x0 = -5.0f + ((float)(rand() % 1001) / 1000.0f)*10.0f;
      float x1 = -5.0f + ((float)(rand() % 1001) / 1000.0f)*10.0f;
      inp_out_vec.push_back( x0      );
      inp_out_vec.push_back( x1      );
      inp_out_vec.push_back( x0 + x1 );
      inp_out_vec.push_back( x0 - x1 );
      inp_out_vec.push_back( 3 * x0  );
      inp_out_vec.push_back( sin(x0) );
      inp_out_vec.push_back( x0 * x1 );

      test_data.push_back(inp_out_vec);
   }


   // 3. train the network
         
   printf("Training %d train steps...\n", NR_TRAIN_STEPS);
   for (int trainstep = 0; trainstep < NR_TRAIN_STEPS; trainstep++)
   {
      // 3.1 randomly generate a training pattern index
      int rnd_idx = rand() % NR_TRAIN_PATTERNS;

      // 3.2 get that train / test pattern
      vector<float> pattern = train_data[ rnd_idx ];

      // 3.3 feed the MLP with the input vector
      this->all_layers[0]->all_neurons_in_this_layer[0]->out = pattern[0];
      this->all_layers[0]->all_neurons_in_this_layer[1]->out = pattern[1];

      // 3.3 forward pass
      this->forward_pass();
      //this->save_mlp_visualization_as_image("feedforward step");

      // 3.4 backprop pass
      float teacher_vec[5];
      teacher_vec[0] = pattern[2];
      teacher_vec[1] = pattern[3];
      teacher_vec[2] = pattern[4];
      teacher_vec[3] = pattern[5];
      teacher_vec[4] = pattern[6];
      this->backpropagation_compute_error_signals( teacher_vec );
      this->backpropagation_change_weights();
      //this->save_mlp_visualization_as_image("backprop step");

      if ((1) && (trainstep % (NR_TRAIN_STEPS / 10) == 0))
      {
         printf("Trained %d of %d steps\n", trainstep, NR_TRAIN_STEPS);
         this->show_min_max_values_per_layer();
         //this->save_mlp_visualization_as_image("after backprop step");
         //_getch();
      }

   } // for (all training vectors)
   printf("Training finished.\n");


   // 4. show learned values of all incoming weights of output neurons
   this->all_layers[1]->show_incoming_weights();
   this->all_layers[2]->show_incoming_weights();
   //this->save_mlp_visualization_as_image("final weights");

   // 5. test the network
   for (int test = 0; test < 15; test++)
   {
      // 5.1 randomly choose a test pattern index
      int rnd_idx = rand() % NR_TEST_PATTERNS;

      // 5.2 get that test pattern
      vector<float> pattern = test_data[rnd_idx];

      // 5.3 feed the MLP with the input vector
      this->all_layers[0]->all_neurons_in_this_layer[0]->out = pattern[0];
      this->all_layers[0]->all_neurons_in_this_layer[1]->out = pattern[1];

      // 5.3 forward pass
      this->forward_pass();

      // 5.4 show current values of input and output neurons
      //     in order to see whether the MLP really learned the XOR function
      //show_output_values_of_neurons_from_layer(0);
      //show_output_values_of_neurons_from_layer(1);
      show_output_values_of_neurons_from_layer(2);
      //show_output_values_of_neurons_from_layer(3);
      //show_output_values_of_neurons_from_layer(4);

      // 5.5 show desired output values as a comparison to the actual outputs
      float teacher_vec[5];
      teacher_vec[0] = pattern[2];
      teacher_vec[1] = pattern[3];
      teacher_vec[2] = pattern[4];
      teacher_vec[3] = pattern[5];
      teacher_vec[4] = pattern[6];
      printf("\nTeacher values are                     : "
      "\t#0: %.2f \t#1: %.2f \t#2: %.2f \t#3: %.2f \t#4: %.2f\n",
         teacher_vec[0],
         teacher_vec[1],
         teacher_vec[2],
         teacher_vec[3],
         teacher_vec[4] );

      printf("\nPress a key to test the MLP with another random input vector x\n");
      _getch();

   } // for (all tests)

   printf("\n\nUnit Test 'Regression' finished.\n");
   printf("Press a key.\n");
   _getch();

} // unit_test_regression


///
/// for the current state of the MLP we will
/// generate a .dot Graphviz visualization file
/// (textual description of a graph)
/// and then call the dot.exe tool that can generate
/// automatically a .png image from that description
///
/// This allows to visualize the graph structure and
/// current weights for small graphs.
///
void mlp_oop::save_mlp_visualization_as_image(string infotxt)
{
   char next_line[500];

   string folder = FOLDER_TO_STORE_MLP_VISUALIZATION_IMAGES;

   // 1. generate .dot file (textual description of graph)

   // 1.1 prepare .dot filename
   char buff[500];
   sprintf_s(buff, 500, "%s\\tmp.dot", folder.c_str());
   string dot_filename = buff;

   // 1.2 open .dot file for writing
   ofstream f( dot_filename );

   // 1.3 write graph

   // 1.3.1 write graph header and desired resolution cmd
   f << "digraph MLP {" << endl;
   f << "   graph[dpi = 300];" << endl;
        
   // 1.3.2 define graph title
   f << "   labelloc = \"t\"" << endl; // means that we want a centered title on the top of the graph
   f << "   label = \"" << infotxt.c_str() << "(" << img_save_counter << ")\"" << endl;

   // 1.3.3 write node (neuron) information         

   // for all neuron layers (including the first)
   for (unsigned int layer_nr = 0; layer_nr < all_layers.size(); layer_nr++)
   {
      // get the next layer
      neuron_layer* next_layer = all_layers[layer_nr];

      // for all neurons j in that layer
      for (int j = 0; j < next_layer->nr_neurons; j++)
      {
         // get pointer to that neuron
         neuron* neuron_j = next_layer->all_neurons_in_this_layer[j];

         // define display text for a node (neuron)
         sprintf_s(next_line, 500, "   n%d_%d [label=\" #%d #%d %.2f / %.2f delta=%.2f \"]",
            layer_nr, j,
            layer_nr, j, neuron_j->net, neuron_j->out, neuron_j->error_signal);

         f << next_line << endl;
      }
   }

   // 1.3.4 write edge information
   
   // for all neuron layers (except the first)   
   for (unsigned int layer_nr = 1; layer_nr < all_layers.size(); layer_nr++)
   {
      // get the next layer
      neuron_layer* next_layer = all_layers[layer_nr];

      // for all neurons j in that layer
      for (int j = 0; j < next_layer->nr_neurons; j++)
      {
         // get pointer to that neuron
         neuron* neuron_j = next_layer->all_neurons_in_this_layer[j];

         // for all incoming connections of that neuron
         for (unsigned int k = 0; k < neuron_j->inputs.size(); k++)
         {
            // get next incoming connection
            neuron_connection* c = neuron_j->inputs[k];

            // Graphviz uses syntax: Nodename1->Nodename2 [label = "some edge text"]            
            sprintf_s(next_line, 500, "   n%d_%d -> n%d_%d [label=""%.4f""]",
             layer_nr-1, k, 
             layer_nr,   j,
             c->weight );
            f << next_line << endl;
         }

      } // for (all neurons)

   } // for (all layers)

   // 1.3.4 write graph end bracket
   f << "}" << endl;

   // 1.4 close .dot file
   f.close();


   // 2. .dot file --> .png image file   

   // 2.1 prepare .png filename
   sprintf_s(buff, 500, "%s\\mlp_%04d.png", folder.c_str(), img_save_counter);
   string png_filename = buff;

   // 2.2 tell user that we are going to generate png image
   printf("Saving visualization of MLP to %s\n", png_filename.c_str());

   // 2.3 execute dot.exe tool from command shell
   char cmd[500];
   sprintf_s(cmd, 500, "dot.exe -Tpng %s -o %s", dot_filename.c_str(), png_filename.c_str() );
   system( cmd );


   // 3. one image saved more ...
   img_save_counter++;

} // save_mlp_visualization_as_image
