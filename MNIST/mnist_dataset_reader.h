/// MNIST dataset reader
///
/// reads in the 60.000 training and 10.000 testing images of
/// the "MNIST database of handwritten digits" (0,1,2,...,9)
/// see http://yann.lecun.com/exdb/mnist/
///
/// note: download yourself the dataset and extract the files to a folder.
///       after extracting, you should get the MNIST dataset files:
///         t10k-images.idx3-ubyte
///         t10k-labels.idx1-ubyte
///         train-images.idx3-ubyte
///         train-labels.idx1-ubyte
///
/// by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org
///
/// parts of the code are inspired by
/// http://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

#pragma once

// Microsoft compiler specific disabling of warnings
// to say sprintf() usage is ok.
// Alternative: use sprintf_s(), the secure version
// of sprint()
#define _CRT_SECURE_NO_WARNINGS

#include <string>
#include <fstream>
#include <iostream>

#include "opencv2/core.hpp"    // for cv::Mat
#include "opencv2/highgui.hpp" // for CV_RGB
#include "opencv2/imgproc.hpp" // for cv::putText

using namespace std;
using namespace cv;

class mnist_dataset_reader
{
  public:

                            mnist_dataset_reader(string path_to_extracted_mnist_files); // reads in all training / test images + labels

    unsigned char**         get_train_images();

    unsigned char**         get_test_images();

    unsigned char*          get_train_labels();

    unsigned char*          get_test_labels();

    void                    get_mnist_image_as_cvmat(cv::Mat* visu, unsigned char** images, int image_idx);

    Mat*                    get_board_of_sample_images(unsigned char** images, unsigned char* labels, int nr_of_images);

    int                     nr_train_images_read;          // should be 60.000

    int                     nr_test_images_read;           // should be 10.000



  private:

    unsigned char**         read_mnist_images(string full_path, int& number_of_images, int& image_size); // for reading in images

    unsigned char*          read_mnist_labels(string full_path, int& number_of_labels);                  // for reading in ground truth image labels


    string                  path_to_extracted_mnist_files; // where you have extracted the files 

    unsigned char**         train_images;                  // all the 60.000 training images of size 28x28 

    unsigned char**         test_images;                   // all the 10.000 test     images of size 28x28

    unsigned  char*         train_labels;                  // all the 60.000 ground truth labels for the training images

    unsigned  char*         test_labels;                   // all the 10.000 ground truth labels for the test     images   
    
};
