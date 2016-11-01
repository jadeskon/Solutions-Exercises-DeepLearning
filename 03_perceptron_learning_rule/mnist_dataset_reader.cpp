#include "mnist_dataset_reader.h"
#include <conio.h> // for _getch()


///
/// read in all the training / test images and their corresponding ground truth labels
///
mnist_dataset_reader::mnist_dataset_reader(std::string path_to_extracted_mnist_files)
{
  // 1. store path to folder where dataset was extracted by user
  this->path_to_extracted_mnist_files = path_to_extracted_mnist_files;


  // 2. read in training data
  int size_of_single_image_read;
  int nr_labels_read;

  // 2.1 read in training images
  cout << endl;  
  train_images = read_mnist_images( path_to_extracted_mnist_files + "\\" + "train-images.idx3-ubyte", nr_train_images_read, size_of_single_image_read);
  cout << "I have read " << nr_train_images_read << " training images of size " << size_of_single_image_read << "!" << endl;

  // 2.2 read in training ground truth labels
  cout << endl;
  train_labels = read_mnist_labels( path_to_extracted_mnist_files + "\\\\" + "train-labels.idx1-ubyte", nr_labels_read);
  cout << "I have read " << nr_labels_read << " labels for the training images!" << endl;


  // 3. read in test data

  // 3.1 read in test images
  cout << endl;
  test_images = read_mnist_images(path_to_extracted_mnist_files + "\\\\" + "t10k-images.idx3-ubyte", nr_test_images_read, size_of_single_image_read);
  cout << "I have read " << nr_test_images_read << " test images of size " << size_of_single_image_read << "!" << endl;

  // 3.2 read in testing ground truth labels
  cout << endl;
  test_labels = read_mnist_labels(path_to_extracted_mnist_files + "\\\\" + "t10k-labels.idx1-ubyte", nr_labels_read);
  cout << "I have read " << nr_labels_read << " labels for the testing images!" << endl;


}



///
/// read in a set of images and return a pointer to the 2D array (pointer char** to 1D array of row pointers char*)
///
unsigned char** mnist_dataset_reader::read_mnist_images(string full_path, int& number_of_images, int& image_size)
{

  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  cout << "reading MNIST file " << full_path << " ... " << endl;

  ifstream file(full_path, ios::binary);

  if (file.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

    file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

    cout << "nr of rows x nr of cols = " << n_rows << " x " << n_cols << endl;

    image_size = n_rows * n_cols;

    unsigned char** _dataset = new unsigned char*[number_of_images];
    for (int i = 0; i < number_of_images; i++) {
      _dataset[i] = new unsigned char[image_size];
      file.read((char*)_dataset[i], image_size);
    }
    cout << ".. all images read!" << endl;

    return _dataset;
  }
  else {
    cout << "Damn! I could not open the file " << full_path << "\n";
    cout << "Press a key to continue. Things will go wrong ...\n";
    _getch();
    return NULL;
  }

} // read_mnist_images


///
/// read label data from the specified MNIST label file
///
unsigned char* mnist_dataset_reader::read_mnist_labels(string full_path, int& number_of_labels)
{
  auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  };

  cout << "reading MNIST file " << full_path << " ... " << endl;

  ifstream file(full_path, ios::binary);

  if (file.is_open()) {
    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

    file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

    uchar* _dataset = new uchar[number_of_labels];
    for (int i = 0; i < number_of_labels; i++) {
      file.read((char*)&_dataset[i], 1);
    }
    cout << ".. all labels read!" << endl;
    return _dataset;
  }
  else {
    throw runtime_error("Unable to open file `" + full_path + "`!");
  }

} // read_mnist_labels



unsigned char** mnist_dataset_reader::get_train_images()
{
  return train_images;
}


unsigned char** mnist_dataset_reader::get_test_images()
{
  return test_images;
}


unsigned char* mnist_dataset_reader::get_train_labels()
{
  return train_labels;
}


unsigned char* mnist_dataset_reader::get_test_labels()
{
  return test_labels;
}


///
/// given a specified set of images (train or test images)
/// this method can be used to render a cv::Mat image of a MNIST digit sample
/// that can be displayed to the user
///
void mnist_dataset_reader::get_mnist_image_as_cvmat(cv::Mat* visu, unsigned char** images, int image_idx)
{ 
  int idx_1d = 0;
  for (int y = 0; y < 28; y++)
  {
    for (int x = 0; x < 28; x++)    
    {
      unsigned char gray_value = images[image_idx][idx_1d++];

      visu->at<uchar>(y,x) = gray_value;

      //printf("%03d ", gray_value);
    }
    //cout << endl;
  }
  
} // get_mnist_image_as_cvmat


///
/// select some sample images randomly and visualize them together with their
/// labels
///
Mat* mnist_dataset_reader::get_board_of_sample_images(unsigned char** images, unsigned char* labels, int nr_of_images)
{
  const int w=1200;
  const int h=800;
  bool still_space_for_drawing = true;
  Mat* visu = new Mat(h,w, CV_8UC3);

  int y = 0;
  int x = 0;
  const int some_space = 7;
  while (still_space_for_drawing)
  {
    // get random image index
    int rnd_idx = rand() % nr_of_images;

    // copy visualization of image of digit to board
    cv::Mat sample_image(28,28, CV_8UC1);
    get_mnist_image_as_cvmat(&sample_image, train_images, rnd_idx );
    Mat sample_image_3channels;
    cvtColor(sample_image, sample_image_3channels, CV_GRAY2RGB);
    Mat* roi = new Mat( *visu, Rect(x,y,28,28) );
    sample_image_3channels.copyTo( *roi );
    x += 28 + some_space;

    // draw image label to the right of image of digit on board as well
    char txt[10];
    sprintf_s(txt, "%d", labels[rnd_idx]);
    putText(*visu,
            txt,
            Point(x, y+20),
            FONT_HERSHEY_SIMPLEX, 0.7, // font face and scale
            CV_RGB(255, 255, 0), // yellow
            1); // line thickness and type
    x += 28 + some_space*2;

    // line break before next visualization of sample digit?
    if (x + 2 * 28 + some_space >= w)
    {
      // yes! line break
      x=0;
      y += 28+2*some_space;

      // is there still enough space for rendering another line?
      if (y+28+some_space>=h)
      {
        // no space for rendering left
        still_space_for_drawing=false;
      }

    } // if
                
  } // while (still_space_for_drawing)

  return visu;

} // get_board_of_sample_images