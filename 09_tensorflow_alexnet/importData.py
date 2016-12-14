# file: importData.py
#
# Contains helper class Dataset
# which is able to return mini batches of a desired
# size of <label, image> pairs
#
# For this the class constructor needs to be passed
# a imagePath in which it expects a subfolder for each
# image category.
#
# It automatically will traverse each subfolder recursively
# in order to generate a image list of the form
# e.g. [['cow', 'cow8439.jpeg'], ['dog', 'dog02.jpeg'], ...]
#
# Images will be read only using OpenCV's imread() function
# when preparing a mini-batch.
# They are not loaded all at once!
#
# AlexNet was presented in a paper in 2012
# and was the winner of the ILSVRC 2012 competition.
# It was one of the reasons for the Deep Learning Tsunami.
#
# Details can be found in the original publication:
# 
# Krizhevsky, A.; Sutskever, I. & Hinton, G. E.:
#    ImageNet Classification with Deep Convolutional Neural Networks
#    Advances in Neural Information Processing Systems 25
#    Curran Associates, Inc., 2012, 1097-1105
#
# The implementation found here is strongly based on the
# AlexNet implementation by Wang Xinbo:
#    https://github.com/SidHard/tfAlexNet
# 
# ---
# Prof. Dr. Juergen Brauer, www.juergenbrauer.org


import numpy as np
import os
import cv2

class Dataset:

    def __init__(self, imagePath, extensions):

        # 1. prepare image list with category information
        #    self.data = [['cow', 'cowimg01.jpeg'], ['dog', 'dogimg3289.jpeg], ...]
        print("\n")
        print("Searching in folder", imagePath, "for images")
        self.data = createImageList(imagePath, extensions)
        NrImgs = len(self.data)
        print("Found", NrImgs, "images")
        print("Here are the first 5 and the last 5 images and their corresponding categories I found:")
        for i in range(0,5):
                print(self.data[i])
        for i in range(NrImgs-5,NrImgs):
                print(self.data[i])
        
        # 2. shuffle the data
        np.random.shuffle(self.data)
        self.num_records = len(self.data)
        self.next_record = 0

        # 3. use zip function to unzip the data into two lists
        #    see https://docs.python.org/3.3/library/functions.html#zip
        self.labels, self.inputs = zip(*self.data)

        # 4. show some random images
        for i in range(0,5):
                rnd_idx = np.random.randint(NrImgs)
                rnd_filename = self.inputs[ rnd_idx ]
                print("random filename = ", rnd_filename)
                img = cv2.imread( rnd_filename )   
                img = cv2.resize(img, (227, 227))     
                img_name = "example image " + str(i)
                cv2.imshow(img_name, img)
                cv2.moveWindow(img_name, 300+i*250,100);
        cv2.waitKey(5000)
        cv2.destroyAllWindows()


        # 5. remove duplicates from list
        category = np.unique(self.labels)

        # 6. how many categories are there?
        self.num_labels = len(category)
        
        # 7. prepare dictionary to map category names to category numbers
        self.category2label = dict(zip(category, range(len(category))))

        # 8. and the other way round:
        #    prepare a dictionary to map category numbers to category names
        self.label2category = {l: k for k, l in self.category2label.items()}

        # 9. prepare ground truth vector for each image where
        #    we can find the ground truth category number for each image i
        #    as the i-th argument
        self.labels = [self.category2label[l] for l in self.labels]


    def __len__(self):
        return self.num_records

    def onehot(self, label):
        # returns a onehot list where all entries are set to 0
        # but the right categroy
        v = np.zeros(self.num_labels)
        v[label] = 1
        return v

    def recordsRemaining(self):
        return len(self) - self.next_record

    def hasNextRecord(self):
        return self.next_record < self.num_records

    def preprocess(self, img):
        # preprocess image by resizing it to 227x227
        pp = cv2.resize(img, (227, 227))

        # and convert OpenCV representation to Numpy array
        # note: asarray does not copy data!
        #       see 
        pp = np.asarray(pp, dtype=np.float32)

        # map values from [0,255] to [0,1]
        pp /= 255

        # prepare array of shape width x height x 3 array
        pp = pp.reshape((pp.shape[0], pp.shape[1], 3))
        return pp

    def nextRecord(self):

        # will return the next training pair
        # consisting of the input image and a one-hot/teacher label vector
        if not self.hasNextRecord():

            # generate new random order of images

            # randomly shuffle the data again
            np.random.shuffle(self.data)
            self.next_record = 0
            self.labels, self.inputs = zip(*self.data)

            category = np.unique(self.labels)
            self.num_labels = len(category)
            self.category2label = dict(zip(category, range(len(category))))
            self.label2category = {l: k for k, l in self.category2label.items()}
    
            # prepare ground-truth label information for all images
            # according to the newly shuffled order of the images
            self.labels = [self.category2label[l] for l in self.labels]

        # prepare one-hot teacher vector for the output neurons
        label = self.onehot(self.labels[self.next_record])

        # read in the image and preprocess the data
        img_filename = self.inputs[self.next_record]
        #print("reading ", img_filename)
        input = self.preprocess(cv2.imread(img_filename))
        self.next_record += 1
        return label, input

    def nextBatch(self, batch_size):

        # creates a mini-batch of the desired size
        records = []
        for i in range(batch_size):
            record = self.nextRecord()
            if record is None:
                break
            records.append(record)
        labels, input = zip(*records)
        return labels, input


def createImageList(imagePath, extensions):

    # 1. start with an empty list of filenames and labels
    imageFilenames = []
    labels = []

    # 2. each subfolder name in imagePath is considered to be
    #    a class label:
    #    e.g. pics/n and pics/p --> categoryList = [n, p]
    categoryList = [None]
    categoryList = [c for c in sorted(os.listdir(imagePath))
                    if c[0] != '.' and
                    os.path.isdir(os.path.join(imagePath, c))]

    # 3. for each of the categories
    for category in categoryList:
        print("category =", category)
        if category:
            walkPath = os.path.join(imagePath, category)
        else:
            walkPath = imagePath
            category = os.path.split(imagePath)[1]

        # create a generator
        w = _walk(walkPath)

        # step through all directories and subdirectories
        while True:

            # get names of dirs and filenames of current dir
            try:
                dirpath, dirnames, filenames = next(w)
            except StopIteration:
                break

            # don't enter directories that begin with '.'
            for d in dirnames[:]:
                if d.startswith('.'):
                    dirnames.remove(d)

            dirnames.sort()

            # ignore files that begin with '.'
            filenames = [f for f in filenames if not f.startswith('.')]
            # only load images with the right extension
            filenames = [f for f in filenames if os.path.splitext(f)[1].lower() in extensions]
            filenames.sort()

            for f in filenames:
                imageFilenames.append([category, os.path.join(dirpath, f)])

    # filenames will be a list of two-tuples [category, filename]
    return imageFilenames


def _walk(top):
    """
    This is a (recursive) directory tree generator.
    What is a generator?
    See:
    http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    In short:
    - generators are iterables that can be iterated only once
    - their values are not stored in contrast e.g. to a list
    - 'yield' is 'like' return    
    """

    # 1. collect directory names in dirs and
    #    non-directory names (filenames) in nondirs
    names = os.listdir(top)
    dirs, nondirs = [], []
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    # 2. "return" information about directory names and filenames
    yield top, dirs, nondirs

    # 3. recursively process each directory found in current top
    #    directory
    for name in dirs:
        path = os.path.join(top, name)
        for x in _walk(path):
            yield x
