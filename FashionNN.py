"""
This project trains a neural network model to classify images of clothing


"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#importing and loading Fashion MNIST data from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
#Labels are an array of integers going from 0 to 9, corresponding to the class of clothing.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#The index of the item in the array corresponds to the label.

"""
60,000 images in training set
60,000 labels in training set
10,000 images in test set
10,000 labels in test set

"""
shape_train = train_images.shape
num_labels = len(train_labels)
shape_test = test_images.shape

#PREPROCESSING THE DATA

#inspecting first image in training set
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#scale values to a range of 0 to 1
train_images = train_images / 255.0

test_images = test_images / 255.0

#display first 25 images from training set and display class name below image

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#BUILD THE MODEL

#Set up the layers
"""
Sequential means a linear stack of layers
Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
Dense layers: The first Dense layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.

"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Compile the model
"""
Optimizer: how the model is updated based on the data and the loss function
Loss function: how accurate the model is during training. Want to minimize this function to move towards right direction
Metrics: used to monitor training and testing steps. ex below uses accuracy (fraction of images correctly classified)

"""
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
Training neural network has following steps:
    1. Feed training data to the model.
    2. Model learns to associate images and labels.
    3. You ask model to make predictions about test set.
    4. Verify predictions match the labels from test_labels.
"""

#Feed the model
#epoch is like the number of passes a training dataset takes around an algorithm.
model.fit(train_images, train_labels, epochs=10)

#Evaluate accuracy
#verbose is the level of detail you want to see
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

"""
Accuracy on test dataset is less than accuracy on training dataset.
This gap represents overfitting (ML model performs worse on new inputs than on training data).
Overfitted model fits to some of the noise which negatively impacts performance.
"""

#Make predictions
#Softmax layer converts the linear outputs (logits) to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

#Functions to graph full set of 10 class predictions

def plot_image(i, predictions_array, true_labels, imgs):
  true_label, img = true_labels[i], imgs[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  
def plot_value_array(i, predictions_array, true_labels):
  true_label = true_labels[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#for 0th image: 
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


#Plot first X test images, predicted labels, and true labels
#Correct predictions in blue and incorrect preductions in red.

def plot_img_with_pred (num_rows, num_cols, i):
  plt.subplot(num_rows, 2*num_cols, 2*i + 1)
  plot_image(i, predictions[i], test_labels, test_images) 
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)

  
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plot_img_with_pred(num_rows, num_cols, i)
plt.tight_layout()
plt.show()


#Making a prediction on a single image

# Grab an image from the test dataset.
img = test_images[1]
# Add the image to a batch where it's the only member. shape is (1, 28, 28)
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
print("Predicted label is", np.argmax(predictions_single[0]))

