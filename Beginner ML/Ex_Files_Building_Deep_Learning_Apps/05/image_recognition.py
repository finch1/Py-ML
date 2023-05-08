import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50
import matplotlib.pyplot as plt 

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model), using Keras
img = image.load_img("D:\\Ex_Files_Building_Deep_Learning_Apps\\05\\bay.jpg", target_size=(224,224))
plt.imshow(img)
plt.show()
# Convert the image to a numpy array
x = image.img_to_array(img)

# The NN expects us to pass in an array and multiple images at once. Right now, we
# only have one image. Fix: add a 4th dimension to our array.
# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis = 0)

# Scale the input image to the range used in the trained network (normalize)
x = resnet50.preprocess_input(x)

import xlsxwriter
workbook = xlsxwriter.Workbook("D:\\Ex_Files_Building_Deep_Learning_Apps\\05\\image.xlsx")
worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(x[0]):
    worksheet.write_column(row, col, data)
workbook.close()

# Run the image through the deep neural network to make a prediction
""" This will return a predictions object. The predictions object is a 1000 element array
of float numbers. each element in the array tells us how likely our picture contains 
each of the 1000 objects the model is trained to recognize. """
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
# Tell us the names of the most likely matches instead of all 1000. 
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

