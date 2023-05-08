import numpy as np 
from keras.preprocessing import image
from keras.applications import resnet50

# load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# load the image file, resizing it to 224x224 pixels ( required by this model)
img = image.load_img("bay.jpg", target_size=(224, 224))

# Convert the image to a numpy array. Converts image to 3d array
x = image.img_to_array(img)

# add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# run the image through the deep NN to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. index zero is the result
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of: ")
for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))