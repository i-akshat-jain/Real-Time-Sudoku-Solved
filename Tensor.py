#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split


#%%
#Get the data
classes = 10 
images = []
labels = [] 

for i in range(0, classes):
    imgList = os.listdir("C:/Users/aksha/Documents/Project of python/AR Sudoku/Data/Data/"+str(i))
    for j in imgList:
        currentImg = cv2.imread("C:/Users/aksha/Documents/Project of python/AR Sudoku/Data/Data/"+str(i)+'/'+str(j))
        currentImg = cv2.resize(currentImg, (28, 28))
        images.append(currentImg)
        labels.append(i)
        
print(len(images))
print(len(labels))


#%%
images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)


#%%
plt.imshow(images[2000])


#%%
#splitting the data to test and train

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.3)
print(train_images.shape)
print(test_images.shape)


#%%
train_images,  val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)
print(train_images.shape)
print(val_images.shape)


#%%
#Processing the data
def preprocessing(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    x ,img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
    img = img/255
    return img

train_images = np.array(list(map(preprocessing, train_images)))
test_images = np.array(list(map(preprocessing, test_images)))
val_images = np.array(list(map(preprocessing, val_images)))


#%%
print(train_images.shape)
print(test_images.shape)
print(val_images.shape)


#%%
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
val_images = val_images.reshape(val_images.shape[0], 28, 28, 1)


#%%

print(train_images.shape)
print(test_images.shape)
print(val_images.shape)


#%%
plt.imshow(images[9200])


#%%
#Defining Callback
class calling_Back(tf.keras.callbacks.Callback):
    def epoch_eng(self, epoch, log  = {}):
        if(logs.get('val_accuracy')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = true


#%%
#Building CNN Model
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1),  activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Conv2D(64, (3,3),  activation="relu"),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation="relu"),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)   
])


#%%
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()


#%%
#Image Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                         height_shift_range= 0.1,
                                                         zoom_range= 0.1,
                                                         rotation_range= 10)


#%%
#Training the model
callbacks = calling_Back()
history = model.fit(datagen.flow(train_images, train_labels),
                              epochs = 30, validation_data = (val_images, val_labels), callbacks = [callbacks])


#%%

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


#%%
#Evaluating the model test
score = model.evaluate(test_images, test_labels, verbose = 1)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


#%%
model.save('Digit_Recognizer')
model.save("Digit_Recognizer.h5")





