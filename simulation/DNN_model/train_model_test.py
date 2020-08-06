import create_data
import numpy as np
from keras import layers
from keras import models

image_shape = 100

(trajectory_full_robot_pose,
trajectory_full_robot_velocity,
trajectory_full_straw_1,
trajectory_full_straw_2,
trajectory_full_straw_3,
trajectory_full_straw_4,
trajectory_full_straw_5) = create_data.build_trajectory_data_for_DNN()


# Sort the dataset into input-output samples:
# INPUT DATA STRUCTURE:
# 1. robot trajectory pos for i^n to i^n+10 -> 11 * 7 (77) inputs
# 2. robot trajectory vel for i^n to i^n+10 -> 11 * 7 (77) inputs
# 3. strawberry 1 pose at i^n -> 3 inputs
# 4. strawberry 2 pose at i^n -> 3 inputs
# 5. strawberry 3 pose at i^n -> 3 inputs
# 6. strawberry 4 pose at i^n -> 3 inputs
# 7. strawberry 5 pose at i^n -> 3 inputs
# TOTAL INPUT SIZE = 159

# OUTPUT DATA STRUCTURE:
# 1. strawberry 1 pose at i^n+1 to i^n+10 -> 3 * 10 (30) outputs
# 2. strawberry 2 pose at i^n+1 to i^n+10 -> 3 * 10 (30) outputs
# 3. strawberry 3 pose at i^n+1 to i^n+10 -> 3 * 10 (30) outputs
# 4. strawberry 4 pose at i^n+1 to i^n+10 -> 3 * 10 (30) outputs
# 5. strawberry 5 pose at i^n+1 to i^n+10 -> 3 * 10 (30) outputs
# TOTAL OUTPUT SIZE = 150

# 11,000 full trajectory samples. 
# each of length 400 
input_trajetories = 

# 1. Build CNN model.
# print("Building CNN model")
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_shape, image_shape, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(layers.Dense(2, activation='softmax'))
# model.summary()

# # 2. Compile the model
# print("Compiling the model")
# model.compile(loss= 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# # 3. Make the data for training and testing
# train_images, train_labels, test_images, test_labels, val_images, val_labels = make_data.make_data(image_shape)

# # 4. Train the model.
# print("Training the model")
# model.fit(np.array(train_images), np.array(train_labels), batch_size=64, epochs=10, verbose=1, validation_data=(np.array(val_images), np.array(val_labels)))

# # 5. Evaluation
# print("Evaluating the model's performance")
# test_loss, test_acc = model.evaluate(np.array(test_images), np.array(test_labels))
# print('Test accuracy:', test_acc)