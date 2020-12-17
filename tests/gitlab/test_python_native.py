import mnist
import json
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import fledge.native as fx
from fledge.federated import FederatedModel, FederatedDataSet

train_images = mnist.train_images()
train_labels = to_categorical(mnist.train_labels())
valid_images = mnist.test_images()
valid_labels = to_categorical(mnist.test_labels())


def preprocess(images):
    # Normalize
    images = (images / 255) - 0.5
    # Flatten
    images = images.reshape((-1, 784))
    return images


# Preprocess the images.
train_images = preprocess(train_images)
valid_images = preprocess(valid_images)

feature_shape = train_images.shape[1]
classes = 10

fl_data = FederatedDataSet(train_images, train_labels, valid_images, valid_labels,
                           batch_size=32, num_classes=classes)


def build_model(feature_shape, classes):
    # Defines the MNIST model
    model = Sequential()
    model.add(Dense(64, input_shape=feature_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


fl_model = FederatedModel(build_model, data_loader=fl_data)

collaborator_models = fl_model.setup(num_collaborators=2)
collaborators = {'one': collaborator_models[0], 'two': collaborator_models[1]}
print(f'Original training data size: {len(train_images)}')
print(f'Original validation data size: {len(valid_images)}\n')

# Collaborator one's data
print(f'Collaborator one\'s training data size: \
        {len(collaborator_models[0].data_loader.X_train)}')
print(f'Collaborator one\'s validation data size: \
        {len(collaborator_models[0].data_loader.X_valid)}\n')

# Collaborator two's data
print(f'Collaborator two\'s training data size: \
        {len(collaborator_models[1].data_loader.X_train)}')
print(f'Collaborator two\'s validation data size: \
        {len(collaborator_models[1].data_loader.X_valid)}\n')

print(json.dumps(fx.get_plan(), indent=4, sort_keys=True))
final_fl_model = fx.run_experiment(collaborators,
                                   override_config={'aggregator.settings.rounds_to_train': 5})
# Save final model
final_fl_model.save_native('final_model')
