import tensorflow as tf
devices = tf.config.list_physical_devices()
print(devices)

# You should see 'GPU' in the list if it's working
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the Mac GPU!")