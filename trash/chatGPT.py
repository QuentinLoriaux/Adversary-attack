'''

A tester. Si ça marche jsuis écoeuraient

'''




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger un modèle pré-entraîné (remplacez-le par votre propre modèle)
model = keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# Fonction pour générer des adversaires
def generate_adversary(model, image, label, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image)  # Convertir l'image en tenseur TensorFlow
    label_tensor = tf.convert_to_tensor(label)  # Convertir l'étiquette en tenseur TensorFlow

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)

    gradient = tape.gradient(loss, image_tensor)
    perturbation = epsilon * tf.sign(gradient)

    adversarial_image = image_tensor + perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)  # Clipper les valeurs pour rester dans la plage [0, 1]

    return adversarial_image.numpy()

# Exemple d'utilisation
image_example = tf.random.normal((1, 224, 224, 3))  # Remplacez cela par votre propre image
label_example = tf.constant([7])  # Remplacez cela par l'étiquette réelle de votre exemple

adversarial_example = generate_adversary(model, image_example, label_example)
