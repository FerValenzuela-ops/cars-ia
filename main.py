import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
# Directorio que contiene las imágenes y los archivos XML
data_dir = '/cars'


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    label = None

    # Obtener las anotaciones y la etiqueta del elemento raíz
    for child in root:
        if child.tag == 'object':
            obj_name = child.find('name').text
            bbox = child.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            annotation = {'name': obj_name, 'bbox': (xmin, ymin, xmax, ymax)}
            annotations.append(annotation)

        elif child.tag == 'label':
            label = child.text

    return annotations, label

# Leer las imágenes y las anotaciones, y generar los datos de entrenamiento
def generate_data(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(data_dir, filename)
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(data_dir, xml_filename)

            if os.path.exists(xml_path):
                # Leer la imagen
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))

                # Procesar el archivo XML y obtener las anotaciones y la etiqueta
                annotations, label = parse_xml(xml_path)

                images.append(image)
                labels.append(label)

    # Convertir las listas a matrices numpy
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Generar los datos de entrenamiento
images, labels = generate_data(data_dir)

# Codificar las etiquetas en valores numéricos
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Aplicar aumento de datos a las imágenes de entrenamiento
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10)
datagen.fit(X_train)

# Calcular el número de clases (tipos de automóviles)
num_classes = len(np.unique(labels_encoded))

# Cargar el modelo base VGG16 pre-entrenado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Agregar capas adicionales al modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


# Construir el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=10)

# Guardar el modelo entrenado
model.save('/models/car_recognition_model.h5')

# Guardar el codificador de etiquetas
np.save('/models/label_encoder.npy', label_encoder.classes_)
