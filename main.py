import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2 #Предтренировочная модель

#Модель нейронной сети PurrWoofNet-AI
#Улучшенная версия PurrWoof-AI на архитектуре MobileNetV2 .

# Константы
EPOCHS = 5
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Загрузка и разделение данных на обучающее и проверочное множества
(train_dataset, validation_dataset, test_dataset) = tfds.load(
    name='cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    as_supervised=True,
    shuffle_files=True #Перемешивание файлов
)
# ограничение в 50 картинок. Для отладки.
#train_dataset = train_dataset.take(50)#берём 50 картинок из тренировочного набора данных
#validation_dataset = validation_dataset.take(50)#берём 50 картинок из проверочного набора данных

# Обработка данных
def preprocess(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=2)
    return image, label

# Ставим метки
train_dataset = train_dataset.map(preprocess).batch(BATCH_SIZE)
validation_dataset = validation_dataset.map(preprocess).batch(BATCH_SIZE)
test_dataset = test_dataset.map(preprocess).batch(BATCH_SIZE)


#Предварительная (prefetching) загрузка данных
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


#Аугментация тренировочных данных
def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.random_flip_left_right(image)
    return image, label

#Применение агументации к тренировочным данным
train_dataset = train_dataset.map(augment)


# Имена классов
class_names = ["Кошка", "Собака"]

# Определение параметров обучения
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']


#Загрузка предварительной модели
mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_batch, label_batch = next(iter(train_dataset))
feature_batch = mobilenet(image_batch)
print(feature_batch.shape)

mobilenet.trainable = False 

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Инициализация модели
# Для бинарной классификации используется оптимизатор Adam Adaptive Moment Estimation
#В качестве основной архитектуры используется MobileNetV2
model = tf.keras.Sequential([
    mobilenet,#предтренировочная модель MobileNetV2
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(2, activation='softmax')
])
#На выходе два нейрона потому что бинарная классификация, т.е собака и кот - два варианта.

#Компиляция модели
model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)


# Обучение модели
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[tensorboard_callback])

# Построение графика точности и потерь
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_images, test_labels = next(iter(test_dataset.take(10)))

# Получение предсказаний нейросети для 10 изображений
predictions = model.predict(test_images)

# Преобразование меток из one-hot encoding в обычный вид
test_labels = np.argmax(test_labels, axis=1)

# Вывод 10 изображений и соответствующих им меток и предсказаний
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    # Отображение изображения
    ax.imshow(test_images[i])
    # Отображение меток и предсказаний
    true_label = class_names[test_labels[i]]
    pred_label = class_names[np.argmax(predictions[i])]
    if true_label == pred_label:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='green')
    else:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='red')

plt.tight_layout()
plt.show()
