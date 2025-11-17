# Reconocimiento de Acciones Humanas en UCF101 mediante Esqueletos 2D

## Introducción

El conjunto de datos **UCF101** es uno de los benchmarks clásicos para el
reconocimiento de acciones en video.  Fue introducido en 2012 y contiene
**101** clases de acciones humanas como *tocar la guitarra*, *surfear* o
*hacer punto*.  Las acciones están agrupadas en cinco grandes tipos
(interacción con objetos, sólo movimiento corporal, interacción
humano‑humano, tocar instrumentos musicales y deportes).  El dataset
incluye **13 320** clips de video obtenidos de YouTube, cada uno
etiquetado con una única acción【428718739272216†screenshot】.  Los clips presentan
variaciones de punto de vista, movimiento de cámara y condiciones de
iluminación, lo que convierte a UCF101 en un reto significativo para
las técnicas de aprendizaje profundo【428718739272216†screenshot】.  El conjunto
completo se divide en tres subconjuntos: *train*, *validation* y
*test*, donde el 75 % de las muestras de cada clase se utiliza para
entrenamiento y el 25 % restante se distribuye equitativamente entre
validación y prueba【428718739272216†screenshot】.  La versión original del
trabajo reportó un modelo de bolsa de palabras que alcanzó un
44,5 % de exactitud en UCF101【367233494771988†L18-L27】.  Modelos más
recientes, como la red I3D preentrenada en Kinetics, superan el 98 %
de exactitud al ser ajustados en UCF101【215100582023531†L20-L27】.

## Anotaciones de esqueletos 2D

Para reducir la complejidad computacional frente a los modelos que
procesan video bruto, se proporciona un archivo con anotaciones de
esqueletos 2D para cada video.  Este archivo (en formato *pickle*)
contiene dos campos:

- **`split`**: un diccionario que mapea el nombre de cada partición (p.
  ej. *train*, *val*) con la lista de identificadores de los videos
  pertenecientes a esa partición.
- **`annotations`**: una lista de anotaciones donde cada elemento es un
  diccionario con los metadatos del video y los datos de los puntos
  clave.  Cada anotación incluye el identificador del video, el número
  de frames, la forma de la imagen y el label de acción.  El campo
  principal es **`keypoint`**, un arreglo NumPy de dimensión
  `[M × T × V × C]` que almacena las coordenadas de cada punto clave
  (persona × tiempo × puntos × coordenadas).  Para esqueletos 2D,
  `C=2` (eje *X* e *Y*) y se proporciona además un arreglo de
  puntajes de confianza por keypoint【374822187822292†L220-L247】.  Esta
  estructura permite trabajar con secuencias de puntos clave sin
  necesidad de procesar los píxeles de la imagen.

## Estrategia y flujo de trabajo

1. **Selección del subconjunto de clases**.
   Debido a las limitaciones de recursos, se puede trabajar con un
   subconjunto de acciones (p. ej. cinco clases) que presenten
   movimientos variados y difícil discriminación.  El código permite
   especificar una lista de índices de clase y limitar el número de
   videos por clase.

2. **Carga y preprocesado de datos**.
   Utilizamos la estructura `split` del archivo `ucf101_2d.pkl` para
   seleccionar los videos de entrenamiento y validación.  Cada
   entrada del dataset devuelve un tensor de forma `[T, V·C]`, donde
   `T` es el número de frames deseado (se muestrean uniformemente o se
   repite el último frame si el video es corto).  Las coordenadas se
   normalizan restando la media y dividiendo por la desviación
   estándar por secuencia para mitigar diferencias de escala entre
   videos.

3. **Modelos de aprendizaje profundo**.

   - **Modelo base (MLP)**.  Como línea base implementamos una
     perceptrón multicapa que aplana toda la secuencia en un único
     vector y lo procesa a través de dos capas ocultas con funciones
     ReLU.  Este modelo ignora el orden temporal de los frames y
     actúa como referencia sencilla.

   - **Modelo propuesto (LSTM)**.  Para capturar la dinámica temporal
     se emplea una red LSTM que recibe como entrada la secuencia de
     vectores `[T, V·C]`.  La salida oculta del último paso temporal
     se conecta a una capa totalmente conectada para clasificar la
     acción.  El número de capas y unidades ocultas se parametriza
     mediante argumentos de línea de comandos.

   Otros modelos posibles incluyen grafos convolucionales (ST‑GCN,
   AGCN) que aprovechan la conectividad del esqueleto y redes 3D
   convolucionales como C3D o I3D.  Estos últimos han reportado
   resultados superiores (≈98 % de exactitud) cuando se preentrenan en
   grandes conjuntos como Kinetics【215100582023531†L20-L27】, pero su
   entrenamiento desde cero es costoso.  La arquitectura modular del
   script facilita la sustitución por modelos más avanzados.

4. **Entrenamiento y validación**.
   Se definen *dataloaders* para las particiones de entrenamiento y
   validación con el tamaño de lote especificado.  Se utilizan
   `CrossEntropyLoss` y optimización Adam.  Tras cada época se
   calcula la pérdida media en entrenamiento y la exactitud en
   validación.  El script soporta entrenamiento en CPU o GPU.

5. **Evaluación y mejoras**.
   La exactitud en el subconjunto de validación permite comparar la
   línea base con el modelo LSTM.  Para mejorar el desempeño se
   pueden introducir técnicas de regularización (dropout, early
   stopping), ajustar hiperparámetros (número de frames `T`, tamaño
   oculto, tasa de aprendizaje) o emplear modelos de grafos que
   incorporen explícitamente la topología del esqueleto.

## Implementación

El archivo `project/ucf101_skeleton_action_recognition.py` contiene la
implementación completa.  Los puntos clave son:

* **UCF101SkeletonDataset**: clase que carga el fichero
  `ucf101_2d.pkl`, filtra videos por partición y clase, muestrea
  uniformemente los frames y normaliza las coordenadas.  Devuelve
  tensores listos para PyTorch.

* **BaselineMLP** y **LSTMClassifier**: modelos de referencia y
  propuesta.  Ambos reciben tensores de forma `[batch, T, V·C]`.
  El modelo MLP aplana la secuencia y la pasa por capas densas,
  mientras que el LSTM procesa la secuencia y utiliza el último
  estado oculto para predecir la acción.

* **Funciones de entrenamiento y evaluación**: implementan el ciclo
  estándar de aprendizaje supervisado (cálculo de pérdida,
  retropropagación y medición de exactitud).

Para ejecutar el script se necesita descargar previamente el archivo
de anotaciones de esqueletos y colocarlo en una ruta accesible.
Ejemplo de ejecución con cinco clases seleccionadas (supóngase que
`12, 20, 34, 55, 70` corresponden a acciones relevantes):

```bash
python project/ucf101_skeleton_action_recognition.py \
  --annotation-path /ruta/al/ucf101_2d.pkl \
  --train-split train --val-split val \
  --selected-classes 12 20 34 55 70 \
  --epochs 15 --batch-size 8 --model lstm --hidden-dim 128
```

Los resultados de la línea base y del modelo LSTM pueden compararse
con la exactitud reportada para técnicas más avanzadas.  Con
preprocesado y ajuste adecuado, la red LSTM debería superar con
claridad al modelo MLP al capturar la secuencia temporal de
movimiento.  Para lograr un desempeño competitivo con el estado del
arte, se recomienda explorar modelos de grafos o redes 3D
convolucionales preentrenadas.
