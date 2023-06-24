# PeopleTracker

Para instalar en un environment nuevo:

```
conda create -n PeopleTracker
conda install -n PeopleTracker python
conda activate PeopleTracker
pip install -r requirements.txt
git clone https://github.com/mikel-brostrom/yolo_tracking.git
cd yolo_tracking
pip install -v -e .
```

El código está organizado de la siguiente manera:
.  
├── csrt_example.py  
├── \_\_init\_\_.py  
├── model_files  
│   ├── Acá se descargan archivos .pt para el model YOLO (lo hace automáticamente)  
├── parse.py  
├── pipelines  
│   ├── csrt_full.py  
│   ├── detectors  
│   │   ├── hog.py  
│   │   ├── \_\_init\_\_.py  
│   │   └── yolo.py  
│   ├── \_\_init\_\_.py  
│   ├── models.py  
│   ├── trackers  
│   │   ├── bytetracker.py  
│   │   └── \_\_init\_\_.py  
│   └── yolo_full.py  
├── README.md  
├── requirements.txt  
├── utils  
│   ├── dataloaders.py  
│   ├── drawing.py  
│   ├── \_\_init\_\_.py  
│   ├── object_selection.py  
│   └── video_editing.py  
└── yolo_example.py  

Cada carpeta contiene:

* Model_files: Carpeta donde se guardan los archivos .pt descargados de YOLO.
* Pipelines: Código con los modelos de detección y trackeo.
* Utils: Código con funciones auxiliares (para cargar videos, dibujar, seleccionar sobre el video, etc.).
* parse.py: Código para parsear los argumentos de la línea de comandos (opciones como elegir archivo de video, a donde guardar, etc.).
* csrt_example.py: Ejemplo de uso del modelo CSRT.
* yolo_example.py: Ejemplo de uso del modelo YOLO.
* requirements.txt: Archivo con las dependencias del proyecto.

La manera más simple de correr un ejemplo es ejecutar csrt_example.py o yolo_example.py. Por ejemplo:

```
python csrt_example.py --video <archivo de video>
```

También se pueden pasar carpetas con listas de imágenes (ordenadas), por ejemplo:

```
python csrt_example.py --video <carpeta con imágenes> --is_img_folder
```

Esto es útil para por ejemplo los datasets típicos encontrados en http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html.

Si no se pasa argumento de video, el default es utilizar la cámara web. Para guardar la salida del video, se puede utilizar el argumento --outpath:

```
python csrt_example.py --video <archivo de video> --outpath <archivo a donde guardar>
```

Al ejecutar este ejemplo, se abrira una ventana de OpenCV en donde se podrá seleccionar el objeto (o los objetos) a trackear. Una vez seleccionado(s), se puede presionar la tecla 'q' para comenzar el trackeo. Por default, el ejemplo viene dado con la opció 'detect', que utiliza el detector HOG o YOLO para generar una lista de personas detectadas. Si se quiere pasar al próximo frame del video para ver si aparecen más detecciones, se puede hacerlo apretando la tecla 'n'. Si se quiere omitir el paso de detección y elegir manualmente los objetos a trackear con una ROI (Region of Interest), se puede hacerlo cambiando la opción del modelo a 'select', en el archivo csrt_example.py.

Para yolo_example.py, funciona únicamente con detecciones, ya que utiliza al modelo YOLO para hacer las detecciones y luego un tracker (ByteTracker) para trackear. Es similar al ejemplo anterior, pero no se puede elegir manualmente las regiones de interés.

Para detener el trackeo y cerrar el programa, se puede presionar la tecla 'q' una vez más.

## Información sobre los modelos

Se implementan dos modelos de trackeo con un funcionamiento distinto: 

* CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability): Es un modelo de trackeo que se basa en correlacionar distintas características del objeto a tracker con una región espacial en la cual se considera probable encontrar al objeto (cercano al frame anterior). Está implementado en OpenCV, así que es muy fácil de utilizar. Se inicializa con una región de interés (o varias, en cuyo caso simplemente se crean varias instancias de trackeo). Para esto, se planteó poder elegir las regiones de interés manualmente, o utilizar regiones detectadas por modelos de detección de personas, en particular HOG (Histogram of Gradients) y YOLO (You Only Look Once).

* YOLO+ByteTracker: Se combinan YOLO (You Only Look Once) versión 8 (en realidad se puede elegir la que se quisiera, pero está es la más reciente) para detectar a las personas en el frame y ByteTracker, que es un algoritmo MOT (multi-object tracking). ByteTracker se encarga de mantener una relación entre las detecciones sucesivas dadas por YOLO en cada frame, así creando una línea temporal para cada detección y entonces trackeando al objeto subyacente. Se utiliza la implementación de YOLO de ultralytics (en el fondo, PyTorch), y la de BoxMOT para ByteTracker (también PyTorch). 

En cuanto a ejemplos cualitativos hechos a ojo, se encontró que CSRT es en general más estable que YOLO+ByteTracker, a pesar de ser más simple. En algunos casos, ByteTracker se destacó en recuperar trazas pérdidas de objetos, y en distinguir entre objetos que se obstruyen entre ellos, pero también fue muy propenso a perder tracks por completo, incluso cuando "a simple vista" el objeto a trackear se ve claramente y no está obstruído por nada. Se considerá que probablemente se puedan obtener mejores resultados con un mejor preprocesamiento del video y los parámetros de los modelos.
