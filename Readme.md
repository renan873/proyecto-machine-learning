# Predicción de la calidad del sueño mediante aprendizaje automático

Proyecto integrador de **Aprendizaje Automático y Minería de Datos**, Maestría en Gestión y Analítica de Datos.

**Docente:** Adriana Collaguazo Jaramillo, Mg.  
**Integrantes:** Renan Rivadeneira, María Sol Tuárez, Diego Niquinga y Gabriel Estupiñán.

[Ver notebook](Proyecto_Machine_Learning.ipynb) · [Abrir en Google Colab](https://colab.research.google.com/github/renan873/proyecto-machine-learning/blob/main/Proyecto_Machine_Learning.ipynb)

## 1. Definición del problema

**Dataset:** Sleep Health and Lifestyle Dataset.  
**Tipo de problema:** regresión supervisada.  
**Modelos:** regresión lineal múltiple y Random Forest de regresión.

**Objetivo general:** desarrollar y evaluar un modelo de aprendizaje automático supervisado de regresión para predecir la puntuación de calidad del sueño (`Quality of Sleep`).

**Pregunta:** ¿qué tan bien se puede estimar la calidad del sueño a partir de la duración del descanso, la frecuencia cardíaca y el nivel de estrés?

El modelo aprende de registros que contienen las variables predictoras y la puntuación conocida. Después se evalúa con registros separados del entrenamiento.

## 2. Obtención del dataset

**Fuente:** [Sleep Health and Lifestyle Dataset — Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).

**Archivo utilizado:** `Sleep_health_and_lifestyle_dataset.csv`, separado por punto y coma (`;`). La carga ejecutada contiene **374 filas y 14 columnas**. Se utilizan estas dimensiones observadas, aunque la descripción introductoria del notebook mencione 400 filas y 13 columnas.

Para reproducir el trabajo debe incorporarse a `datos/` el CSV exacto utilizado por el grupo y documentarse la licencia de la fuente. La versión trabajada contiene la presión arterial separada en dos columnas.

| Variable | Papel en la regresión | Significado |
|---|---|---|
| `Quality of Sleep` | Objetivo (`y`) | Puntuación de calidad del sueño. |
| `Sleep Duration` | Predictor (`X`) | Duración del sueño en horas. |
| `Heart Rate` | Predictor (`X`) | Frecuencia cardíaca en latidos por minuto. |
| `Stress Level` | Predictor (`X`) | Puntuación del nivel de estrés. |

## 3. Exploración: qué muestran los datos

El notebook revisa dimensiones, tipos de datos, primeras filas, estadísticas descriptivas y correlaciones de Pearson mediante un mapa de calor.

| Variable | Media | Mínimo | Máximo |
|---|---:|---:|---:|
| Calidad del sueño | 7,31 puntos | 4 | 9 |
| Duración del sueño | 7,13 horas | 5,8 | 8,5 |
| Frecuencia cardíaca | 70,17 latidos/minuto | 65 | 86 |
| Nivel de estrés | 5,39 | 3 | 8 |

**Lectura:** la puntuación de calidad varía entre 4 y 9. El proyecto busca explicar parte de esa variación mediante las tres variables predictoras. Las correlaciones permiten explorar asociaciones, pero no demuestran causalidad.

## 4. Preparación de los datos

1. Se comprueban los tipos y valores disponibles. Las cuatro variables de regresión tienen 374 valores no nulos en la salida de `df.info()`.
2. Se convierten las columnas numéricas con `pd.to_numeric(errors='coerce')`. El código no implementa imputación: una nueva versión del CSV debe comprobarse después de esta conversión.
3. Se seleccionan duración del sueño, frecuencia cardíaca y estrés como predictores; `Quality of Sleep` es la variable objetivo. No se utilizan identificadores ni variables categóricas como entradas de estos modelos.
4. Se divide la muestra en **280 registros para entrenamiento (75 %)** y **94 para prueba (25 %)**, con `random_state=42`.

Los dos modelos de regresión emplean esta misma partición y las variables sin estandarizar.

## 5. Selección y entrenamiento de los modelos

### Random Forest: exploración con SHAP

Se entrena un bosque de regresión para explorar relaciones que pueden ser no lineales. Su configuración es:

| Parámetro | Valor |
|---|---:|
| `n_estimators` | 300 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |
| `n_jobs` | -1 |

SHAP explica las predicciones del bosque para los 94 registros de prueba, usando entrenamiento como referencia (`Independent`, `max_samples=280`). El notebook incluye un gráfico de barras, un gráfico de distribución y este ranking:

| Variable | Media del valor absoluto SHAP |
|---|---:|
| Duración del sueño | 0,751198 |
| Nivel de estrés | 0,345518 |
| Frecuencia cardíaca | 0,072697 |

**Interpretación:** la duración del sueño tiene la mayor contribución media a las predicciones del bosque, seguida del estrés. La frecuencia cardíaca aporta menos en este ajuste. Los valores están expresados en puntos de la salida del modelo; no son porcentajes ni efectos causales. Esta tabla de valores absolutos no indica por sí sola la dirección de cada contribución.

### Regresión lineal múltiple: estimación de la puntuación

Se utiliza por su interpretación sencilla: estima una puntuación numérica a partir de una combinación de los tres predictores. Se entrena con `LinearRegression()`.

La ecuación obtenida, con coeficientes redondeados, es:

```text
Calidad estimada = 5,3629
                + 0,7203 × duración del sueño
                - 0,0218 × frecuencia cardíaca
                - 0,3067 × nivel de estrés
```

Manteniendo constantes las otras variables, una hora adicional de sueño se asocia con 0,72 puntos más de calidad estimada; un punto adicional de estrés, con 0,31 puntos menos. Estos coeficientes describen el ajuste, no efectos causales demostrados.

## 6. Evaluación y resultados

Las siguientes métricas corresponden a la **regresión lineal sobre los 94 registros de prueba**:

| Métrica | Resultado | Interpretación |
|---|---:|---|
| R² | 0,889 | Explica aproximadamente el 88,9 % de la variabilidad en prueba respecto a la referencia de su media. |
| MSE | 0,172 | Error cuadrático medio, en puntos al cuadrado. |
| RMSE | 0,415 | Error en puntos de calidad, con mayor penalización de errores grandes. |
| MAE | 0,334 | Desviación absoluta media de 0,334 puntos entre predicción y valor real. |

El gráfico de valores reales frente a predichos permite observar la distancia de las estimaciones a la línea de predicción perfecta. **R² no representa un porcentaje de predicciones correctas.**

El Random Forest está entrenado y explicado mediante SHAP, pero el notebook no presenta sus métricas de regresión. Por tanto, todavía no puede afirmarse que uno de los dos modelos prediga mejor que el otro.

## 7. Ajuste y mejora

El notebook utiliza una configuración por modelo y conserva los tres predictores. SHAP aporta interpretación, pero no constituye una mejora predictiva demostrada.

Para completar esta etapa de la guía, queda pendiente probar una alternativa y comparar sus resultados: por ejemplo, variar los predictores o los parámetros del bosque mediante validación cruzada sobre entrenamiento y evaluar la configuración seleccionada en prueba.

El ranking SHAP calculado sobre prueba debe tratarse como explicación posterior. Si se utiliza para elegir variables, esa partición deja de ser una evaluación final independiente.

## 8. Conclusiones y limitaciones

- La regresión lineal estima la puntuación con un error absoluto medio de **0,334 puntos** en la partición evaluada.
- En el bosque exploratorio, la duración del sueño y el estrés presentan las mayores contribuciones medias según SHAP. Este ranking describe el bosque, no los coeficientes del modelo lineal.
- Falta evaluar Random Forest con las mismas métricas para realizar una comparación entre modelos.
- Una sola partición de 374 registros no garantiza el mismo desempeño en otras poblaciones ni demuestra ausencia de sobreajuste.
- El trabajo tiene finalidad académica. Las asociaciones encontradas no constituyen recomendaciones clínicas.

**Origen de los resultados:** salidas guardadas en `Pproyecto_Machine_Learning.ipynb`. No se ha repetido la ejecución completa porque el CSV original no está entre los archivos disponibles para esta revisión.

## 9. Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `README.md` | Objetivo, datos, metodología, resultados y ejecución. |
| `Proyecto_Machine_Learning.ipynb` | Notebook del proyecto. |
| `datos/Sleep_health_and_lifestyle_dataset.csv` | CSV exacto utilizado; pendiente de incorporar. |
| `requirements.txt` | Dependencias de Python. |

El archivo correcto recibido como `Pproyecto_Machine_Learning.ipynb` debe reemplazar al anterior en GitHub con el nombre `Proyecto_Machine_Learning.ipynb`, para conservar estos enlaces.

## 10. Cómo ejecutar

### Preparación
### Google Colab

1. Abrir el notebook actualizado desde el enlace del inicio y guardar una copia en Drive.
2. Subir el CSV exacto a la raíz de «Mi unidad», con el nombre `Sleep_health_and_lifestyle_dataset.csv`.
3. Subir `requirements.txt` al entorno de Colab y ejecutar antes de las importaciones:

   ```python
   %pip install -r /content/requirements.txt
   ```

4. Ejecutar las celdas en orden y autorizar el montaje de Drive. La lectura utilizada es:

   ```python
   df = pd.read_csv('/content/drive/MyDrive/Sleep_health_and_lifestyle_dataset.csv', delimiter=';')
   ```

5. Comprobar la dimensión `(374, 14)` y ejecutar hasta la evaluación de regresión lineal, incluidos Random Forest y SHAP. Las secciones posteriores del notebook quedan fuera del objetivo documentado aquí.

### Jupyter local

1. Clonar el repositorio e instalar las dependencias, preferentemente en un entorno virtual:

   ```bash
   git clone https://github.com/renan873/proyecto-machine-learning.git
   cd proyecto-machine-learning
   python -m pip install -r requirements.txt
   python -m notebook
   ```

2. Guardar el CSV en `datos/` y abrir `Proyecto_Machine_Learning.ipynb`.
3. Omitir las celdas de `google.colab` y montaje de Drive. Reemplazar la lectura por:

   ```python
   df = pd.read_csv('datos/Sleep_health_and_lifestyle_dataset.csv', delimiter=';')
   ```

4. Ejecutar en orden hasta la evaluación de regresión, con la importación del bosque añadida. Conservar los nombres del CSV, incluidos los espacios finales de las columnas de presión, porque las celdas exploratorias los utilizan.

Las dependencias incluyen `shap` para explicar el bosque, `ipython` para su inicialización visual y `notebook` para Jupyter. Se mantienen también las librerías importadas por el archivo recibido. Sus versiones originales no quedaron registradas, por lo que no se garantiza identidad numérica entre entornos.
