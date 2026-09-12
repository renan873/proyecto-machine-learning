# Predicción de la calidad del sueño 

Proyecto integrador de **Aprendizaje Automático y Minería de Datos**, Maestría en Gestión y Analítica de Datos.  
**Docente:** Adriana Collaguazo Jaramillo, Mg.

**Integrantes:** Renan Rivadeneira, María Sol Tuárez, Diego Niquinga y Gabriel Estupiñán.

[Ver notebook](Proyecto_Machine_Learning.ipynb) · [Abrir en Google Colab](https://colab.research.google.com/github/renan873/proyecto-machine-learning/blob/main/Proyecto_Machine_Learning.ipynb)

## 1. Problema y objetivo

¿Es posible estimar la calidad del sueño a partir de su duración, la frecuencia cardíaca y el nivel de estrés?

El objetivo principal es predecir `Quality of Sleep` mediante **regresión lineal múltiple**. Se entrena además un **Random Forest de regresión** para explorar la contribución de las variables con **SHAP**. Como análisis complementario, una **regresión logística** distingue registros con trastorno del sueño de registros sin trastorno registrado.

Son dos tareas diferentes: una predice una puntuación y la otra una categoría. Sus métricas no permiten decidir cuál modelo es mejor que el otro.

## 2. Dataset y fuente

**Archivo utilizado:** `Sleep_health_and_lifestyle_dataset.csv`, separado por punto y coma (`;`). La ejecución guardada contiene **374 registros y 14 columnas** antes de crear la variable binaria.

**Fuente citada en el notebook:** [Sleep Health and Lifestyle Dataset — Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).

La descripción del notebook menciona 400 filas y 13 columnas, pero la carga ejecutada devuelve **374 filas y 14 columnas**; estas últimas son las dimensiones utilizadas en este README. La presión arterial aparece separada en dos columnas. Para reproducir los resultados debe compartirse el CSV exacto utilizado y registrar la licencia de la fuente.

| Variable utilizada | Función en el proyecto |
|---|---|
| `Quality of Sleep` | Puntuación objetivo de regresión; predictor en clasificación. |
| `Sleep Duration` | Horas de sueño; predictor en ambos modelos. |
| `Heart Rate` | Frecuencia cardíaca; predictor de regresión. |
| `Stress Level` | Nivel de estrés; predictor de regresión. |
| `Age` | Edad; predictor de clasificación. |
| `Blood Pressure Diastolic ` y `Blood Pressure Systolic ` | Columnas de presión usadas en clasificación. Los nombres contienen un espacio final. |
| `Sleep Disorder` | Categoría original usada para construir la etiqueta binaria. |

También se exploran actividad física y pasos diarios. `Person ID` no se utiliza como predictor.

## 3. Exploración: qué muestran los datos

Se revisan dimensiones, primeras filas, tipos, valores disponibles, estadísticas descriptivas y una matriz de correlación de Pearson.

| Indicador | Resultado observado |
|---|---:|
| Edad media | 42,18 años |
| Duración media del sueño | 7,13 horas |
| Duración mínima y máxima | 5,8–8,5 horas |
| Calidad media del sueño | 7,31 puntos |
| Calidad mínima y máxima | 4–9 puntos |
| Estrés medio | 5,39 |
| Frecuencia cardíaca media | 70,17 latidos/minuto |
| Pasos diarios medios | 6.816,84 |
| Registros con apnea | 78 |
| Registros con insomnio | 77 |
| Registros con `Sleep Disorder` leído como `NaN` | 219 |

La calidad del sueño varía entre 4 y 9 puntos: existe variación que el modelo busca explicar. El mapa de correlaciones permite explorar asociaciones, sin demostrar causalidad.

## 4. Preparación de los datos

1. Se convierten nueve columnas a formato numérico con `pd.to_numeric(errors='coerce')`. Las salidas muestran 374 valores disponibles en cada una antes de la conversión.
2. Se crea `Has Sleep Disorder` mediante `pd.isna(df['Sleep Disorder']).astype(int)`.
3. Se seleccionan únicamente predictores numéricos; no se codifican género, ocupación ni categoría de IMC.
4. Se divide cada tarea en **75 % de entrenamiento (280 registros)** y **25 % de prueba (94 registros)**, con `random_state=42`. En clasificación se conserva la proporción de clases mediante `stratify`.
5. Para regresión logística, `StandardScaler` se ajusta solo con entrenamiento y transforma entrenamiento y prueba. La regresión lineal usa los valores sin estandarizar.

**Interpretación de la etiqueta realmente programada:**

| Valor | Condición del código | Registros |
|---|---|---:|
| 0 | `Sleep Disorder` contiene apnea o insomnio | 155 |
| 1 | `Sleep Disorder` se leyó como `NaN` | 219 |

La clase 1 se interpreta aquí como **sin trastorno registrado**, siempre que se confirme que esos `NaN` provienen de la categoría original de ausencia de trastorno. Un dato desconocido no debe convertirse automáticamente en ausencia de enfermedad. Los nombres del reporte y de la matriz del notebook están invertidos respecto al código.

## 5. Modelos y resultados

### Random Forest y SHAP: contribución de las variables

Antes de ajustar la regresión lineal, se entrena un `RandomForestRegressor` con los mismos tres predictores y los mismos 280 registros de entrenamiento. Se configura con `n_estimators=300`, `min_samples_leaf=2`, `random_state=42` y `n_jobs=-1`.

SHAP utiliza entrenamiento como referencia (`Independent`, `max_samples=280`) y explica las predicciones de los 94 registros de prueba. El notebook presenta un gráfico de barras, un gráfico de distribución de contribuciones y el siguiente ranking:

| Variable | Media del valor absoluto SHAP |
|---|---:|
| Duración del sueño (`Sleep Duration`) | 0,751198 |
| Estrés (`Stress Level`) | 0,345518 |
| Frecuencia cardíaca (`Heart Rate`) | 0,072697 |

**Lectura:** la duración del sueño presenta la mayor contribución media a las predicciones del bosque, seguida del estrés. La frecuencia cardíaca aporta menos en este ajuste. Estos valores están en puntos de la salida del modelo, no son porcentajes ni efectos causales. Al ser valores absolutos, la tabla no indica por sí sola si cada variable aumenta o disminuye la predicción.

Este ranking explica el **Random Forest**, no los coeficientes de la regresión lineal. El notebook no calcula R², RMSE o MAE para el bosque, por lo que aún no permite comparar su desempeño predictivo con el modelo lineal.

### Regresión lineal: predecir la puntuación

Se elige por su interpretación sencilla y porque permite estimar una puntuación numérica usando tres variables.

La ecuación ajustada, con coeficientes redondeados, es:

```text
Calidad estimada = 5,3629
                + 0,7203 × duración del sueño
                - 0,0218 × frecuencia cardíaca
                - 0,3067 × nivel de estrés
```

Manteniendo las demás variables constantes, una hora adicional de sueño se asocia con 0,72 puntos más de calidad estimada; un punto adicional de estrés, con 0,31 puntos menos. Son asociaciones del modelo, no efectos causales demostrados.

| Métrica en prueba | Resultado | Interpretación |
|---|---:|---|
| R² | 0,889 | Explica aproximadamente el 88,9 % de la variabilidad respecto a la referencia de la media en prueba. |
| MSE | 0,172 | Error cuadrático medio, en puntos al cuadrado. |
| RMSE | 0,415 | Error en puntos de calidad, con mayor penalización de errores grandes. |
| MAE | 0,334 | La predicción se desvía, en promedio absoluto, 0,334 puntos. |

El gráfico de valores reales frente a predichos permite observar cuánto se alejan las estimaciones de la línea de predicción perfecta. **R² no es un porcentaje de predicciones correctas.**

### Regresión logística: clasificar los registros

Se elige porque la etiqueta tiene dos clases. Utiliza ambas columnas de presión arterial, duración del sueño, calidad del sueño y edad, con `max_iter=1000` y el umbral predeterminado de clasificación.

| Métrica en prueba | Resultado |
|---|---:|
| Exactitud | 0,926 |
| Precisión de la clase 1 | 0,962 |
| Recall de la clase 1 | 0,909 |
| F1 de la clase 1 | 0,935 |

La exactitud equivale a **87 aciertos de 94 registros**. La precisión, recall y F1 anteriores corresponden a la clase **sin trastorno registrado**, según la codificación ejecutada.

La matriz, reconstruida a partir del reporte guardado y con los nombres corregidos, es:

| Real / Predicción | Con trastorno (0) | Sin trastorno registrado (1) |
|---|---:|---:|
| Con trastorno (0) | 37 | 2 |
| Sin trastorno registrado (1) | 5 | 50 |

El modelo identifica 37 de los 39 registros con trastorno y 50 de los 55 sin trastorno registrado. Los siete errores muestran que el desempeño alto no implica clasificación perfecta.

### Análisis estadístico complementario

Se ajusta un `Logit` de `statsmodels` sobre los **374 registros**. Calidad del sueño y edad presentan `p < 0,001`; las dos presiones y la duración del sueño presentan `p > 0,05` en ese ajuste.

Este análisis es exploratorio sobre la muestra completa. No sustituye la evaluación de prueba ni demuestra que una variable cause un trastorno.

## 6. Ajuste y mejora: estado actual

El notebook incorpora Random Forest y SHAP como análisis exploratorio, pero conserva los tres predictores de regresión. **No ejecuta una comparación de métricas antes y después de un ajuste**, búsqueda de hiperparámetros ni validación cruzada.

Para completar la etapa 7 de la guía, falta probar una alternativa y documentar su desempeño. Por ejemplo, comparar la regresión con tres variables frente a una con duración y estrés mediante validación cruzada sobre entrenamiento. El ranking SHAP calculado en prueba debe tratarse como explicación posterior; si se usa para seleccionar variables, esa partición deja de ser una prueba independiente y se necesita una evaluación final no utilizada en la selección.

## 7. Conclusiones y límites

- En el Random Forest exploratorio, duración del sueño y estrés tienen las mayores contribuciones medias según SHAP.
- Tres predictores permiten estimar la puntuación con un MAE de 0,334 puntos en la partición evaluada.
- La clasificación alcanza 87 aciertos de 94, pero sus etiquetas deben corregirse para comunicar adecuadamente los resultados.
- Una sola partición y 374 registros no permiten asegurar el mismo desempeño en otras poblaciones. No se ha demostrado ausencia de sobreajuste.
- Deben documentarse la licencia y las modificaciones del CSV respecto a la fuente. Los resultados son académicos; no validan un sistema de diagnóstico.
- Las salidas sugieren una posible inversión de los nombres de presión arterial: la columna llamada diastólica contiene valores como 126 y la llamada sistólica, 83. Debe contrastarse con el CSV original.

**Procedencia de las métricas y del ranking SHAP:** salidas guardadas en el archivo corregido `Pproyecto_Machine_Learning.ipynb`. No se ha repetido la ejecución completa porque el CSV utilizado no está incluido en los archivos disponibles.

## 8. Archivos del repositorio

| Archivo o carpeta | Contenido |
|---|---|
| `README.md` | Descripción, integrantes, datos, proceso, resultados y ejecución. |
| `Proyecto_Machine_Learning.ipynb` | Código y salidas del análisis. |
| `requirements.txt` | Dependencias de Python. |
| `datos/Sleep_health_and_lifestyle_dataset.csv` | CSV exacto utilizado; pendiente de incorporar. |

## 9. Cómo ejecutar

### Preparación del repositorio y corrección de importación

1. Reemplazar el notebook anterior por el archivo correcto recibido como `Pproyecto_Machine_Learning.ipynb`, guardándolo en GitHub con el nombre **`Proyecto_Machine_Learning.ipynb`** para conservar los enlaces de este README.
2. Reemplazar `Readme.md` por `README.md`, agregar `requirements.txt` y publicar el CSV exacto dentro de `datos/`.
3. Añadir en la celda de importaciones la línea que falta en el archivo recibido:

   ```python
   from sklearn.ensemble import RandomForestRegressor
   ```

   Sin esa importación, una sesión nueva fallará al crear el bosque, aunque el notebook tenga salidas guardadas. `RandomForestRegressor` pertenece a `scikit-learn`; no se instala como paquete separado.
4. Corregir los nombres del reporte y la matriz de clasificación a `['Con problemas de sueño (0)', 'Sin trastorno registrado (1)']`, conservando la codificación actual y comprobando el significado de los `NaN`.


### Google Colab: compatible con el código actual

1. Abrir el notebook actualizado con el enlace del inicio y guardar una copia en Drive. Subir `requirements.txt` al entorno de Colab y ejecutar `%pip install -r /content/requirements.txt` antes de importar las librerías.
2. Subir el **CSV exacto utilizado por el grupo** a la raíz de «Mi unidad», con el nombre `Sleep_health_and_lifestyle_dataset.csv`.
3. Ejecutar las celdas en orden y autorizar el montaje de Drive. El notebook lee:

   ```python
   df = pd.read_csv('/content/drive/MyDrive/Sleep_health_and_lifestyle_dataset.csv', delimiter=';')
   ```

4. Comprobar que la carga devuelve `(374, 14)`. Si aparece una sola columna, revisar el separador. Conservar los nombres de columnas, incluidos los espacios finales de presión arterial.
5. Ejecutar todo desde una sesión nueva y comprobar las tablas, los gráficos SHAP y las métricas de este README.

### Ejecución local con Jupyter

1. Clonar e instalar las dependencias, preferentemente en un entorno virtual:

   ```bash
   git clone https://github.com/renan873/proyecto-machine-learning.git
   cd proyecto-machine-learning
   python -m pip install -r requirements.txt
   python -m notebook
   ```

2. Guardar el CSV exacto en `datos/` y abrir `Proyecto_Machine_Learning.ipynb`.
3. Omitir las celdas de `google.colab` y el montaje de Drive. Reemplazar únicamente la lectura por:

   ```python
   df = pd.read_csv('datos/Sleep_health_and_lifestyle_dataset.csv', delimiter=';')
   ```

4. Ejecutar todas las demás celdas en orden.

`requirements.txt` incluye `shap` para las explicaciones, `ipython` para su inicialización visual y `notebook` para la ejecución local. Las importaciones de `google.colab` corresponden al entorno de Colab y se omiten en local.

Las versiones originales de las dependencias no quedaron registradas. `requirements.txt` declara las librerías necesarias, pero no garantiza resultados idénticos entre versiones.
