# Predicción de la calidad del sueño 

Proyecto integrador de **Aprendizaje Automático y Minería de Datos**, Maestría en Gestión y Analítica de Datos.  
**Docente:** Adriana Collaguazo Jaramillo, Mg.

**Integrantes:** Renan Rivadeneira, María Sol Tuárez, Diego Niquinga y Gabriel Estupiñán.

[Ver notebook](Proyecto_Machine_Learning.ipynb) · [Abrir en Google Colab](https://colab.research.google.com/github/renan873/proyecto-machine-learning/blob/main/Proyecto_Machine_Learning.ipynb)

## 1. Problema y objetivo

¿Es posible estimar la calidad del sueño a partir de su duración, la frecuencia cardíaca y el nivel de estrés?

El objetivo principal es predecir `Quality of Sleep` mediante **regresión lineal múltiple**. Como análisis complementario, se utiliza **regresión logística** para distinguir registros con trastorno del sueño de registros sin trastorno registrado.

Son dos tareas diferentes: una predice una puntuación y la otra una categoría. Sus métricas no permiten decidir cuál modelo es mejor que el otro.

## 2. Dataset y fuente

**Archivo utilizado:** `Sleep_health_and_lifestyle_dataset.csv`, separado por punto y coma (`;`). La ejecución guardada contiene **374 registros y 14 columnas** antes de crear la variable binaria.

**Referencia pública del dataset:** [Sleep Health and Lifestyle Dataset — Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).

**Trazabilidad pendiente:** el nombre y las variables coinciden con ese dataset, pero debe confirmarse que fue la fuente de descarga del grupo y registrar su licencia. La versión utilizada tiene la presión arterial separada en dos columnas; para reproducir exactamente los resultados debe publicarse el CSV utilizado, no sustituirlo directamente por otra versión.

El DOI incluido en el notebook, [10.1186/s12888-026-08081-2](https://doi.org/10.1186/s12888-026-08081-2), corresponde a un estudio con 596 pacientes mayores con dolor crónico; no documenta la procedencia de esta tabla de 374 registros.

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

El notebook contiene una configuración por modelo. **Todavía no ejecuta comparación de hiperparámetros, validación cruzada ni una segunda alternativa para la misma tarea.** Random Forest se menciona en la introducción, pero no está implementado.

Para completar la etapa 7 de la guía, falta realizar un ajuste y documentar su resultado: por ejemplo, comparar conjuntos de variables mediante validación cruzada sobre entrenamiento y evaluar la alternativa seleccionada en prueba. No se reportan mejoras que aún no se han ejecutado.

## 7. Conclusiones y límites

- Tres predictores permiten estimar la puntuación con un MAE de 0,334 puntos en la partición evaluada.
- La clasificación alcanza 87 aciertos de 94, pero sus etiquetas deben corregirse para comunicar adecuadamente los resultados.
- Una sola partición y 374 registros no permiten asegurar el mismo desempeño en otras poblaciones. No se ha demostrado ausencia de sobreajuste.
- Deben confirmarse procedencia, licencia y naturaleza sintética de los datos. Los resultados son académicos; no validan un sistema de diagnóstico.
- Las salidas sugieren una posible inversión de los nombres de presión arterial: la columna llamada diastólica contiene valores como 126 y la llamada sistólica, 83. Debe contrastarse con el CSV original.

**Procedencia de las métricas:** salidas guardadas en el notebook adjunto. No se ha repetido la ejecución completa porque el CSV utilizado no está incluido en los archivos disponibles.

## 8. Archivos del repositorio

| Archivo o carpeta | Contenido |
|---|---|
| `README.md` | Descripción, integrantes, datos, proceso, resultados y ejecución. |
| `Proyecto_Machine_Learning.ipynb` | Código y salidas del análisis. |
| `requirements.txt` | Dependencias de Python. |
| `datos/Sleep_health_and_lifestyle_dataset.csv` | CSV exacto utilizado; pendiente de incorporar. |

## 9. Cómo ejecutar

### Google Colab: compatible con el código actual

1. Abrir el notebook con el enlace del inicio y guardar una copia en Drive.
2. Subir el **CSV exacto utilizado por el grupo** a la raíz de «Mi unidad», con el nombre `Sleep_health_and_lifestyle_dataset.csv`.
3. Ejecutar las celdas en orden y autorizar el montaje de Drive. El notebook lee:

   ```python
   df = pd.read_csv('/content/drive/MyDrive/Sleep_health_and_lifestyle_dataset.csv', delimiter=';')
   ```

4. Comprobar que la carga devuelve `(374, 14)`. Si aparece una sola columna, revisar el separador. Conservar los nombres de columnas, incluidos los espacios finales de presión arterial.
5. Ejecutar todo y comprobar las tablas y métricas de este README. Si falta alguna dependencia, subir `requirements.txt` a Colab y ejecutar `%pip install -r /content/requirements.txt`.

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

Las versiones originales de las dependencias no quedaron registradas. `requirements.txt` declara las librerías necesarias, pero no garantiza resultados idénticos entre versiones.
