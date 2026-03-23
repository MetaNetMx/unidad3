# Unidad 3 - Modelos de Machine Learning Aplicados a Finanzas
## Elaboró: Ernesto Ramírez

---

Estimado Profesor,

Este repositorio contiene el desarrollo completo de la **Unidad 3**, donde se aplican 4 modelos de Machine Learning para predecir si un cliente dejará de pagar su tarjeta de crédito (default).

Se utilizó el dataset real **"Default of Credit Card Clients"** del repositorio UCI, con 30,000 registros de clientes de Taiwán.

---

## Contenido del Repositorio

### Dataset

| Archivo | Descripción |
|---------|-------------|
| `dataset_credit_card_default.csv` | Dataset con 30,000 registros y 24 columnas (23 variables predictoras + 1 variable objetivo). Incluye datos demográficos, historial de pagos y montos de facturación. |

### Scripts de Python (código ejecutable)

| Archivo | Qué hace | Modelos que usa |
|---------|----------|-----------------|
| `clasificadores_financieros.py` | Entrena y evalúa 2 clasificadores simples. Genera métricas, matrices de confusión, análisis de costos y gráficas. | **Regresión Logística** y **Árbol de Decisión** |
| `ensambles_financieros.py` | Entrena y evalúa 2 modelos de ensamble (más avanzados). Misma estructura de análisis que el anterior. | **Random Forest** y **Gradient Boosting (GBM)** |
| `cuadro_comparativo_modelos.py` | Ejecuta los 4 modelos juntos y genera una imagen comparativa visual pensada para presentar en junta directiva. | Los 4 modelos |

### Reportes escritos

| Archivo | Contenido |
|---------|-----------|
| `reporte_modelos_ml_finanzas.md` | Reporte detallado de Regresión Logística vs Árbol de Decisión: métricas, interpretación, análisis de costos y conclusiones. |
| `reporte_ensambles_finanzas.md` | Reporte detallado de Random Forest vs GBM: métricas, importancia de variables, análisis de costos y conclusiones. |

### Gráficas generadas

| Archivo | Contenido |
|---------|-----------|
| `graficas_comparacion.png` | Curvas ROC, matrices de confusión y barras de métricas para Regresión Logística vs Árbol de Decisión. |
| `graficas_ensambles.png` | Curvas ROC, matrices de confusión, barras de métricas e importancia de variables para Random Forest vs GBM. |
| `cuadro_comparativo_4_modelos.png` | Infografía ejecutiva comparando los 4 modelos lado a lado con tarjetas resumen, tabla, semáforo visual y conclusiones. |

---

## Los 4 Modelos Explicados de Forma Simple

| Modelo | Tipo | Cómo funciona (en simple) |
|--------|------|---------------------------|
| **Regresión Logística** | Clasificador simple | Calcula una probabilidad usando una fórmula matemática lineal. Es como una balanza que pesa cada variable para decidir si el cliente pagará o no. |
| **Árbol de Decisión** | Clasificador simple | Hace preguntas en secuencia (como un diagrama de flujo): "¿Pagó el mes pasado? ¿Su deuda es mayor a X?", hasta llegar a una decisión. |
| **Random Forest** | Ensamble (Bagging) | Crea 200 árboles de decisión diferentes y los pone a "votar". La decisión final es la que diga la mayoría. Reduce errores individuales. |
| **Gradient Boosting** | Ensamble (Boosting) | Entrena árboles uno tras otro, donde cada nuevo árbol corrige los errores del anterior. Es el más usado en la industria bancaria. |

---

## Cómo Ejecutar

Requisitos: Python 3 con las librerías `numpy`, `pandas`, `scikit-learn` y `matplotlib`.

```bash
# Instalar dependencias (si no las tiene)
pip install numpy pandas scikit-learn matplotlib

# Ejecutar clasificadores simples
python clasificadores_financieros.py

# Ejecutar modelos de ensamble
python ensambles_financieros.py

# Generar cuadro comparativo de los 4 modelos
python cuadro_comparativo_modelos.py
```

Cada script imprime resultados en consola y genera las gráficas automáticamente en la misma carpeta.

Funciona en **Windows, Mac y Linux** sin cambios.

---

## Estructura de Análisis (común en los 3 scripts)

1. Carga y descripción del dataset
2. Preparación de datos (70% entrenamiento / 30% prueba)
3. Entrenamiento de los modelos
4. Evaluación con métricas: Accuracy, Precision, Recall, ROC-AUC
5. Matrices de confusión
6. Análisis de costos de error en contexto financiero real
7. Generación de gráficas
8. Conclusiones

---

## Fuente del Dataset

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.

UCI Machine Learning Repository: Default of Credit Card Clients Dataset.
