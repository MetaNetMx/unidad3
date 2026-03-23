# Reporte: Modelos de Machine Learning Aplicados a Finanzas
## Regresión Logística vs Árbol de Decisión en Predicción de Default de Crédito

---

## 1. Investigación Teórica

### 1.1 Regresión Logística

La regresión logística es un modelo estadístico que estima la probabilidad de un evento binario (por ejemplo, default vs. no default) mediante una función sigmoide aplicada a una combinación lineal de variables predictoras.

**Ventajas en finanzas:**
- **Interpretabilidad:** Los coeficientes tienen interpretación directa como log-odds, lo que permite explicar a reguladores y auditores por qué se rechazó o aprobó un crédito.
- **Probabilidades calibradas:** Genera probabilidades bien calibradas que pueden usarse directamente para scoring crediticio.
- **Robustez:** Es estable con datos razonablemente limpios y no tiende al sobreajuste fácilmente.
- **Cumplimiento regulatorio:** Es el estándar de la industria bancaria (Basilea II/III) precisamente por su transparencia.
- **Eficiencia computacional:** Rápido de entrenar y desplegar, incluso con millones de registros.

**Limitaciones en finanzas:**
- **Linealidad:** Asume relaciones lineales entre variables y el log-odds; no captura interacciones complejas automáticamente.
- **Feature engineering manual:** Requiere que el analista construya variables de interacción y transformaciones.
- **Sensibilidad a multicolinealidad:** Variables correlacionadas distorsionan los coeficientes.
- **No captura patrones no lineales:** En fraude, donde los patrones son complejos y cambiantes, puede quedarse corto.

### 1.2 Árbol de Decisión

El árbol de decisión particiona recursivamente el espacio de variables mediante reglas if-then hasta llegar a nodos terminales con una clasificación.

**Ventajas en finanzas:**
- **Captura no linealidades:** Detecta umbrales y relaciones complejas sin necesidad de transformaciones previas.
- **Interacciones automáticas:** Identifica combinaciones de variables que producen riesgo (e.g., ingreso bajo + deuda alta + historial malo).
- **Visualización intuitiva:** El árbol se puede presentar como un diagrama de flujo comprensible para no-técnicos.
- **No requiere escalado:** Funciona directamente con variables en diferentes escalas.
- **Detección de segmentos:** Útil para segmentación de clientes por perfil de riesgo.

**Limitaciones en finanzas:**
- **Sobreajuste (overfitting):** Sin poda adecuada, memoriza el ruido de los datos de entrenamiento.
- **Inestabilidad:** Pequeños cambios en los datos pueden generar árboles completamente diferentes.
- **Fronteras de decisión rectangulares:** Las particiones son ortogonales a los ejes, lo que puede ser subóptimo.
- **Sesgo hacia variables con muchas categorías:** Tiende a favorecer variables con más valores únicos.
- **Menor aceptación regulatoria:** Es menos aceptado por reguladores bancarios que la regresión logística en modelos de scoring.

### 1.3 Errores Críticos en Contexto Financiero

| Tipo de Error | Descripción | Impacto Financiero |
|---|---|---|
| **Falso Negativo** (No detectar default) | El modelo predice "no default" pero el cliente incumple | **MUY ALTO:** La institución pierde parcial o totalmente el capital prestado. En fraude, el fraude no se detecta. |
| **Falso Positivo** (Rechazar buen cliente) | El modelo predice "default" pero el cliente habría cumplido | **MODERADO:** Se pierde el ingreso por intereses y la relación comercial. |

**Conclusión sobre errores:** En la mayoría de contextos financieros, los **falsos negativos son más críticos**. Un préstamo incobrable puede representar pérdidas del 60-100% del capital, mientras que rechazar un buen cliente solo representa un costo de oportunidad del 5-15% en intereses no ganados. Por esto, en finanzas se prioriza el **recall** (sensibilidad) sobre la precision, y se ajustan los umbrales de decisión para minimizar falsos negativos, aun a costa de más falsos positivos.

Sin embargo, en mercados muy competitivos, un exceso de falsos positivos puede llevar a perder cuota de mercado. El balance óptimo depende del apetito de riesgo de la institución.

---

## 2. Descripción del Dataset

- **Tipo:** Dataset sintético que simula datos de default crediticio
- **Registros:** 2,000 solicitudes de crédito
- **Variables predictoras:**
  - `ingreso`: Ingreso anual del solicitante
  - `edad`: Edad del solicitante
  - `deuda`: Deuda total actual
  - `historial_crediticio`: Score crediticio (300-850)
  - `monto_prestamo`: Monto solicitado
  - `ratio_deuda_ingreso`: Proporción deuda/ingreso
- **Variable objetivo:** `default` (1 = incumplimiento, 0 = cumplimiento)
- **División:** 70% entrenamiento, 30% prueba (estratificada)

El dataset fue diseñado para reflejar relaciones financieras reales: mayor ratio deuda/ingreso incrementa la probabilidad de default, mientras que mejor historial crediticio y mayor ingreso la reducen.

---

## 3. Resultados de las Métricas

*(Los valores exactos se generan al ejecutar `clasificadores_financieros.py`)*

### Métricas evaluadas:

| Métrica | Significado en contexto financiero |
|---|---|
| **Accuracy** | Proporción total de predicciones correctas |
| **Precision** | De los marcados como default, cuántos realmente lo son |
| **Recall** | De los defaults reales, cuántos fueron detectados |
| **ROC-AUC** | Capacidad general de discriminación del modelo |

Las gráficas generadas incluyen:
- Curvas ROC comparativas de ambos modelos
- Matrices de confusión visuales para cada modelo

---

## 4. Comparación entre los Dos Modelos

| Aspecto | Regresión Logística | Árbol de Decisión |
|---|---|---|
| **Interpretabilidad** | Alta - coeficientes directos | Media - reglas visualizables |
| **Manejo de no linealidades** | Limitado | Bueno |
| **Riesgo de sobreajuste** | Bajo | Medio-Alto |
| **Requerimiento regulatorio** | Cumple estándares bancarios | Menos aceptado |
| **Velocidad de inferencia** | Muy rápida | Rápida |
| **Estabilidad** | Alta | Baja (sensible a datos) |
| **Probabilidades generadas** | Bien calibradas | Menos calibradas |

### Cuándo usar cada uno:

- **Regresión Logística:** Scoring crediticio en producción, modelos que necesitan aprobación regulatoria, cuando la interpretabilidad es prioritaria.
- **Árbol de Decisión:** Exploración inicial de datos, segmentación de clientes, detección de fraude donde los patrones son más complejos (preferiblemente como parte de un ensemble como Random Forest o XGBoost).

---

## 5. Reflexión sobre Costos de Error e Impacto en Decisiones Financieras

### El costo asimétrico de los errores

En finanzas, los errores no son iguales. Un **falso negativo** (aprobar un préstamo que terminará en default) puede costar entre el 60-100% del monto prestado. Un **falso positivo** (rechazar un buen cliente) solo cuesta el margen de interés perdido (5-15% del monto).

Esta asimetría tiene implicaciones directas:

1. **Ajuste de umbrales:** En lugar de usar el umbral estándar de 0.5, las instituciones financieras suelen usar umbrales más bajos (e.g., 0.3) para clasificar como default, priorizando la detección de morosos aunque esto signifique rechazar más buenos clientes.

2. **Métricas relevantes:** El accuracy por sí solo es engañoso en datasets desbalanceados (pocos defaults vs. muchos no-defaults). El **recall** y el **ROC-AUC** son métricas más informativas para evaluar modelos de riesgo crediticio.

3. **Impacto en cartera:** Un modelo con alto recall detecta más defaults potenciales, reduciendo las pérdidas por cartera vencida. Esto es especialmente crítico en:
   - Crisis económicas donde la tasa de default aumenta
   - Productos de alto monto (hipotecas, créditos empresariales)
   - Mercados con alta incertidumbre

4. **Consideraciones éticas:** Un modelo demasiado conservador (muchos falsos positivos) puede excluir sistemáticamente a poblaciones vulnerables del acceso al crédito, lo cual tiene implicaciones éticas y regulatorias.

### Recomendación práctica

Para un sistema de scoring crediticio en producción, la recomendación es:
- Usar **regresión logística** como modelo base por su interpretabilidad y aceptación regulatoria
- Complementar con **árboles de decisión** (o ensembles) para detectar patrones no lineales
- Optimizar el umbral de decisión considerando la **relación costo de falso negativo / costo de falso positivo**
- Monitorear continuamente el desempeño del modelo con datos nuevos (model monitoring)

---

## 6. Instrucciones de Ejecución

```bash
pip install numpy pandas scikit-learn matplotlib
python clasificadores_financieros.py
```

El script genera:
- Resultados completos en consola con todas las métricas
- Archivo `graficas_comparacion.png` con curvas ROC y matrices de confusión
- Análisis de costos de error financiero

---

*Reporte elaborado como parte de la Unidad 3: Modelos de Machine Learning aplicados a Finanzas*
