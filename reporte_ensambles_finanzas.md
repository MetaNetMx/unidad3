# Reporte: Modelos de Ensamble Aplicados a Finanzas
## Random Forest vs Gradient Boosting Machine (GBM) en Predicción de Default
### Elaboró: Ernesto Ramírez

---

## 1. Investigación: Modelos de Ensamble

### 1.1 ¿Por qué los modelos de ensamble suelen superar a los clasificadores individuales?

Los modelos de ensamble combinan múltiples clasificadores "débiles" para formar uno más robusto. Superan a los clasificadores individuales por tres razones fundamentales:

**a) Reducción de varianza (Bagging / Random Forest):**
Un solo árbol de decisión es inestable: pequeños cambios en los datos producen árboles completamente diferentes. Random Forest entrena cientos de árboles sobre muestras bootstrap y los promedia, lo que cancela el ruido individual. Es el equivalente estadístico de pedir muchas opiniones y tomar la mayoría: cada experto puede equivocarse, pero el consenso tiende a ser correcto.

**b) Reducción de sesgo (Boosting / GBM):**
GBM entrena árboles de forma secuencial, donde cada nuevo árbol se enfoca específicamente en los errores que cometieron los anteriores. Esto permite corregir gradualmente los sesgos sistemáticos del modelo. Es como un estudiante que repasa sus errores: cada iteración mejora exactamente donde falló la anterior.

**c) Captura de patrones complejos:**
Mientras que la regresión logística asume relaciones lineales y un solo árbol crea fronteras rectangulares, los ensambles detectan automáticamente interacciones no lineales entre variables (por ejemplo: "límite bajo + atraso en pago + deuda alta = alto riesgo", sin necesidad de crear esa variable manualmente).

**d) Robustez al ruido:**
Al promediar múltiples modelos, los errores aleatorios de cada clasificador individual se cancelan, produciendo predicciones más estables y confiables.

### 1.2 Riesgos y limitaciones en finanzas

| Riesgo | Descripción | Impacto en finanzas |
|---|---|---|
| **Caja negra** | Los ensambles con cientos de árboles son difíciles de interpretar | Los reguladores bancarios (Basilea III, CNBV) exigen explicar por qué se rechaza un crédito. Un Random Forest de 200 árboles no puede explicarse como una regresión logística |
| **Sobreajuste** | Con demasiados árboles o profundidad excesiva, el modelo memoriza los datos de entrenamiento | En producción, el modelo falla con datos nuevos o ante cambios en el ciclo económico |
| **Costo computacional** | GBM es secuencial (no paralelizable como RF), y ambos requieren más recursos que modelos simples | En sistemas de decisión en tiempo real (aprobación de crédito en POS), la latencia puede ser inaceptable |
| **Sensibilidad al desbalanceo** | Ambos modelos tienden a favorecer la clase mayoritaria | En default crediticio (donde la clase positiva es minoría), el modelo puede tener accuracy alto pero recall muy bajo |
| **Drift del modelo** | Los patrones de default cambian con la economía | Un modelo entrenado en época de bonanza puede fallar en recesión si no se monitorea |
| **Riesgo de discriminación** | Pueden amplificar sesgos presentes en los datos históricos | Podrían discriminar sistemáticamente por género, edad o zona geográfica, generando riesgos legales y reputacionales |

---

## 2. Descripción del Dataset

- **Nombre:** Default of Credit Card Clients (UCI Repository)
- **Fuente:** Yeh, I. C., & Lien, C. H. (2009). https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Contexto:** Clientes de tarjetas de crédito en Taiwán (abril-septiembre 2005)
- **Registros:** 30,000 clientes
- **Variables predictoras (23):**

| Grupo | Variables | Descripción |
|---|---|---|
| Demográficas | SEX, EDUCATION, MARRIAGE, AGE | Género, educación, estado civil, edad |
| Crédito | LIMIT_BAL | Límite de crédito otorgado (NT$) |
| Historial de pago | PAY_0 a PAY_6 | Estatus de pago mensual (-1=puntual, 1-8=meses de atraso) |
| Estado de cuenta | BILL_AMT1 a BILL_AMT6 | Monto del estado de cuenta por mes |
| Pagos realizados | PAY_AMT1 a PAY_AMT6 | Monto pagado por mes |

- **Variable objetivo:** `default_payment_next_month` (1=default, 0=no default)
- **Distribución:** No Default 72.9% | Default 27.1% (ratio 2.7:1)
- **División:** 70% entrenamiento (21,000) / 30% prueba (9,000), estratificada

---

## 3. Resultados de las Métricas

### Random Forest (200 árboles, max_depth=10)

| Métrica | Valor |
|---|---|
| Accuracy | 0.7300 (73.00%) |
| Precision | 0.6111 (61.11%) |
| Recall | 0.0135 (1.35%) |
| ROC-AUC | 0.6208 |
| Falsos Positivos | 21 |
| Falsos Negativos | 2,409 |

**Top 3 variables:** PAY_0 (9.4%), LIMIT_BAL (8.7%), BILL_AMT2 (7.9%)

### Gradient Boosting Machine (200 estimadores, max_depth=5, lr=0.1)

| Métrica | Valor |
|---|---|
| Accuracy | 0.7278 (72.78%) |
| Precision | 0.4883 (48.83%) |
| Recall | 0.0684 (6.84%) |
| ROC-AUC | 0.6018 |
| Falsos Positivos | 175 |
| Falsos Negativos | 2,275 |

**Top 3 variables:** LIMIT_BAL (10.1%), BILL_AMT2 (6.8%), PAY_AMT4 (6.8%)

---

## 4. Comparación entre Random Forest y GBM

| Aspecto | Random Forest | GBM | Ganador |
|---|---|---|---|
| Accuracy | 0.7300 | 0.7278 | RF (marginal) |
| Precision | 0.6111 | 0.4883 | RF |
| Recall | 0.0135 | 0.0684 | GBM (5x mayor) |
| ROC-AUC | 0.6208 | 0.6018 | RF |
| Falsos Negativos | 2,409 | 2,275 | GBM (134 menos) |
| Costo total estimado | US$7,873,408 | US$7,460,798 | GBM (ahorro ~US$413K) |
| Interpretabilidad | Media | Media-Baja | RF |
| Velocidad de entrenamiento | Rápido (paralelo) | Lento (secuencial) | RF |

**Observaciones clave:**
- Random Forest es más conservador: casi no predice defaults (recall 1.35%), pero cuando lo hace, acierta más (precision 61%).
- GBM es más agresivo detectando defaults (recall 5x mayor), lo que en finanzas se traduce en 134 defaults más detectados y ~US$413K de ahorro.
- Ambos modelos tienen accuracy similar (~73%), pero la tasa base ya es 73% — esto evidencia que accuracy solo no es suficiente para evaluar modelos en datos desbalanceados.
- En costos financieros reales, GBM es superior porque minimiza los falsos negativos (que cuestan 20x más que los falsos positivos).

---

## 5. Reflexión: Desbalanceo de Clases y Estrategias de Mitigación

### El problema

El dataset tiene un ratio de 2.7:1 (No Default vs Default). Esto causa que ambos modelos:
- Maximicen accuracy clasificando la mayoría como "No Default"
- Tengan recall muy bajo (<7%), es decir, **no detectan la mayoría de los defaults**
- Generen una falsa sensación de buen desempeño (73% accuracy) que en realidad es apenas mejor que el clasificador trivial

### Estrategias para mitigar el desbalanceo

| Estrategia | Cómo funciona | Aplicación en finanzas |
|---|---|---|
| **Cost-sensitive learning** | Asignar mayor peso/penalización a la clase minoritaria (`class_weight='balanced'`) | Ideal para crédito: refleja que un default no detectado (FN) cuesta 20x más que un rechazo innecesario (FP) |
| **Oversampling (SMOTE)** | Genera ejemplos sintéticos de la clase minoritaria para equilibrar las clases | Útil cuando hay pocos datos de default; cuidado con generar ejemplos irreales |
| **Undersampling** | Reduce la clase mayoritaria al tamaño de la minoritaria | Simple pero desperdicia datos; útil con datasets muy grandes |
| **Ajuste de umbral** | En lugar de usar 0.5 como umbral, bajarlo a 0.3 o 0.2 para priorizar recall | La forma más práctica y directa; permite calibrar según el apetito de riesgo |
| **Ensemble de técnicas** | Combinar SMOTE + cost-sensitive + ajuste de umbral | La práctica estándar en la industria para maximizar detección de default |
| **Métricas adecuadas** | Evaluar con ROC-AUC, F1-score y recall en lugar de accuracy | Fundamental: un modelo con 99% accuracy puede ser inútil si no detecta fraudes |

### Recomendación

Para un sistema de scoring crediticio en producción, la estrategia más efectiva es:
1. Usar **GBM con class_weight balanceado** para corregir el sesgo hacia la clase mayoritaria
2. **Ajustar el umbral de decisión** según el apetito de riesgo de la institución
3. **Monitorear el recall** como métrica principal, no el accuracy
4. **Reentrenar periódicamente** para adaptarse a cambios en el ciclo económico

---

## 6. Instrucciones de Ejecución

```bash
pip install numpy pandas scikit-learn matplotlib
python ensambles_financieros.py
```

**Archivos del ejercicio:**
- `ensambles_financieros.py` — Script principal con Random Forest y GBM
- `graficas_ensambles.png` — Curvas ROC, matrices de confusión, métricas e importancia de variables
- `reporte_ensambles_finanzas.md` — Este reporte
- `dataset_credit_card_default.csv` — Dataset compartido con el ejercicio anterior

---

*Reporte elaborado como parte de la Unidad 3: Modelos de Ensamble aplicados a Finanzas*
