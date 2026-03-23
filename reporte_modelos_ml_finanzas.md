# Reporte: Modelos de Machine Learning Aplicados a Finanzas
## Regresión Logística vs Árbol de Decisión en Predicción de Default de Crédito

---

## 1. Investigación Teórica

### 1.1 Regresión Logística

La regresión logística es un modelo estadístico que estima la probabilidad de un evento binario (default vs. no default) mediante una función sigmoide aplicada a una combinación lineal de variables predictoras.

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
- **Interacciones automáticas:** Identifica combinaciones de variables que producen riesgo (e.g., límite bajo + atraso en pagos + deuda alta).
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

**Conclusión sobre errores:** En la mayoría de contextos financieros, los **falsos negativos son más críticos**. Un préstamo incobrable puede representar pérdidas del 60-100% del capital, mientras que rechazar un buen cliente solo representa un costo de oportunidad del 3-15% en intereses no ganados. Por esto, en finanzas se prioriza el **recall** (sensibilidad) sobre la precision, y se ajustan los umbrales de decisión para minimizar falsos negativos, aun a costa de más falsos positivos.

---

## 2. Descripción del Dataset

- **Nombre:** Default of Credit Card Clients (réplica fiel del dataset UCI)
- **Fuente original:** Yeh, I. C., & Lien, C. H. (2009). UCI Machine Learning Repository
- **Contexto:** Clientes de tarjetas de crédito en Taiwán (abril-septiembre 2005)
- **Registros:** 30,000 clientes
- **Variables predictoras (23):**

| Variable | Descripción |
|---|---|
| `LIMIT_BAL` | Límite de crédito otorgado (dólares NT) |
| `SEX` | Género (1=masculino, 2=femenino) |
| `EDUCATION` | Nivel educativo (1=posgrado, 2=universidad, 3=preparatoria, 4=otros) |
| `MARRIAGE` | Estado civil (1=casado, 2=soltero, 3=otros) |
| `AGE` | Edad en años |
| `PAY_0..PAY_6` | Estatus de pago mensual (-1=puntual, 0=revolvente, 1-8=meses de atraso) |
| `BILL_AMT1..BILL_AMT6` | Monto del estado de cuenta mensual |
| `PAY_AMT1..PAY_AMT6` | Monto del pago mensual |

- **Variable objetivo:** `default_payment_next_month` (1=default, 0=no default)
- **Tasa de default:** 27.1%
- **División:** 70% entrenamiento (21,000) / 30% prueba (9,000), estratificada

---

## 3. Resultados de las Métricas

### Regresión Logística

| Métrica | Valor |
|---|---|
| Accuracy | 0.7273 (72.73%) |
| Precision | 0.4412 (44.12%) |
| Recall | 0.0184 (1.84%) |
| ROC-AUC | 0.6273 |
| Falsos Positivos | 57 |
| Falsos Negativos | 2,397 |

**Variables más influyentes:** LIMIT_BAL (↓ riesgo), PAY_0 (↑ riesgo), BILL_AMT1 (↑ riesgo)

### Árbol de Decisión

| Métrica | Valor |
|---|---|
| Accuracy | 0.7280 (72.80%) |
| Precision | 0.4899 (48.99%) |
| Recall | 0.0598 (5.98%) |
| ROC-AUC | 0.6068 |
| Falsos Positivos | 152 |
| Falsos Negativos | 2,296 |

**Variables más importantes:** LIMIT_BAL (46.9%), PAY_0 (25.4%), BILL_AMT4 (5.0%)

---

## 4. Comparación entre los Dos Modelos

| Aspecto | Regresión Logística | Árbol de Decisión | Ganador |
|---|---|---|---|
| Accuracy | 0.7273 | 0.7280 | Árbol (marginal) |
| Precision | 0.4412 | 0.4899 | Árbol |
| Recall | 0.0184 | 0.0598 | Árbol |
| ROC-AUC | 0.6273 | 0.6068 | Reg. Logística |
| Falsos Negativos | 2,397 | 2,296 | Árbol (101 menos) |
| Interpretabilidad | Alta | Media | Reg. Logística |
| Aceptación regulatoria | Alta | Baja | Reg. Logística |

**Observaciones clave:**
- Ambos modelos tienen accuracy similar (~73%), pero esto es engañoso: clasificar a todos como "no default" daría 73% de accuracy.
- El recall es muy bajo en ambos modelos (~2-6%), lo cual es preocupante para el contexto financiero.
- La regresión logística tiene mejor capacidad de discriminación general (AUC más alto).
- El árbol de decisión detecta más defaults (recall 3x mayor) pero con más falsos positivos.
- Ambos modelos necesitarían optimización del umbral de decisión para ser útiles en producción.

---

## 5. Reflexión sobre Costos de Error e Impacto en Decisiones Financieras

### Asimetría de costos

Con un límite de crédito promedio de NT$163,345 (~US$5,445):

| Tipo de Error | Costo unitario | Explicación |
|---|---|---|
| Falso Negativo | NT$98,007 (~US$3,267) | 60% del crédito se pierde en default |
| Falso Positivo | NT$4,900 (~US$163) | 3% del límite en intereses perdidos |

**Un falso negativo cuesta 20 veces más que un falso positivo.**

### Costo total por modelo

| Modelo | Costo FP | Costo FN | Costo Total |
|---|---|---|---|
| Regresión Logística | US$9,311 | US$7,830,775 | **US$7,840,086** |
| Árbol de Decisión | US$24,828 | US$7,500,818 | **US$7,525,646** |

El árbol de decisión genera un costo total menor (~US$314,440 de ahorro) porque detecta 101 defaults más, a pesar de rechazar 95 buenos clientes adicionales.

### Implicaciones para la toma de decisiones

1. **El accuracy no basta:** Un modelo con 73% de accuracy puede parecer aceptable, pero si no detecta defaults (recall <6%), las pérdidas son enormes.

2. **Ajuste de umbral:** Bajando el umbral de clasificación de 0.5 a 0.3 o menos, se incrementaría el recall significativamente, detectando más defaults a costa de más falsos positivos — un intercambio económicamente favorable.

3. **Modelos en producción:** Para un sistema de scoring crediticio real:
   - Usar regresión logística como modelo base (transparencia regulatoria)
   - Complementar con ensembles (Random Forest, XGBoost) para mayor poder predictivo
   - Calibrar umbrales según el apetito de riesgo de la institución

4. **Consideraciones éticas:** Un modelo excesivamente conservador puede excluir sistemáticamente a poblaciones de menores ingresos del acceso al crédito, con implicaciones sociales y regulatorias.

5. **Monitoreo continuo:** Los patrones de default cambian con el ciclo económico. Un modelo debe reentrenarse periódicamente y monitorearse con métricas de estabilidad poblacional.

---

## 6. Instrucciones de Ejecución

```bash
pip install numpy pandas scikit-learn matplotlib
python clasificadores_financieros.py
```

**Archivos generados:**
- `dataset_credit_card_default.csv` — Dataset de 30,000 registros
- `graficas_comparacion.png` — Curvas ROC, matrices de confusión y comparación de métricas
- Resultados completos en consola

---

*Reporte elaborado como parte de la Unidad 3: Modelos de Machine Learning aplicados a Finanzas*
*Dataset basado en: Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." Expert Systems with Applications, 36(2), 2473-2480.*
