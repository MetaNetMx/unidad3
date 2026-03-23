"""
Unidad 3 - Práctica Guiada: Modelos de ML en Finanzas
=====================================================
Clasificadores de Regresión Logística y Árbol de Decisión
aplicados a predicción de default de crédito.

Dataset: Generado sintéticamente simulando datos reales de default crediticio.
Variables: ingreso, deuda, historial_crediticio, edad, monto_prestamo, ratio_deuda_ingreso, default.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. GENERACIÓN DEL DATASET SINTÉTICO DE DEFAULT DE CRÉDITO
# ============================================================
np.random.seed(42)
n = 2000

ingreso = np.random.normal(50000, 20000, n).clip(10000)
edad = np.random.normal(40, 12, n).clip(18, 70).astype(int)
deuda = np.random.exponential(15000, n).clip(0)
historial_crediticio = np.random.randint(300, 850, n)
monto_prestamo = np.random.exponential(20000, n).clip(1000)
ratio_deuda_ingreso = deuda / ingreso

# Probabilidad de default basada en factores financieros reales
logit = (
    -2.0
    + 1.5 * (ratio_deuda_ingreso - 0.3)
    - 0.00002 * (ingreso - 50000)
    - 0.003 * (historial_crediticio - 600)
    + 0.00001 * (monto_prestamo - 20000)
    - 0.02 * (edad - 35)
    + np.random.normal(0, 0.5, n)
)
prob_default = 1 / (1 + np.exp(-logit))
default = (np.random.rand(n) < prob_default).astype(int)

df = pd.DataFrame({
    'ingreso': ingreso,
    'edad': edad,
    'deuda': deuda,
    'historial_crediticio': historial_crediticio,
    'monto_prestamo': monto_prestamo,
    'ratio_deuda_ingreso': ratio_deuda_ingreso,
    'default': default
})

print("=" * 70)
print("ANÁLISIS DE MODELOS DE ML APLICADOS A FINANZAS")
print("Dataset: Predicción de Default de Crédito")
print("=" * 70)

# ============================================================
# 2. DESCRIPCIÓN DEL DATASET
# ============================================================
print("\n" + "=" * 70)
print("DESCRIPCIÓN DEL DATASET")
print("=" * 70)
print(f"\nRegistros totales: {len(df)}")
print(f"Variables predictoras: {df.columns.tolist()[:-1]}")
print(f"Variable objetivo: default (0=No default, 1=Default)")
print(f"\nDistribución de la variable objetivo:")
print(df['default'].value_counts().to_string())
print(f"\nTasa de default: {df['default'].mean():.2%}")
print(f"\nEstadísticas descriptivas:")
print(df.describe().round(2).to_string())

# ============================================================
# 3. PREPARACIÓN DE DATOS
# ============================================================
X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatos de entrenamiento: {len(X_train)} registros")
print(f"Datos de prueba: {len(X_test)} registros")

# ============================================================
# 4. MODELO 1: REGRESIÓN LOGÍSTICA
# ============================================================
print("\n" + "=" * 70)
print("MODELO 1: REGRESIÓN LOGÍSTICA")
print("=" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_cm = confusion_matrix(y_test, lr_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"\nMatriz de Confusión:")
print(f"                  Predicho: No Default  Predicho: Default")
print(f"Real: No Default      {lr_cm[0][0]:>10}       {lr_cm[0][1]:>10}")
print(f"Real: Default         {lr_cm[1][0]:>10}       {lr_cm[1][1]:>10}")
print(f"\nMétricas:")
print(f"  Accuracy:  {lr_accuracy:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")
print(f"  ROC-AUC:   {lr_auc:.4f}")
print(f"\nReporte de Clasificación:")
print(classification_report(y_test, lr_pred, target_names=['No Default', 'Default']))

# Coeficientes del modelo
print("Coeficientes del modelo (importancia de variables):")
coefs = pd.Series(lr_model.coef_[0], index=X.columns).sort_values(ascending=False)
for var, coef in coefs.items():
    print(f"  {var:>25}: {coef:>8.4f}")

# ============================================================
# 5. MODELO 2: ÁRBOL DE DECISIÓN
# ============================================================
print("\n" + "=" * 70)
print("MODELO 2: ÁRBOL DE DECISIÓN")
print("=" * 70)

dt_model = DecisionTreeClassifier(
    random_state=42, max_depth=5, min_samples_leaf=20
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

dt_cm = confusion_matrix(y_test, dt_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_prob)

print(f"\nMatriz de Confusión:")
print(f"                  Predicho: No Default  Predicho: Default")
print(f"Real: No Default      {dt_cm[0][0]:>10}       {dt_cm[0][1]:>10}")
print(f"Real: Default         {dt_cm[1][0]:>10}       {dt_cm[1][1]:>10}")
print(f"\nMétricas:")
print(f"  Accuracy:  {dt_accuracy:.4f}")
print(f"  Precision: {dt_precision:.4f}")
print(f"  Recall:    {dt_recall:.4f}")
print(f"  ROC-AUC:   {dt_auc:.4f}")
print(f"\nReporte de Clasificación:")
print(classification_report(y_test, dt_pred, target_names=['No Default', 'Default']))

# Importancia de variables
print("Importancia de variables (feature importance):")
importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
for var, imp in importances.items():
    print(f"  {var:>25}: {imp:>8.4f}")

# ============================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS")
print("=" * 70)

comparacion = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'ROC-AUC',
                'Falsos Positivos', 'Falsos Negativos'],
    'Regresión Logística': [
        f"{lr_accuracy:.4f}", f"{lr_precision:.4f}",
        f"{lr_recall:.4f}", f"{lr_auc:.4f}",
        str(lr_cm[0][1]), str(lr_cm[1][0])
    ],
    'Árbol de Decisión': [
        f"{dt_accuracy:.4f}", f"{dt_precision:.4f}",
        f"{dt_recall:.4f}", f"{dt_auc:.4f}",
        str(dt_cm[0][1]), str(dt_cm[1][0])
    ]
})
print(f"\n{comparacion.to_string(index=False)}")

# ============================================================
# 7. GRÁFICAS
# ============================================================

# Curvas ROC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)

axes[0].plot(lr_fpr, lr_tpr, 'b-', label=f'Reg. Logística (AUC={lr_auc:.3f})')
axes[0].plot(dt_fpr, dt_tpr, 'r--', label=f'Árbol Decisión (AUC={dt_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.5)
axes[0].set_xlabel('Tasa de Falsos Positivos')
axes[0].set_ylabel('Tasa de Verdaderos Positivos')
axes[0].set_title('Curvas ROC - Comparación')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Matrices de confusión
for idx, (cm, title) in enumerate([
    (lr_cm, 'Regresión Logística'), (dt_cm, 'Árbol de Decisión')
]):
    ax = axes[idx + 1]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Matriz de Confusión\n{title}')
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=14)

plt.tight_layout()
plt.savefig('/home/user/unidad3/graficas_comparacion.png', dpi=150, bbox_inches='tight')
print("\nGráficas guardadas en: graficas_comparacion.png")

# ============================================================
# 8. ANÁLISIS DE COSTOS DE ERROR EN FINANZAS
# ============================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE COSTOS DE ERROR EN CONTEXTO FINANCIERO")
print("=" * 70)

monto_promedio = df['monto_prestamo'].mean()
print(f"\nMonto promedio de préstamo: ${monto_promedio:,.0f}")

# Costos estimados de cada tipo de error
costo_fn = monto_promedio * 0.60  # Pérdida parcial del préstamo
costo_fp = monto_promedio * 0.05  # Costo de oportunidad

print(f"\nCosto estimado por Falso Negativo (no detectar default): ${costo_fn:,.0f}")
print(f"  → Se aprueba un préstamo que terminará en default")
print(f"Costo estimado por Falso Positivo (rechazar buen cliente): ${costo_fp:,.0f}")
print(f"  → Se pierde el ingreso por intereses de un cliente solvente")

for name, cm in [('Regresión Logística', lr_cm), ('Árbol de Decisión', dt_cm)]:
    fp_cost = cm[0][1] * costo_fp
    fn_cost = cm[1][0] * costo_fn
    total_cost = fp_cost + fn_cost
    print(f"\n{name}:")
    print(f"  Costo por Falsos Positivos ({cm[0][1]} casos): ${fp_cost:,.0f}")
    print(f"  Costo por Falsos Negativos ({cm[1][0]} casos): ${fn_cost:,.0f}")
    print(f"  Costo Total Estimado:                         ${total_cost:,.0f}")

print("\n" + "=" * 70)
print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
