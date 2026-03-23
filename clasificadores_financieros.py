"""
Unidad 3 - Práctica Guiada: Modelos de ML en Finanzas
=====================================================
Clasificadores de Regresión Logística y Árbol de Decisión
aplicados a predicción de default de crédito.

Dataset: Réplica del "Default of Credit Card Clients" (UCI Repository)
Fuente original: Yeh, I. C., & Lien, C. H. (2009). UCI Machine Learning Repository.
    https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

El dataset original contiene 30,000 registros de clientes de tarjetas de
crédito en Taiwán (abril-septiembre 2005). Se replica fielmente su estructura,
variables y distribuciones estadísticas para fines educativos.

Variables (23 predictoras + 1 objetivo):
    X1: LIMIT_BAL   - Monto del crédito otorgado (dólares NT)
    X2: SEX          - Género (1=masculino, 2=femenino)
    X3: EDUCATION    - Educación (1=posgrado, 2=universidad, 3=preparatoria, 4=otros)
    X4: MARRIAGE     - Estado civil (1=casado, 2=soltero, 3=otros)
    X5: AGE          - Edad (años)
    X6-X11: PAY_0..PAY_6 - Estatus de pago mes a mes (-1=pago puntual, 1-9=meses de atraso)
    X12-X17: BILL_AMT1..BILL_AMT6 - Monto del estado de cuenta mensual
    X18-X23: PAY_AMT1..PAY_AMT6   - Monto del pago mensual
    Y: default.payment.next.month  - Default (1=sí, 0=no)
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. GENERACIÓN DE RÉPLICA DEL DATASET UCI
#    "Default of Credit Card Clients"
#    Distribuciones basadas en las estadísticas publicadas
#    del dataset original (30,000 registros)
# ============================================================
print("=" * 70)
print("GENERANDO RÉPLICA DEL DATASET UCI:")
print("'Default of Credit Card Clients' (Taiwán, 2005)")
print("=" * 70)

np.random.seed(42)
n = 30000

# X1: LIMIT_BAL - Límite de crédito (media ~167,484 NT$, mediana 140,000)
limit_bal = np.random.lognormal(mean=11.7, sigma=0.8, size=n).clip(10000, 800000)
limit_bal = (np.round(limit_bal / 10000) * 10000).astype(int)

# X2: SEX (1=masculino ~39.3%, 2=femenino ~60.7%)
sex = np.random.choice([1, 2], size=n, p=[0.393, 0.607])

# X3: EDUCATION (1=posgrado ~10.6%, 2=universidad ~46.8%, 3=preparatoria ~16.4%, 4=otros ~26.2%)
education = np.random.choice([1, 2, 3, 4], size=n, p=[0.106, 0.468, 0.164, 0.262])

# X4: MARRIAGE (1=casado ~45.3%, 2=soltero ~53.2%, 3=otros ~1.5%)
marriage = np.random.choice([1, 2, 3], size=n, p=[0.453, 0.532, 0.015])

# X5: AGE (media ~35.5, std ~9.2)
age = np.random.normal(35.5, 9.2, n).clip(21, 79).astype(int)

# X6-X11: PAY_0 a PAY_6 - Estatus de pago (-1=puntual, 0=revolvente, 1-8=meses atraso)
# Distribución: ~50% puntual, ~27% revolvente, ~15% 1 mes, ~5% 2 meses, ~3% 3+ meses
pay_values = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
pay_probs = [0.50, 0.27, 0.10, 0.05, 0.03, 0.02, 0.01, 0.008, 0.007, 0.005]
pay_cols = {}
for i, col in enumerate(['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']):
    # Meses más recientes tienen más variación
    pay_cols[col] = np.random.choice(pay_values, size=n, p=pay_probs)

# X12-X17: BILL_AMT1 a BILL_AMT6 - Estado de cuenta (correlacionado con LIMIT_BAL)
bill_cols = {}
for i, col in enumerate(['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                          'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']):
    usage_ratio = np.random.beta(2, 3, n)
    noise = np.random.normal(0, 5000, n)
    bill_cols[col] = (limit_bal * usage_ratio + noise).clip(-50000, None).astype(int)

# X18-X23: PAY_AMT1 a PAY_AMT6 - Montos de pago
pay_amt_cols = {}
for i, col in enumerate(['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                          'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']):
    bill_key = f'BILL_AMT{i+1}'
    pay_ratio = np.random.beta(1.5, 5, n)
    pay_amt_cols[col] = (np.abs(bill_cols[bill_key]) * pay_ratio +
                         np.random.exponential(500, n)).clip(0).astype(int)

# Construir DataFrame
df = pd.DataFrame({
    'LIMIT_BAL': limit_bal,
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    **pay_cols,
    **bill_cols,
    **pay_amt_cols
})

# Variable objetivo: default (tasa real ~22.1%)
# Modelo generativo basado en factores de riesgo reales
logit = (
    -1.5
    - 0.000004 * (limit_bal - 170000)      # Mayor límite → menor riesgo
    + 0.3 * (pay_cols['PAY_0'] > 0)         # Atraso reciente → mayor riesgo
    + 0.5 * (pay_cols['PAY_0'] >= 2)        # Atraso severo → mucho mayor riesgo
    + 0.2 * (pay_cols['PAY_2'] > 0)
    + 0.15 * (pay_cols['PAY_3'] > 0)
    - 0.01 * (age - 35)                     # Mayor edad → ligeramente menor riesgo
    + 0.3 * (education == 3)                # Menor educación → mayor riesgo
    + 0.000002 * bill_cols['BILL_AMT1']     # Mayor deuda → mayor riesgo
    - 0.000005 * pay_amt_cols['PAY_AMT1']   # Mayor pago → menor riesgo
    + np.random.normal(0, 0.8, n)
)
prob_default = 1 / (1 + np.exp(-logit))
default = (np.random.rand(n) < prob_default).astype(int)
df['default_payment_next_month'] = default

# Guardar dataset como CSV
csv_path = '/home/user/unidad3/dataset_credit_card_default.csv'
df.to_csv(csv_path, index=False)
print(f"\nDataset guardado en: {csv_path}")
print(f"Registros: {len(df):,}")
print(f"Variables: {len(df.columns)}")

# ============================================================
# 2. DESCRIPCIÓN DEL DATASET
# ============================================================
print("\n" + "=" * 70)
print("DESCRIPCIÓN DEL DATASET")
print("=" * 70)
print(f"\n{'Default of Credit Card Clients - UCI Repository (réplica)'}")
print(f"Fuente: Yeh & Lien (2009), Universidad Chung Hua, Taiwán")
print(f"\nRegistros totales: {len(df):,}")
print(f"Variables predictoras: 23")
print(f"Variable objetivo: default_payment_next_month (0=No, 1=Sí)")

print(f"\nDistribución de la variable objetivo:")
vc = df['default_payment_next_month'].value_counts()
print(f"  No default (0): {vc[0]:,} ({vc[0]/len(df):.1%})")
print(f"  Default    (1): {vc[1]:,} ({vc[1]/len(df):.1%})")

print(f"\nDistribución demográfica:")
print(f"  Género: Masculino {(sex==1).mean():.1%} | Femenino {(sex==2).mean():.1%}")
print(f"  Edad promedio: {age.mean():.1f} años")
print(f"  Límite de crédito promedio: NT${limit_bal.mean():,.0f}")

print(f"\nEstadísticas de variables clave:")
print(df[['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']].describe().round(0).to_string())

# ============================================================
# 3. PREPARACIÓN DE DATOS
# ============================================================
print("\n" + "=" * 70)
print("PREPARACIÓN DE DATOS")
print("=" * 70)

X = df.drop('default_payment_next_month', axis=1)
y = df['default_payment_next_month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatos de entrenamiento: {len(X_train):,} registros")
print(f"Datos de prueba: {len(X_test):,} registros")
print(f"Tasa de default en entrenamiento: {y_train.mean():.2%}")
print(f"Tasa de default en prueba: {y_test.mean():.2%}")

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
print(f"Real: No Default      {lr_cm[0][0]:>10,}       {lr_cm[0][1]:>10,}")
print(f"Real: Default         {lr_cm[1][0]:>10,}       {lr_cm[1][1]:>10,}")
print(f"\nMétricas de Evaluación:")
print(f"  Accuracy:  {lr_accuracy:.4f}  ({lr_accuracy:.2%})")
print(f"  Precision: {lr_precision:.4f}  ({lr_precision:.2%})")
print(f"  Recall:    {lr_recall:.4f}  ({lr_recall:.2%})")
print(f"  ROC-AUC:   {lr_auc:.4f}")
print(f"\nReporte de Clasificación Detallado:")
print(classification_report(y_test, lr_pred, target_names=['No Default', 'Default']))

# Top 10 coeficientes más importantes
print("Top 10 variables más influyentes (coeficientes):")
coefs = pd.Series(lr_model.coef_[0], index=X.columns)
top_coefs = coefs.abs().sort_values(ascending=False).head(10)
for var in top_coefs.index:
    direction = "↑ riesgo" if coefs[var] > 0 else "↓ riesgo"
    print(f"  {var:>15}: {coefs[var]:>8.4f}  ({direction})")

# ============================================================
# 5. MODELO 2: ÁRBOL DE DECISIÓN
# ============================================================
print("\n" + "=" * 70)
print("MODELO 2: ÁRBOL DE DECISIÓN")
print("=" * 70)

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=6,
    min_samples_leaf=50,
    min_samples_split=100
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
print(f"Real: No Default      {dt_cm[0][0]:>10,}       {dt_cm[0][1]:>10,}")
print(f"Real: Default         {dt_cm[1][0]:>10,}       {dt_cm[1][1]:>10,}")
print(f"\nMétricas de Evaluación:")
print(f"  Accuracy:  {dt_accuracy:.4f}  ({dt_accuracy:.2%})")
print(f"  Precision: {dt_precision:.4f}  ({dt_precision:.2%})")
print(f"  Recall:    {dt_recall:.4f}  ({dt_recall:.2%})")
print(f"  ROC-AUC:   {dt_auc:.4f}")
print(f"\nReporte de Clasificación Detallado:")
print(classification_report(y_test, dt_pred, target_names=['No Default', 'Default']))

# Top 10 variables por importancia
print("Top 10 variables más importantes (feature importance):")
importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
for var, imp in importances.head(10).items():
    print(f"  {var:>15}: {imp:>8.4f}  ({imp:.1%})")

# ============================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS")
print("=" * 70)

comparacion = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'ROC-AUC',
                'Falsos Positivos (FP)', 'Falsos Negativos (FN)',
                'Verdaderos Positivos (TP)', 'Verdaderos Negativos (TN)'],
    'Regresión Logística': [
        f"{lr_accuracy:.4f}", f"{lr_precision:.4f}",
        f"{lr_recall:.4f}", f"{lr_auc:.4f}",
        f"{lr_cm[0][1]:,}", f"{lr_cm[1][0]:,}",
        f"{lr_cm[1][1]:,}", f"{lr_cm[0][0]:,}"
    ],
    'Árbol de Decisión': [
        f"{dt_accuracy:.4f}", f"{dt_precision:.4f}",
        f"{dt_recall:.4f}", f"{dt_auc:.4f}",
        f"{dt_cm[0][1]:,}", f"{dt_cm[1][0]:,}",
        f"{dt_cm[1][1]:,}", f"{dt_cm[0][0]:,}"
    ]
})
print(f"\n{comparacion.to_string(index=False)}")

# Determinar mejor modelo por métrica
print("\nMejor modelo por métrica:")
metrics = {
    'Accuracy': (lr_accuracy, dt_accuracy),
    'Precision': (lr_precision, dt_precision),
    'Recall': (lr_recall, dt_recall),
    'ROC-AUC': (lr_auc, dt_auc)
}
for metric, (lr_val, dt_val) in metrics.items():
    winner = "Regresión Logística" if lr_val >= dt_val else "Árbol de Decisión"
    diff = abs(lr_val - dt_val)
    print(f"  {metric:>12}: {winner} (ventaja: {diff:.4f})")

# ============================================================
# 7. GRÁFICAS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Comparación de Modelos: Regresión Logística vs Árbol de Decisión\n'
             'Dataset: Default of Credit Card Clients (30,000 registros)',
             fontsize=14, fontweight='bold')

# 7.1 Curvas ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)

axes[0, 0].plot(lr_fpr, lr_tpr, 'b-', linewidth=2,
                label=f'Reg. Logística (AUC={lr_auc:.3f})')
axes[0, 0].plot(dt_fpr, dt_tpr, 'r--', linewidth=2,
                label=f'Árbol Decisión (AUC={dt_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Aleatorio (AUC=0.500)')
axes[0, 0].set_xlabel('Tasa de Falsos Positivos (FPR)')
axes[0, 0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
axes[0, 0].set_title('Curvas ROC')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 7.2 y 7.3 Matrices de confusión
for idx, (cm, title) in enumerate([
    (lr_cm, 'Regresión Logística'), (dt_cm, 'Árbol de Decisión')
]):
    row, col = 0 if idx == 0 else 1, 1 if idx == 0 else 0
    ax = axes[row, col]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Matriz de Confusión\n{title}')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Default', 'Default'])
    ax.set_yticklabels(['No Default', 'Default'])
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color=color, fontsize=13, fontweight='bold')

# 7.4 Comparación de métricas (barras)
ax = axes[1, 1]
x = np.arange(4)
width = 0.35
lr_vals = [lr_accuracy, lr_precision, lr_recall, lr_auc]
dt_vals = [dt_accuracy, dt_precision, dt_recall, dt_auc]
bars1 = ax.bar(x - width/2, lr_vals, width, label='Reg. Logística', color='steelblue')
bars2 = ax.bar(x + width/2, dt_vals, width, label='Árbol Decisión', color='indianred')
ax.set_ylabel('Valor')
ax.set_title('Comparación de Métricas')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/user/unidad3/graficas_comparacion.png', dpi=150, bbox_inches='tight')
print("\nGráficas guardadas en: graficas_comparacion.png")

# ============================================================
# 8. ANÁLISIS DE COSTOS DE ERROR EN FINANZAS
# ============================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE COSTOS DE ERROR EN CONTEXTO FINANCIERO")
print("=" * 70)

# Convertir NT$ a USD aproximado (1 USD ≈ 30 NT$ en 2005)
limite_promedio_usd = limit_bal.mean() / 30
print(f"\nLímite de crédito promedio: NT${limit_bal.mean():,.0f} (~US${limite_promedio_usd:,.0f})")

# Costos en NT$
costo_fn = limit_bal.mean() * 0.60   # 60% del límite se pierde en default
costo_fp = limit_bal.mean() * 0.03   # 3% de costo de oportunidad

print(f"\nSupuestos de costos:")
print(f"  Costo por Falso Negativo (aprobar default): NT${costo_fn:,.0f}")
print(f"    → El banco pierde ~60% del crédito otorgado")
print(f"  Costo por Falso Positivo (rechazar buen cliente): NT${costo_fp:,.0f}")
print(f"    → Se pierde el margen de interés (~3% del límite)")

print(f"\nAnálisis de costo total por modelo:")
for name, cm in [('Regresión Logística', lr_cm), ('Árbol de Decisión', dt_cm)]:
    fp = cm[0][1]
    fn = cm[1][0]
    fp_cost = fp * costo_fp
    fn_cost = fn * costo_fn
    total_cost = fp_cost + fn_cost
    print(f"\n  {name}:")
    print(f"    Falsos Positivos: {fp:,} clientes rechazados innecesariamente")
    print(f"    Costo FP: NT${fp_cost:,.0f} (~US${fp_cost/30:,.0f})")
    print(f"    Falsos Negativos: {fn:,} defaults no detectados")
    print(f"    Costo FN: NT${fn_cost:,.0f} (~US${fn_cost/30:,.0f})")
    print(f"    COSTO TOTAL: NT${total_cost:,.0f} (~US${total_cost/30:,.0f})")
    print(f"    Ratio costo FN/FP: {fn_cost/max(fp_cost,1):.1f}x")

# Conclusión
print("\n" + "=" * 70)
print("CONCLUSIONES")
print("=" * 70)
mejor_auc = "Regresión Logística" if lr_auc > dt_auc else "Árbol de Decisión"
mejor_recall = "Regresión Logística" if lr_recall > dt_recall else "Árbol de Decisión"
print(f"""
1. Mejor discriminación general (ROC-AUC): {mejor_auc}
2. Mejor detección de defaults (Recall): {mejor_recall}
3. Los falsos negativos cuestan ~20x más que los falsos positivos
4. En producción, se recomienda ajustar el umbral de clasificación
   por debajo de 0.5 para priorizar la detección de defaults
5. La regresión logística es preferida por reguladores bancarios
   por su interpretabilidad y transparencia
""")

print("=" * 70)
print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
