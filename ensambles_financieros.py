"""
Unidad 3 - Modelos de Ensamble en Finanzas
===========================================
Elaboró: Ernesto Ramírez

Clasificadores Random Forest y Gradient Boosting Machine (GBM)
aplicados a predicción de default de crédito.

Dataset: "Default of Credit Card Clients" (UCI Repository)
Fuente: Yeh, I. C., & Lien, C. H. (2009). UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

30,000 registros de clientes de tarjetas de crédito en Taiwán (2005).
23 variables predictoras + 1 variable objetivo (default_payment_next_month).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_auc_score, roc_curve, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. CARGA DEL DATASET
# ============================================================
print("=" * 70)
print("MODELOS DE ENSAMBLE APLICADOS A FINANZAS")
print("Random Forest vs Gradient Boosting Machine (GBM)")
print("=" * 70)

csv_path = os.path.join(BASE_DIR, 'dataset_credit_card_default.csv')
df = pd.read_csv(csv_path)
print(f"\nDataset cargado desde: {csv_path}")
print(f"Registros: {len(df):,}")
print(f"Variables: {len(df.columns)}")

# ============================================================
# 2. DESCRIPCIÓN DEL DATASET
# ============================================================
print("\n" + "=" * 70)
print("DESCRIPCIÓN DEL DATASET")
print("=" * 70)
print(f"\nDefault of Credit Card Clients - UCI Repository")
print(f"Fuente: Yeh & Lien (2009), Universidad Chung Hua, Taiwán")
print(f"\nRegistros totales: {len(df):,}")
print(f"Variables predictoras: 23")
print(f"Variable objetivo: default_payment_next_month (0=No, 1=Sí)")

vc = df['default_payment_next_month'].value_counts()
print(f"\nDistribución de la variable objetivo:")
print(f"  No default (0): {vc[0]:,} ({vc[0]/len(df):.1%})")
print(f"  Default    (1): {vc[1]:,} ({vc[1]/len(df):.1%})")
ratio_desbalanceo = vc[0] / vc[1]
print(f"  Ratio de desbalanceo: {ratio_desbalanceo:.1f}:1")

print(f"\nDistribución demográfica:")
print(f"  Género: Masculino {(df['SEX']==1).mean():.1%} | Femenino {(df['SEX']==2).mean():.1%}")
print(f"  Edad promedio: {df['AGE'].mean():.1f} años")
print(f"  Límite de crédito promedio: NT${df['LIMIT_BAL'].mean():,.0f}")

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

print(f"\nDatos de entrenamiento: {len(X_train):,} registros")
print(f"Datos de prueba: {len(X_test):,} registros")
print(f"Tasa de default en entrenamiento: {y_train.mean():.2%}")
print(f"Tasa de default en prueba: {y_test.mean():.2%}")

# ============================================================
# 4. MODELO 1: RANDOM FOREST
# ============================================================
print("\n" + "=" * 70)
print("MODELO 1: RANDOM FOREST")
print("=" * 70)
print("\nEntrenando Random Forest (200 árboles, max_depth=10)...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_cm = confusion_matrix(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"\nMatriz de Confusión:")
print(f"                  Predicho: No Default  Predicho: Default")
print(f"Real: No Default      {rf_cm[0][0]:>10,}       {rf_cm[0][1]:>10,}")
print(f"Real: Default         {rf_cm[1][0]:>10,}       {rf_cm[1][1]:>10,}")
print(f"\nMétricas de Evaluación:")
print(f"  Accuracy:  {rf_accuracy:.4f}  ({rf_accuracy:.2%})")
print(f"  Precision: {rf_precision:.4f}  ({rf_precision:.2%})")
print(f"  Recall:    {rf_recall:.4f}  ({rf_recall:.2%})")
print(f"  ROC-AUC:   {rf_auc:.4f}")
print(f"\nReporte de Clasificación Detallado:")
print(classification_report(y_test, rf_pred, target_names=['No Default', 'Default']))

print("Top 10 variables más importantes (feature importance):")
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
for var, imp in rf_importances.head(10).items():
    print(f"  {var:>15}: {imp:>8.4f}  ({imp:.1%})")

# ============================================================
# 5. MODELO 2: GRADIENT BOOSTING MACHINE (GBM)
# ============================================================
print("\n" + "=" * 70)
print("MODELO 2: GRADIENT BOOSTING MACHINE (GBM)")
print("=" * 70)
print("\nEntrenando GBM (200 estimadores, max_depth=5, lr=0.1)...")

gbm_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42
)
gbm_model.fit(X_train, y_train)
gbm_pred = gbm_model.predict(X_test)
gbm_prob = gbm_model.predict_proba(X_test)[:, 1]

gbm_cm = confusion_matrix(y_test, gbm_pred)
gbm_accuracy = accuracy_score(y_test, gbm_pred)
gbm_precision = precision_score(y_test, gbm_pred)
gbm_recall = recall_score(y_test, gbm_pred)
gbm_auc = roc_auc_score(y_test, gbm_prob)

print(f"\nMatriz de Confusión:")
print(f"                  Predicho: No Default  Predicho: Default")
print(f"Real: No Default      {gbm_cm[0][0]:>10,}       {gbm_cm[0][1]:>10,}")
print(f"Real: Default         {gbm_cm[1][0]:>10,}       {gbm_cm[1][1]:>10,}")
print(f"\nMétricas de Evaluación:")
print(f"  Accuracy:  {gbm_accuracy:.4f}  ({gbm_accuracy:.2%})")
print(f"  Precision: {gbm_precision:.4f}  ({gbm_precision:.2%})")
print(f"  Recall:    {gbm_recall:.4f}  ({gbm_recall:.2%})")
print(f"  ROC-AUC:   {gbm_auc:.4f}")
print(f"\nReporte de Clasificación Detallado:")
print(classification_report(y_test, gbm_pred, target_names=['No Default', 'Default']))

print("Top 10 variables más importantes (feature importance):")
gbm_importances = pd.Series(gbm_model.feature_importances_, index=X.columns).sort_values(ascending=False)
for var, imp in gbm_importances.head(10).items():
    print(f"  {var:>15}: {imp:>8.4f}  ({imp:.1%})")

# ============================================================
# 6. COMPARACIÓN DE MODELOS
# ============================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS: RANDOM FOREST vs GBM")
print("=" * 70)

comparacion = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'ROC-AUC',
                'Falsos Positivos (FP)', 'Falsos Negativos (FN)',
                'Verdaderos Positivos (TP)', 'Verdaderos Negativos (TN)'],
    'Random Forest': [
        f"{rf_accuracy:.4f}", f"{rf_precision:.4f}",
        f"{rf_recall:.4f}", f"{rf_auc:.4f}",
        f"{rf_cm[0][1]:,}", f"{rf_cm[1][0]:,}",
        f"{rf_cm[1][1]:,}", f"{rf_cm[0][0]:,}"
    ],
    'GBM': [
        f"{gbm_accuracy:.4f}", f"{gbm_precision:.4f}",
        f"{gbm_recall:.4f}", f"{gbm_auc:.4f}",
        f"{gbm_cm[0][1]:,}", f"{gbm_cm[1][0]:,}",
        f"{gbm_cm[1][1]:,}", f"{gbm_cm[0][0]:,}"
    ]
})
print(f"\n{comparacion.to_string(index=False)}")

print("\nMejor modelo por métrica:")
metrics = {
    'Accuracy': (rf_accuracy, gbm_accuracy),
    'Precision': (rf_precision, gbm_precision),
    'Recall': (rf_recall, gbm_recall),
    'ROC-AUC': (rf_auc, gbm_auc)
}
for metric, (rf_val, gbm_val) in metrics.items():
    winner = "Random Forest" if rf_val >= gbm_val else "GBM"
    diff = abs(rf_val - gbm_val)
    print(f"  {metric:>12}: {winner} (ventaja: {diff:.4f})")

# ============================================================
# 7. GRÁFICAS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Modelos de Ensamble: Random Forest vs GBM\n'
             'Dataset: Default of Credit Card Clients (30,000 registros)',
             fontsize=14, fontweight='bold')

# 7.1 Curvas ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
gbm_fpr, gbm_tpr, _ = roc_curve(y_test, gbm_prob)

axes[0, 0].plot(rf_fpr, rf_tpr, 'b-', linewidth=2,
                label=f'Random Forest (AUC={rf_auc:.3f})')
axes[0, 0].plot(gbm_fpr, gbm_tpr, 'r--', linewidth=2,
                label=f'GBM (AUC={gbm_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Aleatorio (AUC=0.500)')
axes[0, 0].set_xlabel('Tasa de Falsos Positivos (FPR)')
axes[0, 0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
axes[0, 0].set_title('Curvas ROC')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 7.2 Matrices de confusión
for idx, (cm, title) in enumerate([
    (rf_cm, 'Random Forest'), (gbm_cm, 'GBM')
]):
    ax = axes[0, idx + 1]
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

# 7.3 Comparación de métricas (barras)
ax = axes[1, 0]
x = np.arange(4)
width = 0.35
rf_vals = [rf_accuracy, rf_precision, rf_recall, rf_auc]
gbm_vals = [gbm_accuracy, gbm_precision, gbm_recall, gbm_auc]
bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='steelblue')
bars2 = ax.bar(x + width/2, gbm_vals, width, label='GBM', color='indianred')
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

# 7.4 Feature Importance - Random Forest (Top 10)
ax = axes[1, 1]
top10_rf = rf_importances.head(10)
colors_rf = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
ax.barh(range(10), top10_rf.values[::-1], color=colors_rf)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_rf.index[::-1], fontsize=9)
ax.set_xlabel('Importancia')
ax.set_title('Top 10 Variables\nRandom Forest')
ax.grid(True, axis='x', alpha=0.3)

# 7.5 Feature Importance - GBM (Top 10)
ax = axes[1, 2]
top10_gbm = gbm_importances.head(10)
colors_gbm = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
ax.barh(range(10), top10_gbm.values[::-1], color=colors_gbm)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_gbm.index[::-1], fontsize=9)
ax.set_xlabel('Importancia')
ax.set_title('Top 10 Variables\nGBM')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
graficas_path = os.path.join(BASE_DIR, 'graficas_ensambles.png')
plt.savefig(graficas_path, dpi=150, bbox_inches='tight')
print(f"\nGráficas guardadas en: graficas_ensambles.png")

# ============================================================
# 8. ANÁLISIS DE IMPACTO DEL DESBALANCEO DE CLASES
# ============================================================
print("\n" + "=" * 70)
print("ANÁLISIS DEL DESBALANCEO DE CLASES")
print("=" * 70)

print(f"\nRatio de desbalanceo: {ratio_desbalanceo:.1f}:1 (No Default : Default)")
print(f"Tasa base (clasificar todo como No Default): {1 - y_test.mean():.2%}")
print(f"\nImpacto del desbalanceo en los modelos:")
print(f"  Random Forest - Recall: {rf_recall:.2%} (detecta {rf_recall:.1%} de defaults)")
print(f"  GBM           - Recall: {gbm_recall:.2%} (detecta {gbm_recall:.1%} de defaults)")
print(f"\n  Ambos modelos tienen accuracy > {min(rf_accuracy, gbm_accuracy):.0%}, pero la tasa base")
print(f"  ya es {1 - y_test.mean():.0%}. El verdadero valor está en el recall y ROC-AUC.")

# ============================================================
# 9. ANÁLISIS DE COSTOS DE ERROR
# ============================================================
print("\n" + "=" * 70)
print("ANÁLISIS DE COSTOS DE ERROR EN CONTEXTO FINANCIERO")
print("=" * 70)

limite_promedio = df['LIMIT_BAL'].mean()
costo_fn = limite_promedio * 0.60
costo_fp = limite_promedio * 0.03

print(f"\nLímite de crédito promedio: NT${limite_promedio:,.0f} (~US${limite_promedio/30:,.0f})")
print(f"Costo por Falso Negativo: NT${costo_fn:,.0f} (60% del crédito perdido)")
print(f"Costo por Falso Positivo: NT${costo_fp:,.0f} (3% costo de oportunidad)")

for name, cm in [('Random Forest', rf_cm), ('GBM', gbm_cm)]:
    fp, fn = cm[0][1], cm[1][0]
    fp_cost = fp * costo_fp
    fn_cost = fn * costo_fn
    total = fp_cost + fn_cost
    print(f"\n  {name}:")
    print(f"    FP: {fp:,} rechazados innecesariamente → NT${fp_cost:,.0f} (~US${fp_cost/30:,.0f})")
    print(f"    FN: {fn:,} defaults no detectados     → NT${fn_cost:,.0f} (~US${fn_cost/30:,.0f})")
    print(f"    COSTO TOTAL: NT${total:,.0f} (~US${total/30:,.0f})")

# ============================================================
# 10. CONCLUSIONES
# ============================================================
print("\n" + "=" * 70)
print("CONCLUSIONES")
print("=" * 70)
mejor_auc = "Random Forest" if rf_auc > gbm_auc else "GBM"
mejor_recall = "Random Forest" if rf_recall > gbm_recall else "GBM"
print(f"""
1. Mejor discriminación general (ROC-AUC): {mejor_auc}
2. Mejor detección de defaults (Recall): {mejor_recall}
3. Los ensambles superan a clasificadores individuales porque:
   - Reducen la varianza (Random Forest via bagging)
   - Reducen el sesgo (GBM via boosting secuencial)
   - Capturan interacciones complejas entre variables
4. El desbalanceo de clases ({ratio_desbalanceo:.1f}:1) afecta el recall.
   Estrategias de mitigación:
   - Cost-sensitive learning (class_weight='balanced')
   - Oversampling con SMOTE
   - Ajuste de umbral de decisión
5. En producción financiera, GBM (o XGBoost/LightGBM) es el estándar
   de la industria para scoring crediticio por su alto poder predictivo.
""")

print("=" * 70)
print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
