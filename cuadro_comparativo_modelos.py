"""
Cuadro Comparativo de 4 Modelos de ML - Para Presentación en Junta
===================================================================
Elaboró: Ernesto Ramírez

Genera una imagen visual y ejecutiva comparando los 4 modelos usados
en la predicción de default de tarjetas de crédito.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. CARGAR DATOS Y ENTRENAR LOS 4 MODELOS
# ============================================================
csv_path = os.path.join(BASE_DIR, 'dataset_credit_card_default.csv')
df = pd.read_csv(csv_path)

X = df.drop('default_payment_next_month', axis=1)
y = df['default_payment_next_month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar los 4 modelos
modelos = {}

# 1) Regresión Logística
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
modelos['Regresión\nLogística'] = {
    'pred': lr.predict(X_test_scaled),
    'prob': lr.predict_proba(X_test_scaled)[:, 1],
    'color': '#3498db',
    'tipo': 'Clasificador Simple',
    'icono': 'Lineal',
    'complejidad': 1,
    'interpretable': 5,
    'velocidad': 5,
}

# 2) Árbol de Decisión
dt = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_leaf=50, min_samples_split=100)
dt.fit(X_train, y_train)
modelos['Árbol de\nDecisión'] = {
    'pred': dt.predict(X_test),
    'prob': dt.predict_proba(X_test)[:, 1],
    'color': '#e67e22',
    'tipo': 'Clasificador Simple',
    'icono': 'No lineal',
    'complejidad': 2,
    'interpretable': 4,
    'velocidad': 5,
}

# 3) Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
modelos['Random\nForest'] = {
    'pred': rf.predict(X_test),
    'prob': rf.predict_proba(X_test)[:, 1],
    'color': '#2ecc71',
    'tipo': 'Ensamble (Bagging)',
    'icono': '200 árboles',
    'complejidad': 4,
    'interpretable': 2,
    'velocidad': 3,
}

# 4) GBM (Gradient Boosting)
gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  min_samples_leaf=20, subsample=0.8, random_state=42)
gbm.fit(X_train, y_train)
modelos['Gradient\nBoosting'] = {
    'pred': gbm.predict(X_test),
    'prob': gbm.predict_proba(X_test)[:, 1],
    'color': '#e74c3c',
    'tipo': 'Ensamble (Boosting)',
    'icono': '200 estimadores',
    'complejidad': 5,
    'interpretable': 1,
    'velocidad': 2,
}

# Calcular métricas
limite_promedio = df['LIMIT_BAL'].mean()
costo_fn = limite_promedio * 0.60
costo_fp = limite_promedio * 0.03

for nombre, m in modelos.items():
    m['accuracy'] = accuracy_score(y_test, m['pred'])
    m['precision'] = precision_score(y_test, m['pred'])
    m['recall'] = recall_score(y_test, m['pred'])
    m['auc'] = roc_auc_score(y_test, m['prob'])
    cm = confusion_matrix(y_test, m['pred'])
    m['cm'] = cm
    fp, fn = cm[0][1], cm[1][0]
    m['costo_total_usd'] = (fp * costo_fp + fn * costo_fn) / 30

nombres = list(modelos.keys())
colores = [modelos[n]['color'] for n in nombres]

# ============================================================
# 2. GENERAR CUADRO COMPARATIVO VISUAL
# ============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#f8f9fa')

# --- TÍTULO PRINCIPAL ---
fig.text(0.5, 0.97, 'COMPARATIVO DE MODELOS DE MACHINE LEARNING',
         ha='center', va='top', fontsize=24, fontweight='bold', color='#2c3e50')
fig.text(0.5, 0.955, 'Predicción de Default en Tarjetas de Crédito  |  30,000 clientes  |  Dataset UCI',
         ha='center', va='top', fontsize=13, color='#7f8c8d')

# ============================================================
# PANEL 1: TARJETAS RESUMEN DE CADA MODELO (fila superior)
# ============================================================
for i, nombre in enumerate(nombres):
    m = modelos[nombre]
    ax = fig.add_axes([0.03 + i * 0.245, 0.82, 0.22, 0.12])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color(m['color'])
        spine.set_linewidth(3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Barra de color superior
    ax.axhspan(8.5, 10, color=m['color'], alpha=0.15)

    ax.text(5, 9.2, nombre.replace('\n', ' '), ha='center', va='center',
            fontsize=14, fontweight='bold', color=m['color'])
    ax.text(5, 7.5, m['tipo'], ha='center', va='center',
            fontsize=10, color='#555555', style='italic')
    ax.text(5, 5.8, f"AUC: {m['auc']:.4f}", ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')
    ax.text(5, 4.0, f"Accuracy: {m['accuracy']:.1%}", ha='center', va='center',
            fontsize=11, color='#555555')
    ax.text(5, 2.5, f"Recall: {m['recall']:.1%}", ha='center', va='center',
            fontsize=11, color='#555555')
    ax.text(5, 1.0, f"Costo: US${m['costo_total_usd']:,.0f}", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#c0392b')

# ============================================================
# PANEL 2: BARRAS COMPARATIVAS DE MÉTRICAS
# ============================================================
ax_bars = fig.add_axes([0.08, 0.58, 0.88, 0.20])
ax_bars.set_facecolor('white')

metricas_nombres = ['Accuracy\n(Acierto global)', 'Precision\n(Precisión al predecir default)',
                    'Recall\n(Defaults detectados)', 'ROC-AUC\n(Poder de discriminación)']
metricas_keys = ['accuracy', 'precision', 'recall', 'auc']

x = np.arange(len(metricas_keys))
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, nombre in enumerate(nombres):
    m = modelos[nombre]
    vals = [m[k] for k in metricas_keys]
    bars = ax_bars.bar(x + offsets[i] * width, vals, width,
                       label=nombre.replace('\n', ' '), color=m['color'], alpha=0.85,
                       edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        ax_bars.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.2%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax_bars.set_xticks(x)
ax_bars.set_xticklabels(metricas_nombres, fontsize=10)
ax_bars.set_ylim(0, 1.15)
ax_bars.set_ylabel('Valor (0 a 1)', fontsize=11)
ax_bars.set_title('Comparación de Métricas Clave', fontsize=14, fontweight='bold', pad=10)
ax_bars.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax_bars.grid(True, axis='y', alpha=0.3)
ax_bars.axhline(y=1-y_test.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax_bars.text(3.5, 1-y_test.mean()+0.01, f'Línea base (predecir siempre "No Default"): {1-y_test.mean():.1%}',
             fontsize=8, color='gray', ha='right')

# ============================================================
# PANEL 3: TABLA RESUMEN EJECUTIVA
# ============================================================
ax_tabla = fig.add_axes([0.08, 0.40, 0.88, 0.14])
ax_tabla.axis('off')
ax_tabla.set_title('Resumen Ejecutivo - Tabla Comparativa', fontsize=14,
                   fontweight='bold', pad=15, loc='left')

tabla_data = []
for nombre in nombres:
    m = modelos[nombre]
    tabla_data.append([
        nombre.replace('\n', ' '),
        m['tipo'],
        f"{m['accuracy']:.2%}",
        f"{m['precision']:.2%}",
        f"{m['recall']:.2%}",
        f"{m['auc']:.4f}",
        f"US${m['costo_total_usd']:,.0f}",
    ])

col_labels = ['Modelo', 'Tipo', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Costo Total']
tabla = ax_tabla.table(cellText=tabla_data, colLabels=col_labels, loc='center',
                       cellLoc='center', colColours=['#ecf0f1'] * 7)
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.0, 1.8)

# Colorear la primera columna con el color de cada modelo
for i, nombre in enumerate(nombres):
    tabla[i + 1, 0].set_facecolor(modelos[nombre]['color'])
    tabla[i + 1, 0].set_text_props(color='white', fontweight='bold')

# Resaltar el mejor valor por columna
best_idx = {
    2: max(range(4), key=lambda i: modelos[nombres[i]]['accuracy']),
    3: max(range(4), key=lambda i: modelos[nombres[i]]['precision']),
    4: max(range(4), key=lambda i: modelos[nombres[i]]['recall']),
    5: max(range(4), key=lambda i: modelos[nombres[i]]['auc']),
    6: min(range(4), key=lambda i: modelos[nombres[i]]['costo_total_usd']),
}
for col, row_idx in best_idx.items():
    tabla[row_idx + 1, col].set_facecolor('#d5f5e3')
    tabla[row_idx + 1, col].set_text_props(fontweight='bold')

# ============================================================
# PANEL 4: SEMÁFORO VISUAL - Qué tan bueno es cada modelo
# ============================================================
ax_sem = fig.add_axes([0.08, 0.22, 0.88, 0.14])
ax_sem.set_facecolor('white')
ax_sem.set_title('Evaluación Visual por Criterio (escala 1-5)', fontsize=14,
                 fontweight='bold', pad=10, loc='left')

criterios = ['Precisión\nGeneral', 'Detección\nde Riesgo', 'Velocidad de\nEntrenamiento',
             'Facilidad de\nExplicar', 'Menor Costo\nde Errores']

# Puntajes cualitativos para cada modelo en cada criterio
# [Accuracy relativa, Recall relativa, Velocidad, Interpretabilidad, Costo inverso]
scores = {}
accs = [modelos[n]['accuracy'] for n in nombres]
recs = [modelos[n]['recall'] for n in nombres]
costs = [modelos[n]['costo_total_usd'] for n in nombres]

for i, nombre in enumerate(nombres):
    m = modelos[nombre]
    scores[nombre] = [
        3 + 2 * (m['accuracy'] - min(accs)) / max(max(accs) - min(accs), 0.001),
        1 + 4 * (m['recall'] - min(recs)) / max(max(recs) - min(recs), 0.001),
        m['velocidad'],
        m['interpretable'],
        1 + 4 * (max(costs) - m['costo_total_usd']) / max(max(costs) - min(costs), 0.001),
    ]

y_pos = np.arange(len(criterios))
bar_height = 0.18

for i, nombre in enumerate(nombres):
    vals = scores[nombre]
    ax_sem.barh(y_pos + (i - 1.5) * bar_height, vals, bar_height,
                color=modelos[nombre]['color'], alpha=0.8, label=nombre.replace('\n', ' '))
    for j, v in enumerate(vals):
        symbol = ['', '', '', '', ''][min(int(v) - 1, 4)]
        ax_sem.text(v + 0.05, y_pos[j] + (i - 1.5) * bar_height, symbol,
                    va='center', fontsize=8)

ax_sem.set_yticks(y_pos)
ax_sem.set_yticklabels(criterios, fontsize=10)
ax_sem.set_xlim(0, 5.8)
ax_sem.set_xticks([1, 2, 3, 4, 5])
ax_sem.set_xticklabels(['Bajo', 'Regular', 'Bueno', 'Muy Bueno', 'Excelente'], fontsize=9)
ax_sem.legend(loc='lower right', fontsize=9)
ax_sem.grid(True, axis='x', alpha=0.3)

# ============================================================
# PANEL 5: CONCLUSIONES EJECUTIVAS
# ============================================================
ax_conc = fig.add_axes([0.08, 0.02, 0.88, 0.17])
ax_conc.set_facecolor('#2c3e50')
ax_conc.set_xlim(0, 10)
ax_conc.set_ylim(0, 10)
ax_conc.set_xticks([])
ax_conc.set_yticks([])
for spine in ax_conc.spines.values():
    spine.set_visible(False)

# Encontrar mejores modelos
mejor_auc_nombre = max(nombres, key=lambda n: modelos[n]['auc'])
mejor_recall_nombre = max(nombres, key=lambda n: modelos[n]['recall'])
menor_costo_nombre = min(nombres, key=lambda n: modelos[n]['costo_total_usd'])

ax_conc.text(5, 9.3, 'CONCLUSIONES PARA LA JUNTA', ha='center', va='center',
             fontsize=16, fontweight='bold', color='white')

conclusiones = [
    f"Se evaluaron 4 modelos de inteligencia artificial para predecir qué clientes dejarán de pagar.",
    f"",
    f"Mejor poder de discriminación (ROC-AUC):  {mejor_auc_nombre.replace(chr(10), ' ')}  →  AUC = {modelos[mejor_auc_nombre]['auc']:.4f}",
    f"Mejor detección de morosos (Recall):  {mejor_recall_nombre.replace(chr(10), ' ')}  →  detecta {modelos[mejor_recall_nombre]['recall']:.1%} de los defaults",
    f"Menor costo por errores:  {menor_costo_nombre.replace(chr(10), ' ')}  →  US${modelos[menor_costo_nombre]['costo_total_usd']:,.0f}",
    f"",
    f"Recomendación: Gradient Boosting es el estándar de la industria bancaria por su alto",
    f"poder predictivo. Se sugiere como modelo base, complementado con ajuste de umbral",
    f"para mejorar la detección de clientes en riesgo de default.",
]

for i, linea in enumerate(conclusiones):
    color = 'white' if i in [0, 6, 7, 8] else '#f1c40f'
    fontw = 'bold' if i in [2, 3, 4] else 'normal'
    ax_conc.text(0.3, 7.8 - i * 0.9, linea, ha='left', va='center',
                 fontsize=10.5, color=color, fontweight=fontw)

plt.savefig(os.path.join(BASE_DIR, 'cuadro_comparativo_4_modelos.png'), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("=" * 70)
print("CUADRO COMPARATIVO GENERADO EXITOSAMENTE")
print("=" * 70)
print(f"\nArchivo: cuadro_comparativo_4_modelos.png")
print(f"\nContenido:")
print(f"  1. Tarjetas resumen de cada modelo")
print(f"  2. Barras comparativas de métricas")
print(f"  3. Tabla ejecutiva con mejores valores resaltados en verde")
print(f"  4. Evaluación visual por criterio (semáforo)")
print(f"  5. Conclusiones para la junta")
