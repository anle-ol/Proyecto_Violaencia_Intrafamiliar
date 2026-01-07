# Modelo de Riesgo de Victimización por Violencia Intrafamiliar en Colombia

**Elaborado por:** Angie Alejandra Olarte Varga, Santiago Mayorga Carvajal, Sandra Milena Alzate Alzate, Julia Carolina Torres Lozano, Óscar Alfredo Gómez Sanchez, Astrid Viviana Naranjo Abril y Laura Milena Gutiérrez Bustos

---

## Resumen

La violencia intrafamiliar en Colombia continúa representando un problema crítico de salud pública y justicia. Si bien se cuenta con datos abiertos de casos reportados desde 2014, el potencial de esta información para predecir escenarios de riesgo no ha sido adecuadamente explotado. Esta limitación analítica se manifiesta operativamente en la carencia de herramientas predictivas que permitan a las instituciones implementar intervenciones preventivas, manteniéndose en un enfoque reactivo centrado en la respuesta al daño ya consumado.

Para superar esta situación, el presente proyecto propone desarrollar e implementar un modelo predictivo que identifique niveles de riesgo de victimización mediante el análisis de variables sociodemográficas y contextuales reportadas por las víctimas entre 2014 y 2025. El objetivo fundamental es transformar los datos históricos en conocimiento accionable que sustente decisiones públicas fundamentadas.

La implementación exitosa de este modelo facilitará la transición hacia un enfoque preventivo basado en evidencia, permitiendo a las instituciones optimizar la asignación de recursos y diseñar intervenciones tempranas focalizadas, proporcionando a las políticas públicas evidencia cuantitativa sólida para el diseño de estrategias, y favoreciendo a la sociedad mediante la reducción progresiva de casos a través de una prevención proactiva que mejore la protección integral de potenciales víctimas.

---

## Abstract

Intimate partner violence in Colombia remains a critical public health and justice system concern. Although open data on reported cases has been available since 2014, the potential of this information to predict risk scenarios has not been adequately leveraged. This analytical limitation manifests operationally in the lack of predictive tools that would enable institutions to implement preventive interventions, thereby keeping them in a reactive approach focused on responding to harm after it has occurred.

To address this situation, this project proposes the development and implementation of a predictive model to identify levels of victimization risk through the analysis of sociodemographic and contextual variables reported by victims between 2014 and 2025. The primary objective is to transform historical data into actionable knowledge that supports evidence-based public decision-making.

The successful implementation of this model will facilitate a transition towards an evidence-based, preventive approach. This will allow institutions to optimize resource allocation and design targeted early interventions, provide public policy with robust quantitative evidence for strategy design, and benefit society through a progressive reduction in cases via proactive prevention, thereby enhancing the comprehensive protection of potential victims.

---

## Planteamiento del Problema

La violencia intrafamiliar persiste como un problema crítico en Colombia. Pese a la disponibilidad de datos abiertos desde 2014, su potencial para anticipar escenarios de riesgo de victimización ha sido subutilizado. Esta limitación se refleja en la carencia, por parte de las instituciones, de herramientas predictivas que faciliten una intervención preventiva y oportuna.

Frente a este escenario, el presente proyecto propone desarrollar un modelo que transforme dichos datos en conocimiento accionable, capaz de identificar y jerarquizar los factores de riesgo sociodemográficos y contextuales, con el fin de fundamentar la toma de decisiones públicas. De esta manera, se busca transitar de un enfoque reactivo —centrado en la respuesta al daño— a uno preventivo, sustentado en la evidencia.

---

## Justificación

La violencia intrafamiliar en Colombia representa uno de los tipos de violencia hacia la mujer y otros géneros que afecta notablemente la sociedad y que va en un aumento constante con el reporte de más de mil casos anuales. A pesar de que el país cuenta con fuentes oficiales de información abiertas desde 2015, como el Instituto Nacional De Medicina Legal Y Ciencias Forenses, Bogotá D.C. Estas bases de datos no han sido aprovechadas plenamente para anticipar o prevenir escenarios de riesgo de victimización.

En Colombia, las instituciones responsables de la atención y prevención de la violencia operan bajo un enfoque predominantemente reactivo, centrado en medir el nivel de afectación de las víctimas y en aplicar los protocolos correspondientes una vez ocurrido el hecho. Esta dinámica evidencia la carencia de herramientas predictivas capaces de identificar patrones de riesgo, lo que limita la posibilidad de orientar las decisiones institucionales hacia la prevención temprana.

La justificación del presente proyecto radica en la necesidad de transformar los datos en conocimiento útil y aplicable a las problemáticas sociales actuales, mediante la implementación de modelos de analítica predictiva. Esta innovación metodológica permitirá generar evidencia empírica sólida que oriente el diseño de acciones más efectivas y focalizadas, dirigidas a reducir los casos de violencia intrafamiliar.

Desde una perspectiva general, el proyecto contribuye a la protección de las víctimas potenciales y promueve entornos familiares más seguros, al posibilitar la caracterización precisa de los casos. Desde el enfoque técnico y científico, introduce el uso de herramientas de aprendizaje automático propias de la Inteligencia Artificial, aplicadas a la gestión social, lo que impulsa la modernización de las prácticas sociales contemporáneas y fomenta el aprovechamiento de los datos abiertos y las estadísticas como recursos estratégicos.

En conclusión, desde una perspectiva institucional, la implementación del modelo predictivo favorecerá una mejor asignación de recursos, una planificación proactiva de las intervenciones preventivas, y la reducción de los casos de violencia, contribuyendo así a la equidad y al fortalecimiento del bienestar social.

---

## Introducción

La violencia intrafamiliar es un problema social persistente en Colombia que afecta de manera desigual a distintos grupos poblacionales. Factores como el sexo, la edad, la escolaridad, la pertenencia étnica y la identidad de género influyen en los niveles de vulnerabilidad y en las formas en que se manifiesta este tipo de violencia.

Este proyecto busca identificar y caracterizar grupos vulnerables frente a la violencia intrafamiliar en Colombia entre 2014 y 2024, utilizando técnicas de aprendizaje automático supervisado y no supervisado. Mediante el uso del clustering (K-Modes), se analizarán patrones sociales y demográficos que permitan descubrir perfiles comunes entre las víctimas y comprender mejor las dinámicas de riesgo.

El objetivo final es aportar una visión más estructurada del fenómeno y generar información útil para orientar estrategias de prevención y políticas públicas con enfoque diferencial.

---

## Objetivos

### Objetivo General

Desarrollar un modelo predictivo que identifique el nivel de riesgo de victimización en casos de violencia intrafamiliar en Colombia, a partir de variables sociodemográficas y contextuales suministrados por las víctimas desde 2014 hasta el año 2024.

### Objetivos Específicos

1. Recopilar y limpiar datos de violencia intrafamiliar del Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF) para el período 2014-2024.
2. Realizar un análisis exploratorio de datos (EDA) para identificar patrones, tendencias y características relevantes en los casos reportados.
3. Implementar técnicas de clustering (K-Modes) para identificar perfiles de riesgo y grupos vulnerables.
4. Desarrollar modelos predictivos utilizando algoritmos de machine learning para clasificar niveles de riesgo.
5. Implementar modelos supervisados de regresión (Regresión Logística, Regresión Lineal, Gradient Boosting) para predecir scores de riesgo continuos.
6. Validar y evaluar el rendimiento de los modelos desarrollados (clasificación y regresión).
6. Generar visualizaciones y reportes que faciliten la interpretación de los resultados.
7. Proponer recomendaciones de políticas públicas basadas en los hallazgos del análisis.

---

## Metodología

### 1. Recopilación y Limpieza de Datos

El primer paso consiste en cargar y preparar los datos para el análisis:

```python
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Cargar el dataset limpio para modelado
# NOTA: Este archivo contiene 9 columnas optimizadas para modelado,
# incluyendo la columna 'Porcentaje de riesgo' que puede usarse como variable objetivo
df = pd.read_csv('intrafamiliar_modelov2.csv', encoding='utf-8')

# Exploración inicial
print(f"Dimensiones del dataset: {df.shape}")
print(f"\nColumnas disponibles: {df.columns.tolist()}")
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nValores faltantes:")
print(df.isnull().sum())
```

### 2. Análisis Exploratorio de Datos (EDA)

Realizamos un análisis exploratorio para entender la distribución y características de los datos:

```python
# Análisis descriptivo básico
def analisis_exploratorio(df):
    """Realiza análisis exploratorio de los datos"""
    
    # Estadísticas descriptivas para variables categóricas
    print("=" * 80)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 80)
    
    # Verificar columnas disponibles
    print(f"\nColumnas disponibles: {df.columns.tolist()}")
    print(f"Total de registros: {len(df):,}")
    
    # Distribución por grupo de edad
    if 'Grupo de Edad judicial' in df.columns:
        print("\n1. DISTRIBUCIÓN POR GRUPO DE EDAD JUDICIAL")
        print(df['Grupo de Edad judicial'].value_counts().head(10))
    
    # Distribución por sexo del agresor
    if 'Sexo del Agresor' in df.columns:
        print("\n2. DISTRIBUCIÓN POR SEXO DEL AGRESOR")
        print(df['Sexo del Agresor'].value_counts())
        print(f"\nPorcentajes:")
        print(df['Sexo del Agresor'].value_counts(normalize=True) * 100)
    
    # Distribución por presunto agresor
    if 'Presunto Agresor Detallado' in df.columns:
        print("\n3. DISTRIBUCIÓN POR PRESUNTO AGRESOR DETALLADO")
        print(df['Presunto Agresor Detallado'].value_counts().head(10))
    
    # Factores desencadenantes
    if 'Factor Desencadenante de la Agresión' in df.columns:
        print("\n4. FACTORES DESENCADENANTES MÁS FRECUENTES")
        print(df['Factor Desencadenante de la Agresión'].value_counts().head(10))
    
    # Distribución por departamento
    if 'Departamento del hecho DANE' in df.columns:
        print("\n5. TOP 10 DEPARTAMENTOS CON MÁS CASOS")
        print(df['Departamento del hecho DANE'].value_counts().head(10))
    
    # Escolaridad
    if 'Escolaridad' in df.columns:
        print("\n6. DISTRIBUCIÓN POR ESCOLARIDAD")
        print(df['Escolaridad'].value_counts().head(10))
    
    # Escenario del hecho
    if 'Escenario del Hecho' in df.columns:
        print("\n7. DISTRIBUCIÓN POR ESCENARIO DEL HECHO")
        print(df['Escenario del Hecho'].value_counts().head(10))
    
    # Porcentaje de riesgo (si existe)
    if 'Porcentaje de riesgo' in df.columns:
        print("\n8. ESTADÍSTICAS DEL PORCENTAJE DE RIESGO")
        print(df['Porcentaje de riesgo'].describe())
    
    return df

# Ejecutar análisis exploratorio
df_analizado = analisis_exploratorio(df)
```

### 3. Preparación de Datos para Clustering

Preparamos las variables categóricas para el algoritmo K-Modes:

```python
def preparar_datos_clustering(df):
    """Prepara los datos para el clustering con K-Modes"""
    
    # Seleccionar variables relevantes para el clustering (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_clustering = [var for var in variables_disponibles if var in df.columns]
    
    # Crear dataset para clustering
    df_clustering = df[variables_clustering].copy()
    
    # Manejar valores faltantes
    df_clustering = df_clustering.fillna('Sin información')
    
    # Convertir a string para evitar problemas con tipos mixtos
    for col in df_clustering.columns:
        df_clustering[col] = df_clustering[col].astype(str)
    
    print(f"Dataset para clustering: {df_clustering.shape}")
    print(f"\nVariables seleccionadas: {variables_clustering}")
    
    return df_clustering

# Preparar datos
df_clustering = preparar_datos_clustering(df)
```

### 4. Implementación de Clustering K-Modes

Implementamos el algoritmo K-Modes para identificar perfiles de riesgo:

```python
def aplicar_kmodes(df_clustering, n_clusters=3, random_state=42):
    """Aplica el algoritmo K-Modes para clustering"""
    
    print(f"\n{'='*80}")
    print(f"APLICANDO K-MODES CON {n_clusters} CLUSTERS")
    print(f"{'='*80}\n")
    
    # Convertir a array numpy
    data_array = df_clustering.values
    
    # Aplicar K-Modes
    kmodes = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1, random_state=random_state)
    clusters = kmodes.fit_predict(data_array)
    
    # Agregar clusters al dataframe original
    df_clustering['Cluster'] = clusters
    
    # Estadísticas de clusters
    print("\nDISTRIBUCIÓN DE CLUSTERS:")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_percentages = pd.Series(clusters).value_counts(normalize=True).sort_index() * 100
    
    for i in range(n_clusters):
        print(f"\nCluster {i}:")
        print(f"  - Cantidad de casos: {cluster_counts[i]:,}")
        print(f"  - Porcentaje: {cluster_percentages[i]:.2f}%")
    
    # Modas de cada cluster
    print("\n\nMODAS DE CADA CLUSTER:")
    for i in range(n_clusters):
        print(f"\n{'='*80}")
        print(f"CLUSTER {i} - PERFIL DE RIESGO")
        print(f"{'='*80}")
        cluster_data = df_clustering[df_clustering['Cluster'] == i]
        
        for col in df_clustering.columns[:-1]:  # Excluir columna Cluster
            mode_value = cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else 'N/A'
            mode_count = (cluster_data[col] == mode_value).sum()
            mode_percentage = (mode_count / len(cluster_data)) * 100
            print(f"\n{col}:")
            print(f"  Moda: {mode_value}")
            print(f"  Frecuencia: {mode_count:,} ({mode_percentage:.2f}%)")
    
    return kmodes, clusters, df_clustering

# Aplicar K-Modes con 3 clusters
kmodes_model, clusters, df_clustered = aplicar_kmodes(df_clustering, n_clusters=3, random_state=42)
```

### 5. Análisis Detallado de Clusters

Realizamos un análisis detallado de cada cluster identificado:

```python
def analizar_clusters_detallado(df, df_clustered):
    """Realiza un análisis detallado de cada cluster"""
    
    # Agregar cluster al dataframe original
    df['Cluster'] = df_clustered['Cluster']
    
    print("\n" + "="*80)
    print("ANÁLISIS DETALLADO DE CLUSTERS")
    print("="*80)
    
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        total_casos = len(cluster_data)
        porcentaje = (total_casos / len(df)) * 100
        
        print(f"\n\n{'='*80}")
        print(f"CLUSTER {cluster_id} - {porcentaje:.2f}% de los casos ({total_casos:,} casos)")
        print(f"{'='*80}\n")
        
        # Sexo del Agresor
        if 'Sexo del Agresor' in cluster_data.columns:
            print("SEXO DEL AGRESOR:")
            print(cluster_data['Sexo del Agresor'].value_counts().head(3))
        
        # Presunto Agresor Detallado
        if 'Presunto Agresor Detallado' in cluster_data.columns:
            print("\nPRESUNTO AGRESOR DETALLADO:")
            print(cluster_data['Presunto Agresor Detallado'].value_counts().head(5))
        
        # Grupo de Edad Judicial
        if 'Grupo de Edad judicial' in cluster_data.columns:
            print("\nGRUPO DE EDAD JUDICIAL:")
            print(cluster_data['Grupo de Edad judicial'].value_counts().head(5))
        
        # Escolaridad
        if 'Escolaridad' in cluster_data.columns:
            print("\nESCOLARIDAD:")
            print(cluster_data['Escolaridad'].value_counts().head(5))
        
        # Factor Desencadenante
        if 'Factor Desencadenante de la Agresión' in cluster_data.columns:
            print("\nFACTOR DESENCADENANTE:")
            print(cluster_data['Factor Desencadenante de la Agresión'].value_counts().head(5))
        
        # Escenario del Hecho
        if 'Escenario del Hecho' in cluster_data.columns:
            print("\nESCENARIO DEL HECHO:")
            print(cluster_data['Escenario del Hecho'].value_counts().head(5))
        
        # Porcentaje de riesgo (si existe)
        if 'Porcentaje de riesgo' in cluster_data.columns:
            print("\nPORCENTAJE DE RIESGO:")
            print(cluster_data['Porcentaje de riesgo'].describe())
    
    return df

# Realizar análisis detallado
df_final = analizar_clusters_detallado(df, df_clustered)
```

### 6. Visualizaciones

Creamos visualizaciones para entender mejor los clusters:

```python
def crear_visualizaciones(df):
    """Crea visualizaciones de los clusters"""
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Figura 1: Distribución de clusters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribución de clusters
    cluster_counts = df['Cluster'].value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Distribución de Clusters', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Número de Casos')
    for i, v in enumerate(cluster_counts.values):
        axes[0, 0].text(cluster_counts.index[i], v + 1000, f'{v:,}', 
                        ha='center', fontweight='bold')
    
    # 2. Sexo del Agresor por Cluster (si existe la columna)
    if 'Sexo del Agresor' in df.columns:
        sexo_agresor_cluster = pd.crosstab(df['Cluster'], df['Sexo del Agresor'])
        sexo_agresor_cluster.plot(kind='bar', ax=axes[0, 1], color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_title('Sexo del Agresor por Cluster', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Número de Casos')
        axes[0, 1].legend(title='Sexo del Agresor')
        axes[0, 1].tick_params(axis='x', rotation=0)
    else:
        # Si no existe, mostrar otra variable
        if 'Escolaridad' in df.columns:
            escolaridad_cluster = pd.crosstab(df['Cluster'], df['Escolaridad'])
            top_escolaridad = escolaridad_cluster.sum().nlargest(5).index
            escolaridad_cluster[top_escolaridad].plot(kind='bar', ax=axes[0, 1], width=0.8)
            axes[0, 1].set_title('Escolaridad por Cluster (Top 5)', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Número de Casos')
            axes[0, 1].legend(title='Escolaridad', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Presunto Agresor por Cluster (Top 5)
    if 'Presunto Agresor Detallado' in df.columns:
        agresor_cluster = pd.crosstab(df['Cluster'], df['Presunto Agresor Detallado'])
        top_agresores = agresor_cluster.sum().nlargest(5).index
        agresor_cluster[top_agresores].plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Top 5 Presuntos Agresores por Cluster', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Número de Casos')
        axes[1, 0].legend(title='Presunto Agresor', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=0)
    else:
        axes[1, 0].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Presunto Agresor por Cluster', fontsize=14, fontweight='bold')
    
    # 4. Factor Desencadenante por Cluster (Top 5)
    if 'Factor Desencadenante de la Agresión' in df.columns:
        factor_cluster = pd.crosstab(df['Cluster'], df['Factor Desencadenante de la Agresión'])
        top_factores = factor_cluster.sum().nlargest(5).index
        factor_cluster[top_factores].plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title('Top 5 Factores Desencadenantes por Cluster', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Número de Casos')
        axes[1, 1].legend(title='Factor Desencadenante', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=0)
    else:
        axes[1, 1].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Factores Desencadenantes por Cluster', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analisis_clusters.png', dpi=300, bbox_inches='tight')
    print("\nVisualización guardada como 'analisis_clusters.png'")
    plt.close()
    
    # Figura 2: Análisis temporal
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribución por departamento (en lugar de año si no está disponible)
    if 'Departamento del hecho DANE' in df.columns:
        dept_cluster = pd.crosstab(df['Departamento del hecho DANE'], df['Cluster'])
        top_depts = dept_cluster.sum().nlargest(10).index
        dept_cluster.loc[top_depts].plot(kind='barh', ax=axes[0], width=0.8)
        axes[0].set_title('Top 10 Departamentos por Cluster', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Número de Casos')
        axes[0].set_ylabel('Departamento')
        axes[0].legend(title='Cluster', labels=[f'Cluster {i}' for i in sorted(df['Cluster'].unique())])
    else:
        axes[0].text(0.5, 0.5, 'Datos no disponibles', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Distribución Temporal', fontsize=14, fontweight='bold')
    
    # Distribución por grupo de edad
    edad_cluster = pd.crosstab(df['Grupo de Edad judicial'], df['Cluster'])
    top_edades = edad_cluster.sum().nlargest(10).index
    edad_cluster.loc[top_edades].plot(kind='barh', ax=axes[1], width=0.8)
    axes[1].set_title('Top 10 Grupos de Edad por Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Número de Casos')
    axes[1].set_ylabel('Grupo de Edad Judicial')
    axes[1].legend(title='Cluster', labels=[f'Cluster {i}' for i in sorted(df['Cluster'].unique())])
    
    plt.tight_layout()
    plt.savefig('analisis_temporal_edad.png', dpi=300, bbox_inches='tight')
    print("Visualización guardada como 'analisis_temporal_edad.png'")
    plt.close()

# Crear visualizaciones
crear_visualizaciones(df_final)
```

### 7. Modelo Predictivo (Opcional)

Si se desea desarrollar un modelo predictivo para clasificar nuevos casos:

```python
def crear_modelo_predictivo(df):
    """Crea un modelo predictivo para clasificar casos en clusters"""
    
    # Seleccionar variables predictoras (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_predictoras = [var for var in variables_disponibles if var in df.columns]
    
    # Preparar datos
    X = df[variables_predictoras].copy()
    y = df['Cluster'].copy()
    
    # Codificar variables categóricas
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo Random Forest
    print("\n" + "="*80)
    print("ENTRENANDO MODELO PREDICTIVO (RANDOM FOREST)")
    print("="*80 + "\n")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predecir
    y_pred = rf_model.predict(X_test)
    
    # Evaluar modelo
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=[f'Cluster {i}' for i in sorted(y.unique())]))
    
    print("\nMATRIZ DE CONFUSIÓN:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Importancia de características
    print("\nTOP 10 VARIABLES MÁS IMPORTANTES:")
    feature_importance = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': rf_model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    print(feature_importance.head(10))
    
    return rf_model, label_encoders, feature_importance

# Crear modelo predictivo (opcional)
# modelo_rf, encoders, importancia = crear_modelo_predictivo(df_final)
```

### 8. Modelos Supervisados de Regresión

Implementamos modelos de regresión supervisados para predecir niveles de riesgo y probabilidades de victimización:

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_curve, auc, roc_auc_score
import statsmodels.api as sm

def crear_variable_riesgo(df):
    """Crea o utiliza la variable de riesgo continua basada en características del caso"""
    
    # Si ya existe la columna 'Porcentaje de riesgo', usarla directamente
    if 'Porcentaje de riesgo' in df.columns:
        print("\n" + "="*80)
        print("VARIABLE DE RIESGO EXISTENTE DETECTADA")
        print("="*80)
        print(f"\nUsando columna 'Porcentaje de riesgo' existente")
        print(f"Estadísticas del Porcentaje de Riesgo:")
        print(df['Porcentaje de riesgo'].describe())
        
        # Normalizar a escala 0-10 si es necesario
        if df['Porcentaje de riesgo'].max() <= 1:
            df['Riesgo_Normalizado'] = df['Porcentaje de riesgo'] * 10
        else:
            df['Riesgo_Normalizado'] = df['Porcentaje de riesgo']
        
        return df
    
    # Si no existe, crear una nueva
    df_riesgo = df.copy()
    
    # Asignar pesos a diferentes factores de riesgo
    riesgo_score = 0
    
    # Factor 1: Edad de la víctima (menores tienen mayor riesgo)
    edad_riesgo = {
        '(00 a 04)': 5,
        '(05 a 09)': 4,
        '(10 a 13)': 4,
        '(14 a 17)': 3,
        '(18 a 19)': 2,
        '(20 a 24)': 2,
        '(25 a 29)': 1,
        '(30 a 34)': 1,
        '(35 a 39)': 1,
        '(40 a 44)': 1,
        '(45 a 49)': 1,
        '(50 a 54)': 1,
        '(55 a 59)': 1,
        '(60 a 64)': 1,
        '(65 a 69)': 1,
        '(70 a 74)': 1,
        '(75 a 79)': 1,
        '(80 y más)': 1
    }
    
    # Factor 2: Tipo de agresor (padre/madre hacia menores = mayor riesgo)
    agresor_riesgo = {
        'Padre': 5,
        'Madre': 4,
        'Hijo(a)': 3,
        'Hermano(a)': 3,
        'Cónyuge': 4,
        'Pareja': 3,
        'Otro familiar': 2,
        'Ex pareja': 2,
        'Otro': 1
    }
    
    # Factor 3: Factor desencadenante
    factor_riesgo = {
        'Intolerancia / Machismo': 5,
        'Consumo de alcohol y/o sustancias psicoactivas': 4,
        'Problemas económicos': 3,
        'Celos': 3,
        'Problemas de convivencia': 2,
        'Otro': 1
    }
    
    # Factor 4: Escolaridad (menor escolaridad = mayor riesgo)
    escolaridad_riesgo = {
        'Sin escolaridad': 5,
        'Preescolar': 4,
        'Primaria': 3,
        'Secundaria': 2,
        'Técnica': 1,
        'Tecnológica': 1,
        'Universitaria': 1,
        'Postgrado': 1,
        'Sin información': 2
    }
    
    # Calcular score de riesgo para cada caso
    scores = []
    for idx, row in df_riesgo.iterrows():
        score = 0
        
        # Edad
        edad = str(row.get('Grupo de Edad judicial', 'Sin información'))
        score += edad_riesgo.get(edad, 1)
        
        # Agresor
        agresor = str(row.get('Presunto Agresor Detallado', 'Otro'))
        score += agresor_riesgo.get(agresor, 1)
        
        # Factor desencadenante
        factor = str(row.get('Factor Desencadenante de la Agresión', 'Otro'))
        score += factor_riesgo.get(factor, 1)
        
        # Escolaridad
        escolaridad = str(row.get('Escolaridad', 'Sin información'))
        score += escolaridad_riesgo.get(escolaridad, 1)
        
        scores.append(score)
    
    df_riesgo['Riesgo_Score'] = scores
    
    # Normalizar el score a escala 0-10
    df_riesgo['Riesgo_Normalizado'] = (
        (df_riesgo['Riesgo_Score'] - df_riesgo['Riesgo_Score'].min()) / 
        (df_riesgo['Riesgo_Score'].max() - df_riesgo['Riesgo_Score'].min())
    ) * 10
    
    print("\n" + "="*80)
    print("VARIABLE DE RIESGO CREADA")
    print("="*80)
    print(f"\nEstadísticas del Score de Riesgo:")
    print(df_riesgo['Riesgo_Score'].describe())
    print(f"\nEstadísticas del Riesgo Normalizado (0-10):")
    print(df_riesgo['Riesgo_Normalizado'].describe())
    
    return df_riesgo

# Crear variable de riesgo
df_con_riesgo = crear_variable_riesgo(df_final)
```

#### 8.1. Regresión Logística para Clasificación de Niveles de Riesgo

```python
def modelo_regresion_logistica(df):
    """Implementa regresión logística para clasificar niveles de riesgo"""
    
    print("\n" + "="*80)
    print("MODELO DE REGRESIÓN LOGÍSTICA")
    print("="*80 + "\n")
    
    # Crear variable objetivo categórica (bajo, medio, alto riesgo)
    df_reg = df.copy()
    
    # Clasificar riesgo en categorías
    def clasificar_riesgo(score):
        if score <= 3.33:
            return 0  # Bajo riesgo
        elif score <= 6.66:
            return 1  # Medio riesgo
        else:
            return 2  # Alto riesgo
    
    df_reg['Nivel_Riesgo'] = df_reg['Riesgo_Normalizado'].apply(clasificar_riesgo)
    
    # Seleccionar variables predictoras (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_predictoras = [var for var in variables_disponibles if var in df.columns]
    
    # Preparar datos
    X = df_reg[variables_predictoras].copy()
    y = df_reg['Nivel_Riesgo'].copy()
    
    # Codificar variables categóricas
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    # Usar OneHotEncoder para variables categóricas
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), list(range(len(X.columns))))],
        remainder='passthrough'
    )
    
    X_encoded = ct.fit_transform(X)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Estandarizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo de regresión logística
    print("Entrenando modelo de Regresión Logística...")
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = log_reg.predict(X_test_scaled)
    y_pred_proba = log_reg.predict_proba(X_test_scaled)
    
    # Evaluar modelo
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']))
    
    print("\nMATRIZ DE CONFUSIÓN:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calcular AUC-ROC para cada clase
    print("\nAUC-ROC por clase:")
    for i, clase in enumerate(['Bajo Riesgo', 'Medio Riesgo', 'Alto Riesgo']):
        y_binary = (y_test == i).astype(int)
        y_proba_binary = y_pred_proba[:, i]
        if len(np.unique(y_binary)) > 1:
            auc_score = roc_auc_score(y_binary, y_proba_binary)
            print(f"  {clase}: {auc_score:.4f}")
    
    # Coeficientes del modelo
    print("\nTOP 10 COEFICIENTES MÁS IMPORTANTES (Clase: Alto Riesgo):")
    feature_names = ct.get_feature_names_out()
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': log_reg.coef_[2]  # Coeficientes para clase "Alto Riesgo"
    }).sort_values('Coeficiente', ascending=False, key=abs)
    print(coef_df.head(10))
    
    return log_reg, scaler, ct, df_reg

# Entrenar modelo de regresión logística
modelo_log, scaler_log, encoder_log, df_reg_log = modelo_regresion_logistica(df_con_riesgo)
```

#### 8.2. Regresión Lineal para Predicción de Score de Riesgo

```python
def modelo_regresion_lineal(df):
    """Implementa regresión lineal para predecir el score de riesgo continuo"""
    
    print("\n" + "="*80)
    print("MODELO DE REGRESIÓN LINEAL")
    print("="*80 + "\n")
    
    # Seleccionar variables predictoras (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_predictoras = [var for var in variables_disponibles if var in df.columns]
    
    # Preparar datos
    X = df[variables_predictoras].copy()
    y = df['Riesgo_Normalizado'].copy()
    
    # Codificar variables categóricas usando OneHotEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), list(range(len(X.columns))))],
        remainder='passthrough'
    )
    
    X_encoded = ct.fit_transform(X)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Estandarizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo de regresión lineal
    print("Entrenando modelo de Regresión Lineal...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    
    # Predecir
    y_pred = lin_reg.predict(X_test_scaled)
    
    # Evaluar modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nMÉTRICAS DE EVALUACIÓN:")
    print(f"  MSE (Mean Squared Error): {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  R² (Coeficiente de Determinación): {r2:.4f}")
    
    # Visualizar predicciones vs valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Riesgo Real')
    plt.ylabel('Riesgo Predicho')
    plt.title('Regresión Lineal: Predicciones vs Valores Reales')
    plt.grid(True, alpha=0.3)
    plt.savefig('regresion_lineal_predicciones.png', dpi=300, bbox_inches='tight')
    print("\nVisualización guardada como 'regresion_lineal_predicciones.png'")
    plt.close()
    
    # Coeficientes más importantes
    print("\nTOP 15 COEFICIENTES MÁS IMPORTANTES:")
    feature_names = ct.get_feature_names_out()
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': lin_reg.coef_
    }).sort_values('Coeficiente', ascending=False, key=abs)
    print(coef_df.head(15))
    
    return lin_reg, scaler, ct

# Entrenar modelo de regresión lineal
modelo_lin, scaler_lin, encoder_lin = modelo_regresion_lineal(df_con_riesgo)
```

#### 8.3. Gradient Boosting Regressor para Predicción de Riesgo

```python
def modelo_gradient_boosting(df):
    """Implementa Gradient Boosting Regressor para predecir score de riesgo"""
    
    print("\n" + "="*80)
    print("MODELO DE GRADIENT BOOSTING REGRESSOR")
    print("="*80 + "\n")
    
    # Seleccionar variables predictoras (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_predictoras = [var for var in variables_disponibles if var in df.columns]
    
    # Preparar datos
    X = df[variables_predictoras].copy()
    y = df['Riesgo_Normalizado'].copy()
    
    # Codificar variables categóricas
    from sklearn.preprocessing import LabelEncoder
    
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo Gradient Boosting
    print("Entrenando modelo Gradient Boosting Regressor...")
    gb_reg = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=1
    )
    gb_reg.fit(X_train, y_train)
    
    # Predecir
    y_pred = gb_reg.predict(X_test)
    
    # Evaluar modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nMÉTRICAS DE EVALUACIÓN:")
    print(f"  MSE (Mean Squared Error): {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  R² (Coeficiente de Determinación): {r2:.4f}")
    
    # Visualizar predicciones vs valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Riesgo Real')
    plt.ylabel('Riesgo Predicho')
    plt.title('Gradient Boosting: Predicciones vs Valores Reales')
    plt.grid(True, alpha=0.3)
    plt.savefig('gradient_boosting_predicciones.png', dpi=300, bbox_inches='tight')
    print("\nVisualización guardada como 'gradient_boosting_predicciones.png'")
    plt.close()
    
    # Importancia de características
    print("\nTOP 15 VARIABLES MÁS IMPORTANTES:")
    feature_importance = pd.DataFrame({
        'Variable': X.columns,
        'Importancia': gb_reg.feature_importances_
    }).sort_values('Importancia', ascending=False)
    print(feature_importance.head(15))
    
    # Visualizar importancia de características
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importancia'].values)
    plt.yticks(range(len(top_features)), top_features['Variable'].values)
    plt.xlabel('Importancia')
    plt.title('Importancia de Variables - Gradient Boosting')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('importancia_variables_gb.png', dpi=300, bbox_inches='tight')
    print("Visualización guardada como 'importancia_variables_gb.png'")
    plt.close()
    
    return gb_reg, label_encoders

# Entrenar modelo Gradient Boosting
modelo_gb, encoders_gb = modelo_gradient_boosting(df_con_riesgo)
```

#### 8.4. Comparación de Modelos de Regresión

```python
def comparar_modelos_regresion(df):
    """Compara el rendimiento de diferentes modelos de regresión"""
    
    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS DE REGRESIÓN")
    print("="*80 + "\n")
    
    # Preparar datos comunes (disponibles en intrafamiliar_modelov2.csv)
    variables_disponibles = [
        'Grupo de Edad judicial',
        'Escolaridad',
        'Sexo del Agresor',
        'Presunto Agresor Detallado',
        'Factor Desencadenante de la Agresión',
        'Escenario del Hecho',
        'Actividad Durante el Hecho',
        'Departamento del hecho DANE'
    ]
    
    # Filtrar solo las variables que existen en el dataframe
    variables_predictoras = [var for var in variables_disponibles if var in df.columns]
    
    X = df[variables_predictoras].copy()
    y = df['Riesgo_Normalizado'].copy()
    
    # Codificar variables
    from sklearn.preprocessing import LabelEncoder
    
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Estandarizar para modelos que lo requieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a comparar
    modelos = {
        'Regresión Lineal': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    resultados = []
    
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        
        # Entrenar modelo
        if nombre == 'Regresión Lineal':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        resultados.append({
            'Modelo': nombre,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
        
        print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('R²', ascending=False)
    
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(df_resultados.to_string(index=False))
    
    # Visualizar comparación
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    modelos_nombres = df_resultados['Modelo'].values
    metricas = ['MSE', 'RMSE', 'MAE', 'R²']
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx // 2, idx % 2]
        valores = df_resultados[metrica].values
        bars = ax.bar(modelos_nombres, valores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title(f'Comparación: {metrica}', fontsize=12, fontweight='bold')
        ax.set_ylabel(metrica)
        ax.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos_regresion.png', dpi=300, bbox_inches='tight')
    print("\nVisualización guardada como 'comparacion_modelos_regresion.png'")
    plt.close()
    
    return df_resultados

# Comparar modelos
resultados_comparacion = comparar_modelos_regresion(df_con_riesgo)
```

---

## Resultados del Análisis de Clusters

### Cluster 0: Perfil de Riesgo Matriarcal y de Menores (35.12%)

Este cluster representa el grupo más grande y se caracteriza por:

- **Sexo del Agresor:** Mayoritariamente Mujer
- **Presunto Agresor Detallado:** Los picos se observan en Madre, seguido de otros familiares cercanos
- **Grupo de Edad Judicial:** Predominio en población infantil y adolescente: (05 a 09), (10 a 13) y (14 a 17) años
- **Escolaridad:** Alta concentración en Primaria, Sin escolaridad y Sin información
- **Factor desencadenante:** Predominan la intolerancia o el machismo, seguidos del consumo de alcohol y otras sustancias
- **Conclusión del Perfil:** Este cluster identifica la violencia intrafamiliar ejercida principalmente por mujeres (madres) hacia menores, con un claro enfoque en el riesgo de agresión materna en el hogar, especialmente en contextos de baja escolaridad.

### Cluster 1: Perfil de Riesgo Adulto Joven y Familiar Cercano (43.76%)

Este es el grupo más grande y define un patrón de agresión entre adultos jóvenes:

- **Sexo del Agresor:** Mayoritariamente Hombre
- **Presunto Agresor Detallado:** Los picos se observan en Hermano(a) e Hijo(a), sugiriendo conflictos entre pares dentro del núcleo familiar o con descendientes directos
- **Grupo de Edad Judicial:** Predominio en el segmento de adultos jóvenes (20 a 24) y (25 a 29) años
- **Factor desencadenante:** Predominan la intolerancia o el machismo, seguidos del consumo de alcohol y otras sustancias, lo que refleja un cambio en las dinámicas de riesgo
- **Conclusión del Perfil:** Este cluster abarca las agresiones perpetradas por hombres (o en menor medida hermanos/hijos) en un rango de edad adulta joven, lo que puede estar ligado a dinámicas de convivencia y tensión económica o social en el hogar.

### Cluster 2: Perfil de Riesgo Paternal y de la Infancia (21.12%)

Este es el grupo más pequeño, pero define un patrón muy específico de agresión hacia los menores por parte del padre:

- **Sexo del Agresor:** Mayoritariamente Hombre
- **Presunto Agresor Detallado:** El Padre es el agresor detallado más frecuente
- **Grupo de Edad Judicial:** Se concentra en la población infantil y adolescente más joven: (10 a 13) y (05 a 09) años
- **Escolaridad:** Alta concentración en Primaria, Sin escolaridad y Sin información
- **Factor desencadenante:** Al igual que en el Cluster 1, predomina la intolerancia o el machismo, seguido del consumo de alcohol y otras sustancias
- **Conclusión del Perfil:** Este grupo identifica la violencia intrafamiliar ejercida por hombres (padres), dirigida específicamente a niños y adolescentes, con un claro enfoque en el riesgo de agresión paterna en el hogar.

---

## Conclusiones

Según el análisis de violencia intrafamiliar en Colombia (2014-2024), la violencia está comprendida mayormente dentro del núcleo familiar cercano como lo son: padres, madres, hijos o hermanos.

Según lo anterior podemos concluir que:

1. **Los tres clusters reflejan que las causas más frecuentes son la intolerancia y el machismo**, seguidas por problemas con consumo de sustancias y baja escolaridad.

2. **La mayoría de las agresiones ocurren en el hogar o durante actividades cotidianas**, lo que sugiere que la violencia no es un hecho aislado, sino parte de la dinámica familiar y social.

3. **Los resultados muestran que la baja escolaridad no solo afecta el desarrollo económico**, sino también las habilidades emocionales y la capacidad de resolver conflictos de manera pacífica.

4. **Esto plantea la necesidad de políticas que aborden tanto el desarrollo educativo y económico**, como la educación emocional y familiar.

---

## Propuestas de Políticas Públicas

### Cluster 0 – Matriarcal y de Menores (35.12%)

**Violencia ejercida principalmente por madres hacia menores, en contextos de baja escolaridad.**

**Política propuesta:**

- Crear programas de educación emocional y crianza sin violencia, dirigidos a madres y cuidadores, especialmente en zonas de baja escolaridad.
- Incluir talleres comunitarios sobre manejo del estrés, resolución de conflictos y comunicación afectiva.
- Dado que ya se cuenta con acompañamiento psicosocial en centros comunitarios o instituciones educativas, incrementar el personal de atención de primer nivel.

### Cluster 1 – Adulto Joven y Familiar Cercano (43.76%)

**Violencia entre hermanos(as) o hijos(as) hacia familiares, asociada a intolerancia, machismo y consumo de sustancias.**

**Política propuesta:**

- Implementar campañas nacionales de prevención del machismo y promoción de nuevas masculinidades, con apoyo de medios y redes sociales.
- Fortalecer programas de prevención del consumo de alcohol y drogas en jóvenes, combinando educación emocional y acceso a oportunidades laborales.
- Promover programas escolares y universitarios de gestión emocional, liderazgo positivo y convivencia pacífica.

### Cluster 2 – Paternal e Infantil (21.12%)

**Violencia ejercida por padres hacia hijas menores, con factores de machismo e intolerancia.**

**Política propuesta:**

- Desarrollar programas de educación familiar y crianza responsable, con enfoque en la prevención de la violencia y la igualdad de género.
- Ofrecer intervenciones familiares obligatorias para agresores reincidentes, incluyendo terapia psicológica y talleres de control de impulsos.
- Reforzar la presencia institucional (ICBF, comisarías, etc.) en sectores con alta incidencia de violencia intrafamiliar. (Seguimiento de casos).

---

## Síntesis General

Los resultados muestran que la violencia intrafamiliar está estrechamente ligada a la falta de educación emocional, la desigualdad de género y las limitaciones económicas.

Por tanto, las políticas deben priorizar la educación emocional, la equidad de género, la formación de nuevas masculinidades, crianza responsable, el fortalecimiento del entorno familiar y comunitario.

Más allá de castigar al agresor, el enfoque debe ser preventivo, educativo y brindar nuevas oportunidades para que pueda mejorar su calidad de vida, atacando las causas culturales y sociales que perpetúan la violencia.

---

## Bibliografía

1. Herramientas de predicción de violencia basada en género y feminicidio mediante la Inteligencia Artificial (Roa Avella, M. del P.; Sanabria-Moyano, J. E.; Dinas-Hurtado, K., 2023). Esta investigación aborda la predicción de violencia basada en género y feminicidio con IA en Colombia.

2. Juran, J. M. (1951). Quality Control Handbook. McGraw-Hill.

3. Data Analytics - an overview | ScienceDirect Topics. (s. f.-c). https://www.sciencedirect.com/topics/computer-science/data-analytics

4. ¿Qué es Python? - Explicación del lenguaje Python - AWS. (s. f.). Amazon Web Services, Inc. https://aws.amazon.com/es/what-is/python

5. Team, K. (s. f.). Keras: Deep Learning for humans. https://keras.io/

6. Clark, B. (2025, 16 octubre). What is Scikit-Learn (Sklearn)? IBM. https://www.ibm.com/think/topics/scikit-learn

7. Hernández Sampieri, R., Fernández Collado, C., & Baptista Lucio, P. (2014). Metodología de la investigación (6ª ed.). McGraw-Hill.

8. ¿Qué es la regresión logística? - Explicación del modelo de regresión logística - AWS. (s. f.). Amazon Web Services, Inc. https://aws.amazon.com/es/what-is/logistic-regression/

9. Hosseinzadeh, M., Azhir, E., Ahmed, O. H., Ghafour, M. Y., Ahmed, S. H., Rahmani, A. M., & Vo, B. (s. f.). Data cleansing mechanisms and approaches for big data analytics: a systematic study. Journal Of Ambient Intelligence And Humanized Computing, 14(1), 99-111. https://doi.org/10.1007/s12652-021-03590-2

10. M. Triola, R. Levin & D. Rubin, D. Lind, A. Marchal & W. Wathen, o W. Mendenhall, J. Wackerly & D. D. C. K. Mendenhall. Estadística o Estadística para Administración y Economía

11. Daniel. (2023, 30 octubre). Random Forest: Bosque aleatorio. Definición y funcionamiento. DataScientest. https://datascientest.com/es/random-forest-bosque-aleatorio-definicion-y-funcionamiento

---

## Anexos

### Anexo A: Instalación de Dependencias

Para ejecutar el código de este proyecto, es necesario instalar las siguientes librerías:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kmodes statsmodels
```

**Nota:** `statsmodels` es opcional y se usa para análisis estadísticos avanzados de regresión.

### Anexo B: Estructura de Archivos del Proyecto

```
Proyecto_Tech/
├── intrafamiliar_original.csv          # Dataset original
├── intrafamiliar_limpiofinal.csv       # Dataset limpio (23 columnas)
├── intrafamiliar_modelo.csv            # Dataset para modelado
├── intrafamiliar_modelov2.csv          # Dataset limpio para modelado (9 columnas, incluye Porcentaje de riesgo)
├── grafico_simple.py                   # Script de visualización
└── Modelo_Riesgo_Victimizacion_Violencia_Intrafamiliar.md  # Este documento
```

### Anexo C: Variables del Dataset

#### Variables del Dataset `intrafamiliar_modelov2.csv` (9 columnas):

Este dataset está optimizado para modelado y contiene las siguientes variables:

1. **`Grupo de Edad judicial`** - Rango de edad específico de la víctima (ej: "(00 a 04)", "(05 a 09)", etc.)
2. **`Escolaridad`** - Nivel educativo estandarizado de la víctima
3. **`Departamento del hecho DANE`** - Departamento donde ocurrió el incidente
4. **`Escenario del Hecho`** - Lugar específico donde ocurrió el incidente
5. **`Actividad Durante el Hecho`** - Actividad que realizaba la víctima al momento del hecho
6. **`Sexo del Agresor`** - Sexo del agresor (Hombre, Mujer, etc.)
7. **`Presunto Agresor Detallado`** - Relación específica del agresor con la víctima (Padre, Madre, Hijo(a), etc.)
8. **`Factor Desencadenante de la Agresión`** - Causa o factor que desencadenó la agresión
9. **`Porcentaje de riesgo`** - Variable objetivo: porcentaje de riesgo calculado (0-1 o 0-100)

**Nota:** Este dataset es una versión optimizada y limpia del dataset original, enfocada en las variables más relevantes para el modelado predictivo. La columna `Porcentaje de riesgo` puede ser utilizada directamente como variable objetivo en los modelos de regresión.

---

*Documento generado basado en el modelo de riesgo de victimización por violencia intrafamiliar en Colombia*  
*Fecha: Diciembre 2024*  
*Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF)*
