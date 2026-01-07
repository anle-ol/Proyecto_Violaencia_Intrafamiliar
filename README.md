# Modelo de Riesgo de Victimización por Violencia Intrafamiliar en Colombia

## 📋 Introducción

### Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo predictivo que identifique niveles de riesgo de victimización en casos de violencia intrafamiliar en Colombia, utilizando técnicas de aprendizaje automático supervisado y no supervisado sobre datos del Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF) del período 2014-2024.

El proyecto busca transformar datos históricos en conocimiento accionable que sustente decisiones públicas fundamentadas, permitiendo a las instituciones implementar intervenciones preventivas en lugar de mantener un enfoque reactivo.

### Objetivos del Proyecto

#### Objetivo General
Desarrollar un modelo predictivo que identifique el nivel de riesgo de victimización en casos de violencia intrafamiliar en Colombia, a partir de variables sociodemográficas y contextuales suministradas por las víctimas desde 2014 hasta 2024.

#### Objetivos Específicos
1. Recopilar y limpiar datos de violencia intrafamiliar del INMLCF (2014-2024)
2. Realizar análisis exploratorio de datos (EDA)
3. Implementar técnicas de clustering (K-Modes) para identificar perfiles de riesgo
4. Desarrollar modelos predictivos de machine learning para clasificar niveles de riesgo
5. Implementar modelos supervisados de regresión para predecir scores de riesgo continuos
6. Validar y evaluar el rendimiento de los modelos desarrollados
7. Generar visualizaciones y reportes interpretables
8. Proponer recomendaciones de políticas públicas basadas en los hallazgos

### Dataset

#### Datos Utilizados
- **Fuente:** Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF)
- **Período:** 2014-2024
- **Registros totales:** 236,840 casos
- **Columnas originales:** 36 variables

#### Variables Principales
1. **Grupo de Edad judicial** - Rango de edad de la víctima
2. **Escolaridad** - Nivel educativo
3. **Sexo del Agresor** - Hombre, Mujer, Otros
4. **Presunto Agresor Detallado** - Relación específica (Padre, Madre, Hijo(a), etc.)
5. **Factor Desencadenante de la Agresión** - Causa principal
6. **Escenario del Hecho** - Lugar donde ocurrió
7. **Actividad Durante el Hecho** - Actividad de la víctima
8. **Departamento del hecho DANE** - Ubicación geográfica
9. **Porcentaje de riesgo** - Variable objetivo calculada

### Tecnologías Utilizadas

#### Librerías de Python
- **pandas** - Manipulación y análisis de datos
- **numpy** - Cálculos numéricos
- **matplotlib** - Visualizaciones básicas
- **seaborn** - Visualizaciones estadísticas avanzadas
- **scikit-learn** - Machine learning (clasificación, regresión, métricas)
- **kmodes** - Clustering para variables categóricas
- **catboost** - Modelo de gradient boosting para datos categóricos
- **scipy** - Estadísticas y pruebas (chi2_contingency, etc.)

---

## 🧹 Limpieza de Datos

### Notebook: `Violencia_intrafamiliar_V2.ipynb`

**Objetivo:** Limpieza, preparación y transformación de datos del dataset original

#### Proceso de Limpieza

1. **Carga de Datos Iniciales**
   - Dataset original con 236,840 registros y 36 columnas
   - Identificación de variables categóricas y numéricas
   - Revisión de valores faltantes y datos inconsistentes

2. **Renombramiento y Estandarización de Categorías**
   Se normalizaron y consolidaron categorías en múltiples variables:
   - **Grupo Mayor Menor de Edad:** Unificación de diferentes formatos de texto
   - **Escolaridad:** Consolidación de categorías similares (ej: "Básica primaria" → "Primaria")
   - **Estado Civil:** Estandarización de formatos
   - **Tipo de Discapacidad:** Normalización de nombres
   - **Pertenencia Grupal:** Agrupación de categorías relacionadas (ej: diferentes formas de referirse a la comunidad LGBT)
   - **Mes y Día del hecho:** Capitalización consistente
   - **Escenario del Hecho:** Normalización de nombres extensos y descriptivos
   - **Actividad Durante el Hecho:** Limpieza de puntuación y formatos
   - **Presunto Agresor Detallado:** Estandarización de paréntesis y formatos
   - **Factor Desencadenante de la Agresión:** Consolidación de categorías similares
   - **Días de Incapacidad Medicolegal:** Normalización de valores
   - **Pueblo Indígena:** Mapeo de valores "Ninguno" a "No aplica"

3. **Transformación de Variables**

   - **Severidad Ordinal:** 
     - Conversión de "Días de Incapacidad Medicolegal" a escala ordinal (0-3)
     - Mapeo: 'Cero' → 0, '1 a 30' → 1, '31 a 90' → 2, 'Más de 90' → 3
     - Exclusión de registros con 'Sin información'
   
   - **Riesgo Extremo Grave:**
     - Variable binaria creada a partir de Severidad Ordinal
     - Valor 1 si severidad ≥ 2, 0 en caso contrario
   
   - **Factor Desencadenante:**
     - Aplicación de One-Hot Encoding (OHE)
     - Creación de columnas binarias para cada factor
     - Exclusión de registros con 'Sin información'
   
   - **Presunto Agresor:**
     - Agrupación en 4 clases principales:
       - Familiar Directo (Padre, Madre, Hijo(a), Hermano(a))
       - Familiar Extendido (Abuelo(a), Tío(a), Primo(a), etc.)
       - Cuidado/Tutela (Encargado del cuidado, Madrastra, Padrastro)
       - Otros/Bajo Riesgo (Profesor(a), etc.)
     - Aplicación de One-Hot Encoding (OHE)
     - Exclusión de registros con 'Sin información'
   
   - **Escolaridad:**
     - Conversión a escala ordinal (0-6)
     - Mapeo: 'Sin escolaridad' → 0, 'Preescolar' → 1, 'Primaria' → 2, 'Secundaria' → 3, 'Técnica o Tecnológica' → 4, 'Profesional' → 5, 'Posgrado' → 6
     - Exclusión de registros con 'Sin información'
   
   - **Pueblo Indígena:**
     - Conversión a variable binaria (0/1)
     - Valor 1 para pertenencia a cualquier pueblo indígena, 0 en caso contrario

4. **Análisis de Correlaciones**
   - Cálculo de asociaciones entre variables
   - Análisis de correlaciones entre factores desencadenantes y severidad
   - Análisis de correlaciones entre presunto agresor y severidad
   - Análisis de correlaciones entre escolaridad y severidad

5. **Limpieza Final**
   - Exclusión de registros con información faltante crítica
   - Validación de integridad de datos
   - Exportación de datasets limpios

#### Outputs Generados
- `intrafamiliar_modified.csv` - Dataset modificado con categorías estandarizadas
- `intrafamiliar_limpiofinal.csv` - Dataset limpio final (23 columnas) listo para análisis

---

## 📊 Análisis con Metodología Pareto

### Notebook: `violencia_intrafamiliar_pareto_final.ipynb`

**Objetivo:** Análisis de riesgo utilizando metodología Pareto y clasificación ABC

#### Funciones Personalizadas Implementadas

1. **Análisis de Asociaciones Categóricas**
   - **V de Cramer:** Para medir asociación categórica-categórica (0 = sin asociación, 1 = asociación perfecta)
   - **Ratio de Correlación (η):** Para medir asociación categórica-numérica
   - **Matrices de Correlación:** Para variables categóricas y mixtas
   - **Visualizaciones:** Mapas de calor para visualizar asociaciones

2. **Función de Asignación Pareto**
   - Clasificación ABC basada en frecuencia acumulada
   - Asignación de valores según límites: < 70% (valor 3), 70-90% (valor 2), ≥ 90% (valor 1)
   - Permite identificar las categorías más frecuentes y relevantes

3. **Función de Ojivas**
   - Graficación de polígonos de frecuencia acumulada
   - Visualización de distribución acumulativa de variables

#### Variables Procesadas con Pareto

Se aplicó la metodología Pareto a 8 variables clave:

1. **Grupo de Edad judicial** - Clasificación ABC basada en frecuencia de rangos etarios
2. **Escolaridad** - Priorización de niveles educativos más frecuentes
3. **Departamento del hecho DANE** - Identificación de departamentos con mayor incidencia
4. **Escenario del Hecho** - Clasificación de escenarios más comunes
5. **Actividad Durante el Hecho** - Priorización de actividades durante la agresión
6. **Sexo del Agresor** - Clasificación por género del agresor
7. **Presunto Agresor Detallado** - Priorización de relaciones agresor-víctima
8. **Factor Desencadenante de la Agresión** - Identificación de factores más frecuentes

#### Proceso de Cálculo de Riesgo

1. **Clasificación Pareto Individual**
   - Cada variable recibe valores 3, 2 o 1 según su frecuencia acumulada
   - Valores mayores indican mayor frecuencia/riesgo

2. **Cálculo de Total Pareto**
   - Suma de todos los valores Pareto individuales
   - Rango teórico: 8-24 (8 variables × 3 = máximo, 8 variables × 1 = mínimo)

3. **Cálculo de Porcentaje de Riesgo**
   - Normalización del Total Pareto a escala 0-1
   - Fórmula: `(Total Pareto - 10) / (24 - 10)`
   - Permite comparar niveles de riesgo entre casos

#### Análisis y Visualizaciones

1. **Mapas de Calor por Pares de Variables**
   - Visualización de combinaciones de variables Pareto
   - Identificación de patrones de riesgo combinados
   - Análisis de todas las combinaciones posibles entre variables

2. **Análisis de Combinaciones**
   - Matriz de porcentajes de combinaciones con valor máximo (6)
   - Identificación de pares de variables que frecuentemente alcanzan valores altos de riesgo

3. **Heatmap de Correlaciones Pareto**
   - Visualización de correlaciones entre variables Pareto
   - Identificación de variables fuertemente asociadas

#### Output Generado
- `intrafamiliar_modelo.csv` - Dataset final para modelado (14 columnas) con variables originales y variable `Porcentaje de riesgo` calculada
- Dataset incluye 13 variables seleccionadas más `Porcentaje de riesgo` como variable objetivo

---

## 📈 Modelos de Regresión

### Notebook: `modelocatboost.ipynb`

**Objetivo:** Implementación de modelo de regresión usando CatBoost para predecir el porcentaje de riesgo

#### Modelo CatBoost Regressor

**Características del Modelo:**
- **Algoritmo:** CatBoostRegressor
- **Variables de entrada:** Todas las variables categóricas del dataset
- **Variable objetivo:** `Porcentaje de riesgo` (escala 0-1)
- **Manejo de variables categóricas:** Automático (sin necesidad de encoding manual)

#### Configuración del Modelo
- **Iteraciones:** 1000
- **Learning rate:** 0.05
- **Profundidad máxima:** 6
- **Función de pérdida:** RMSE (Root Mean Squared Error)
- **División train/test:** 80/20 con estratificación por variable objetivo
- **Semilla aleatoria:** 42 (para reproducibilidad)

#### Resultados del Modelo

**Métricas de Evaluación:**
- **RMSE:** 0.0107 (error cuadrático medio muy bajo)
- **R² Score:** 0.9936 (excelente capacidad predictiva - 99.36% de varianza explicada)

**Interpretación:**
- El modelo muestra una capacidad excepcional para predecir el porcentaje de riesgo
- Un R² de 0.9936 indica que el modelo explica el 99.36% de la variabilidad en la variable objetivo
- El RMSE de 0.0107 indica predicciones muy precisas con un error promedio muy bajo

#### Funcionalidades Implementadas
- Predicción sobre conjunto de prueba
- Predicción sobre nuevas muestras individuales
- Comparación de valores reales vs predichos
- Visualización de diferencias entre predicciones y valores reales

#### Variables Utilizadas
- Grupo de Edad judicial
- Escolaridad
- Departamento del hecho DANE
- Escenario del Hecho
- Actividad Durante el Hecho
- Sexo del Agresor
- Presunto Agresor Detallado
- Factor Desencadenante de la Agresión

**Ventaja de CatBoost:**
- Manejo automático de variables categóricas sin necesidad de encoding manual
- Optimizado para datasets con muchas variables categóricas
- Previene overfitting mediante técnicas avanzadas de regularización

---

## 🎯 Modelo de Clasificación (Clustering)

### Notebook: `modelo_kmodes.ipynb`

**Objetivo:** Implementación de clustering no supervisado usando el algoritmo K-Modes para identificar perfiles de riesgo

#### Algoritmo K-Modes

**Características:**
- **Algoritmo:** K-Modes (adaptación de K-Means para variables categóricas)
- **Número de clusters:** 3 (identificados como óptimos)
- **Inicialización:** Método Huang
- **Iteraciones:** 5 inicializaciones diferentes para encontrar el mejor resultado
- **Semilla aleatoria:** 42 (para reproducibilidad)

#### Proceso de Clustering

1. **Preparación de Datos**
   - Selección de variables categóricas relevantes
   - Codificación con LabelEncoder para todas las variables categóricas
   - Conversión a formato numérico compatible con K-Modes

2. **Aplicación del Algoritmo**
   - Ejecución de 5 inicializaciones diferentes
   - Selección del resultado con menor costo (distancia total)
   - Asignación de cada caso a uno de los 3 clusters

3. **Análisis de Clusters**
   - Cálculo de estadísticas descriptivas por cluster
   - Identificación de modas (valores más frecuentes) en cada cluster
   - Análisis de distribución de casos entre clusters

#### Perfiles de Riesgo Identificados

##### **Cluster 0 - Perfil Matriarcal y de Menores**
- **Proporción:** 49,405 casos (35.12%)
- **Características principales:**
  - **Sexo del Agresor:** Mayoritariamente Mujer (Madre)
  - **Grupo de Edad judicial:** Población infantil/adolescente (14-17 años)
  - **Escolaridad:** Primaria
  - **Presunto Agresor Detallado:** Madre
  - **Factor Desencadenante:** Intolerancia o Machismo
  - **Escenario:** Vivienda
  - **Actividad:** Actividades Vitales / Cuidado Personal
  - **Departamento:** Bogotá, D.C.
  - **Porcentaje de riesgo promedio:** 0.8300

**Interpretación:** Este cluster identifica violencia ejercida principalmente por madres hacia menores, con un claro enfoque en el riesgo de agresión materna en el hogar, especialmente en contextos de baja escolaridad.

##### **Cluster 1 - Perfil Adulto Joven y Familiar Cercano**
- **Proporción:** 61,566 casos (43.76%)
- **Características principales:**
  - **Sexo del Agresor:** Mayoritariamente Hombre
  - **Grupo de Edad judicial:** Adultos jóvenes (20-24 años)
  - **Escolaridad:** Secundaria
  - **Presunto Agresor Detallado:** Hermano(a)
  - **Factor Desencadenante:** Intolerancia o Machismo
  - **Escenario:** Vivienda
  - **Actividad:** Trabajo Doméstico No Remunerado
  - **Departamento:** Bogotá, D.C.
  - **Porcentaje de riesgo promedio:** 0.8421

**Interpretación:** Este cluster abarca las agresiones perpetradas por hombres (o en menor medida hermanos/hijos) en un rango de edad adulta joven, lo que puede estar ligado a dinámicas de convivencia y tensión económica o social en el hogar.

##### **Cluster 2 - Perfil Paternal e Infantil**
- **Proporción:** 29,711 casos (21.12%)
- **Características principales:**
  - **Sexo del Agresor:** Hombre (Padre)
  - **Grupo de Edad judicial:** Población infantil más joven (10-13 años)
  - **Escolaridad:** Primaria
  - **Presunto Agresor Detallado:** Padre
  - **Factor Desencadenante:** Intolerancia o Machismo
  - **Escenario:** Vivienda
  - **Actividad:** Actividades Vitales / Cuidado Personal
  - **Departamento:** Bogotá, D.C.
  - **Porcentaje de riesgo promedio:** 0.8434

**Interpretación:** Este cluster identifica la violencia intrafamiliar ejercida por hombres (padres), dirigida específicamente a niños y adolescentes más jóvenes, con un claro enfoque en el riesgo de agresión paterna en el hogar.

#### Output Generado
- `intrafamiliar_clusters.csv` - Dataset con asignación de clusters para cada caso
- Incluye todas las variables originales más la columna `cluster` con valores 0, 1 o 2

---

## 🔍 Hallazgos y Conclusiones

### Hallazgos Principales

#### 1. Perfiles de Riesgo Identificados

El análisis de clustering reveló **3 perfiles distintos de violencia intrafamiliar**:

- **Perfil más frecuente (43.76%):** Adultos jóvenes agredidos por hermanos/as o hijos, principalmente hombres, en contextos domésticos. El factor desencadenante predominante es la intolerancia o machismo.

- **Segundo perfil más frecuente (35.12%):** Menores agredidos por sus madres, concentrados en población infantil/adolescente con baja escolaridad. También relacionado con intolerancia y machismo.

- **Tercer perfil (21.12%):** Niños más pequeños agredidos por sus padres, con características similares al segundo perfil pero con el padre como agresor principal.

#### 2. Factores Desencadenantes Comunes

**Intolerancia o Machismo** es el factor desencadenante más frecuente en los **3 clusters**, seguido por:
- Consumo de alcohol y/o sustancias psicoactivas
- Problemas económicos y de convivencia

#### 3. Características Demográficas

- **Ubicación:** Bogotá, D.C. concentra la mayor cantidad de casos en todos los perfiles
- **Escenario:** La Vivienda es el lugar más frecuente donde ocurren los hechos
- **Escolaridad:** Baja escolaridad (Primaria o Sin escolaridad) está presente en los perfiles de menores

#### 4. Rendimiento de Modelos

**Modelo de Regresión (CatBoost):**
- R² Score de **0.9936** demuestra una capacidad excepcional para predecir el porcentaje de riesgo
- RMSE de **0.0107** indica predicciones muy precisas
- El modelo puede ser utilizado de manera confiable para predecir niveles de riesgo en nuevos casos

**Modelo de Clustering (K-Modes):**
- Identificación clara de 3 perfiles distintos y bien definidos
- Cada cluster muestra características demográficas y contextuales consistentes
- Los perfiles identificados son interpretables y accionables para políticas públicas

#### 5. Metodología Pareto

- La aplicación de metodología Pareto permitió:
  - Identificar las categorías más frecuentes y relevantes en cada variable
  - Crear un score de riesgo combinado basado en múltiples factores
  - Priorizar las variables más importantes para el análisis

### Conclusiones

1. **Violencia en el Núcleo Familiar:** Los resultados confirman que la violencia intrafamiliar en Colombia está comprendida mayormente dentro del núcleo familiar cercano (padres, madres, hijos, hermanos). Los agresores más frecuentes son familiares directos en todos los perfiles identificados.

2. **Factores Culturales y Sociales:** La intolerancia y el machismo emergen como el factor desencadenante más frecuente en todos los perfiles, seguido por problemas relacionados con consumo de sustancias. Esto plantea la necesidad de políticas que aborden tanto el desarrollo educativo y económico, como la educación emocional y familiar.

3. **Vulnerabilidad de Menores:** Dos de los tres perfiles (representando el 56.24% de los casos) involucran a menores de edad, principalmente en rangos de 10-17 años. Esto resalta la necesidad urgente de programas de protección infantil y prevención de violencia hacia menores.

4. **Baja Escolaridad como Factor de Riesgo:** La mayoría de los casos en perfiles de menores están asociados con baja escolaridad (Primaria o Sin escolaridad), lo que sugiere que la educación no solo afecta el desarrollo económico, sino también las habilidades emocionales y la capacidad de resolver conflictos de manera pacífica.

5. **Utilidad del Modelo Predictivo:** El modelo de regresión desarrollado muestra una capacidad excepcional para predecir niveles de riesgo, lo que permite su implementación práctica para:
   - Identificación temprana de casos de alto riesgo
   - Priorización de intervenciones preventivas
   - Asignación eficiente de recursos institucionales

### Recomendaciones de Políticas Públicas

#### Para el Cluster 0 (Matriarcal y de Menores - 35.12%)
- Crear programas de educación emocional y crianza sin violencia, dirigidos a madres y cuidadores, especialmente en zonas de baja escolaridad
- Incluir talleres comunitarios sobre manejo del estrés, resolución de conflictos y comunicación afectiva
- Incrementar el personal de atención de primer nivel en centros comunitarios e instituciones educativas

#### Para el Cluster 1 (Adulto Joven y Familiar Cercano - 43.76%)
- Implementar campañas nacionales de prevención del machismo y promoción de nuevas masculinidades
- Fortalecer programas de prevención del consumo de alcohol y drogas en jóvenes
- Promover programas escolares y universitarios de gestión emocional, liderazgo positivo y convivencia pacífica

#### Para el Cluster 2 (Paternal e Infantil - 21.12%)
- Desarrollar programas de educación familiar y crianza responsable, con enfoque en la prevención de la violencia y la igualdad de género
- Ofrecer intervenciones familiares obligatorias para agresores reincidentes, incluyendo terapia psicológica y talleres de control de impulsos
- Reforzar la presencia institucional (ICBF, comisarías, etc.) en sectores con alta incidencia de violencia intrafamiliar

### Síntesis General

Los resultados muestran que la violencia intrafamiliar está estrechamente ligada a la falta de educación emocional, la desigualdad de género y las limitaciones económicas. Por tanto, las políticas deben priorizar:

- **Educación emocional** desde temprana edad
- **Equidad de género** y promoción de nuevas masculinidades
- **Formación de familias** en crianza responsable
- **Fortalecimiento del entorno** familiar y comunitario

Más allá de castigar al agresor, el enfoque debe ser **preventivo, educativo y brindar nuevas oportunidades** para mejorar la calidad de vida, atacando las causas culturales y sociales que perpetúan la violencia.

---

## 📁 Estructura de Archivos del Proyecto

### Archivos de Documentación

- `Modelo_Riesgo_Victimizacion_Violencia_Intrafamiliar.md` - Documentación completa del proyecto con metodología detallada
- `Modelo de Riesgo de Victimización por Violencia Intrafamiliar en Colombia (1).pdf` - Documentación en formato PDF

### Notebooks de Jupyter

1. `Violencia_intrafamiliar_V2.ipynb` - Limpieza y preparación de datos
2. `violencia_intrafamiliar_pareto_final.ipynb` - Análisis con metodología Pareto
3. `modelo_kmodes.ipynb` - Clustering no supervisado (K-Modes)
4. `modelocatboost.ipynb` - Modelo de regresión (CatBoost)

### Archivos CSV Generados

- `intrafamiliar_modified.csv` - Dataset modificado con categorías estandarizadas
- `intrafamiliar_limpiofinal.csv` - Dataset limpio final (23 columnas)
- `intrafamiliar_modelo.csv` - Dataset final para modelado (14 columnas, incluye Porcentaje de riesgo)
- `intrafamiliar_modelov2.csv` - Dataset optimizado para modelado (9 columnas)
- `intrafamiliar_clusters.csv` - Dataset con asignación de clusters

**Nota:** Estos archivos CSV deben ser generados ejecutando los notebooks en el orden recomendado o estar disponibles previamente.

### Orden de Ejecución Recomendado

1. `Violencia_intrafamiliar_V2.ipynb` - Limpieza y preparación de datos
2. `violencia_intrafamiliar_pareto_final.ipynb` - Análisis Pareto y creación de variables de riesgo
3. `modelo_kmodes.ipynb` - Clustering no supervisado
4. `modelocatboost.ipynb` - Modelo predictivo de regresión

---

## 👥 Autores

Elaborado por:
- Angie Alejandra Olarte Varga
- Santiago Mayorga Carvajal
- Sandra Milena Alzate Alzate
- Julia Carolina Torres Lozano
- Óscar Alfredo Gómez Sanchez
- Astrid Viviana Naranjo Abril
- Laura Milena Gutiérrez Bustos

---

## 📚 Referencias

Para más detalles sobre la metodología, resultados y propuestas de políticas públicas, consultar el archivo:
- `Modelo_Riesgo_Victimizacion_Violencia_Intrafamiliar.md`

---

## ⚠️ Consideraciones Éticas

Este proyecto utiliza datos sensibles sobre violencia intrafamiliar. Los análisis y modelos están orientados a:
- Prevenir casos futuros mediante identificación temprana de factores de riesgo
- Informar políticas públicas preventivas
- Proteger la privacidad de las víctimas (los datos utilizados son anónimos)

---

## 📄 Licencia

Proyecto de investigación académica para el programa TalentoTech.

---

*Última actualización: Diciembre 2024*
