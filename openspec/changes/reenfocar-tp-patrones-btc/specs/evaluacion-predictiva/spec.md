# Spec Delta

## Purpose

Define cómo se mide si los patrones de velas sirven para predecir sube/baja/lateral a 7 días: contra qué se compara, con qué métricas y con qué pruebas estadísticas, respetando la temporalidad.

## ADDED Requirements

### Requirement: Sin fuga de información del futuro
Toda normalización, umbral, agrupamiento o elección de hiperparámetros SHALL calcularse solo con datos de entrenamiento (o validación para elegir), y entre períodos SHALL haber un embargo de al menos 7 días.

#### Scenario: Umbral de lateral
- **WHEN** se calcula el umbral k del target
- **THEN** se usa solo el período de entrenamiento y el test nunca se mira para elegir nada

### Requirement: Monte Carlo como referencia
El notebook SHALL simular trayectorias de precio con al menos un modelo estándar (paseo aleatorio con la volatilidad reciente) y SHALL convertir esas simulaciones en predicción sube/baja/lateral con la misma regla del target.

#### Scenario: Comparación con la CNN
- **WHEN** se evalúa en test
- **THEN** la tabla de resultados muestra Monte Carlo junto a la CNN, la LSTM y los baselines, con la distancia entre ellos

#### Scenario: Probabilidades de Monte Carlo
- **WHEN** se simulan las trayectorias para un día
- **THEN** la proporción de trayectorias que terminan en cada clase es la probabilidad que da Monte Carlo para ese día

### Requirement: CNN vs recurrente
El notebook SHALL comparar en la misma tabla y con las mismas métricas la CNN sobre imágenes y la LSTM sobre secuencias, explicando que la recurrente es lo más usado en la literatura.

#### Scenario: Tabla CNN vs LSTM
- **WHEN** se muestran los resultados
- **THEN** la CNN y la LSTM aparecen con su resultado en validación y en test

### Requirement: P-valores robustos
Los p-valores de cada modelo SHALL calcularse con una prueba que respete la autocorrelación de targets solapados (permutación por desplazamiento circular) y SHALL acompañarse de un intervalo de confianza por bootstrap por bloques.

#### Scenario: P-valor de un modelo
- **WHEN** se evalúa un modelo en test
- **THEN** la tabla muestra balanced accuracy, IC 95%, p-valor por desplazamiento circular y p-valor corregido

### Requirement: Corrección por comparaciones múltiples
Los p-valores SHALL corregirse por Holm sobre todos los modelos evaluados en test, y un modelo SHALL llamarse "significativo" solo si su p-valor corregido es menor a 0,05.

#### Scenario: Muchos modelos
- **WHEN** se evalúan varios modelos en test
- **THEN** la columna "significativo" usa el p-valor corregido, no el crudo

### Requirement: Inversión simulada para todos
La inversión simulada SHALL calcularse para todos los modelos de la tabla final (incluidos los pre-entrenados y Monte Carlo) y para buy & hold, con comisiones.

#### Scenario: Kronos en la inversión
- **WHEN** se muestra la tabla de inversión
- **THEN** Kronos y Monte Carlo aparecen con su capital final igual que el resto

### Requirement: Clasificación correcta de cada modelo
Cada modelo en la tabla SHALL tener el tipo correcto (baseline, simulación, red propia, mejora descartada, pre-entrenado).

#### Scenario: Variante con criptos
- **WHEN** aparece la variante entrenada con otras criptos
- **THEN** su tipo es "mejora descartada" (o el que corresponda), no "red propia"
