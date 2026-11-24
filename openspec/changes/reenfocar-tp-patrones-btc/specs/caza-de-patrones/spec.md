# Spec Delta

## Purpose

Busca y describe los patrones de velas que aparecen en Bitcoin (de libro, los del profesor, por fase del ciclo de halving y los que encuentra la máquina sola) y mide qué pasó después de cada uno, sin entrenar con el futuro.

## ADDED Requirements

### Requirement: Patrones clásicos en BTC
El notebook SHALL detectar un conjunto de patrones clásicos de velas (al menos martillo, estrella fugaz, envolvente alcista/bajista, doji) en las velas diarias de BTC y SHALL informar cuántas veces aparece cada uno y qué hizo el precio en los 7 días siguientes.

#### Scenario: Tabla de patrones clásicos
- **WHEN** se ejecuta la detección
- **THEN** se obtiene una tabla por patrón con frecuencia, retorno medio a 7 días, % de veces que subió/bajó/lateral y la comparación contra el promedio de todos los días

#### Scenario: Patrón con pocas apariciones
- **WHEN** un patrón aparece menos de 30 veces
- **THEN** la tabla lo marca como "muestra chica" y no se saca conclusión de él

### Requirement: Planilla del profesor replicada en BTC
El notebook SHALL calcular sobre BTC las mismas medidas de la planilla del profesor (mechas, cuerpo, proporciones, "sharpe" de 3 ruedas, detector HAMMER y P(retorno futuro > umbral | sharpe > 0,9)) y SHALL compararlas con los resultados de LLY.

#### Scenario: Probabilidad condicional
- **WHEN** se calcula P(retorno 3 ruedas > 3% | sharpe > 0,9) en BTC
- **THEN** se muestra junto a la probabilidad incondicional y junto al valor de LLY

### Requirement: Patrones por fase del halving
El notebook SHALL alinear los ciclos con el halving como día 0 y SHALL comparar, por fase del ciclo, la frecuencia de cada tipo de vela y los retornos.

#### Scenario: Ciclos superpuestos
- **WHEN** se grafica el precio normalizado de cada ciclo desde su halving
- **THEN** los ciclos 2012, 2016, 2020 y 2024 aparecen superpuestos con el día actual del ciclo 2024 marcado

#### Scenario: ¿Se repite en la misma fase?
- **WHEN** se compara la distribución de tipos de vela por fase entre ciclos
- **THEN** se reporta si las diferencias entre fases son mayores que las esperables por azar

### Requirement: Familias de velas descubiertas
El notebook SHALL agrupar ventanas cortas de velas normalizadas en familias sin usar el resultado futuro y SHALL mostrar un ejemplo típico de cada familia y qué pasó después en cada período (train/val/test).

#### Scenario: Agrupamiento sin mirar el futuro
- **WHEN** se forman las familias
- **THEN** el agrupamiento se ajusta solo con datos de entrenamiento y las ventanas de validación y test solo se asignan a la familia más cercana

#### Scenario: Familia que se mantiene
- **WHEN** una familia tiene un retorno posterior distinto del promedio en train
- **THEN** se informa si esa diferencia se mantiene en validación y test

### Requirement: Shazam del ciclo actual
El notebook SHALL buscar los momentos del pasado más parecidos a las últimas velas disponibles y SHALL mostrar qué pasó después en esos momentos.

#### Scenario: Búsqueda de parecidos
- **WHEN** se ejecuta el Shazam con las últimas velas
- **THEN** se muestran los 5 momentos más parecidos (fecha, distancia, retorno posterior a 7 días) y la figura con ventana actual y parecidas superpuestas

### Requirement: Resumen de qué esconden las velas
La caza de patrones SHALL cerrar con una tabla resumen: patrón, cuántas veces, qué pasó después, si se sostiene fuera de train.

#### Scenario: Resumen
- **WHEN** termina la sección
- **THEN** existe la tabla resumen y se guarda en resultados
