# Spec Delta

## Purpose

Ubica a Bitcoin en su historia (eventos que movieron el precio) y lo compara con otros mercados, para justificar por qué es el activo elegido y cómo se ven sus velas frente a las de otros activos.

## ADDED Requirements

### Requirement: Línea de tiempo de eventos
El notebook SHALL mostrar un gráfico del precio diario de BTC (escala log) con los halvings y una lista curada de eventos (hackeos, regulaciones, ETFs, quiebras, etc.) marcados en su fecha, cada uno con una fuente citada.

#### Scenario: Evento marcado
- **WHEN** se mira la línea de tiempo
- **THEN** cada evento tiene fecha, título corto y fuente, y se ve el movimiento del precio alrededor

#### Scenario: Reacción del precio a los eventos
- **WHEN** se calcula el retorno de BTC en los 7 días posteriores a cada evento
- **THEN** se muestra una tabla con evento, fecha y retorno a 7 días, comparado con la distribución de retornos a 7 días de cualquier día

### Requirement: Hitos de subidas y caídas fuertes
El notebook SHALL detectar en las velas diarias, sin elegirlos ni fijar su cantidad a mano, todos los tramos en que el precio de cierre se duplicó o se cortó a la mitad antes de revertirse en la misma medida, y SHALL explicar cada uno con el contexto de ese momento y su fuente.

#### Scenario: Tabla de hitos
- **WHEN** se ejecuta la sección
- **THEN** hay una tabla con fecha de inicio, fecha de fin, días, precio inicial y final, % de movimiento, tipo (subida/caída), contexto en 1-2 líneas y fuente, y los hitos aparecen marcados en el gráfico de precio

#### Scenario: Hito sin contexto cargado
- **WHEN** aparece un hito que no tiene contexto escrito (por ejemplo, porque entraron datos nuevos)
- **THEN** se muestra igual con "contexto pendiente" y un aviso, sin cortar la ejecución

#### Scenario: Contexto redactado con cuidado
- **WHEN** se lee el contexto de un hito
- **THEN** está redactado como "contexto que coincidió con el movimiento", nunca como causa probada

### Requirement: Las velas antes de cada hito
Para cada hito, el notebook SHALL mostrar las 20 velas japonesas diarias anteriores a su inicio y SHALL indicar qué predecían la CNN y la LSTM esos días, cuando hay predicción disponible.

#### Scenario: ¿Las velas lo anunciaban?
- **WHEN** se mira un hito
- **THEN** se ven sus 20 velas previas dibujadas y, si caen en validación o test, la clase que dieron la CNN y la LSTM para esos días comparada con lo que pasó

#### Scenario: Hito anterior al período evaluado
- **WHEN** un hito cae en el período de entrenamiento o antes de 2013
- **THEN** se muestran las velas igual y se aclara que no hay predicción fuera de muestra para ese tramo

### Requirement: Benchmark contra otros activos
El notebook SHALL comparar BTC con al menos: otras criptos (ETH, LTC, XRP), S&P 500, Nasdaq, oro y LLY, en un período común, usando velas diarias.

#### Scenario: Tabla comparativa
- **WHEN** se ejecuta la sección de benchmark
- **THEN** se obtiene una tabla por activo con rendimiento anual, volatilidad anual, peor caída, proporción de días con mechas largas y tamaño medio del cuerpo de vela

#### Scenario: Fuente de datos no disponible
- **WHEN** falla la descarga de un activo
- **THEN** ese activo se omite con un aviso visible y el resto de la sección se ejecuta igual

### Requirement: Por qué Bitcoin
La sección de benchmark SHALL terminar con un párrafo que justifique la elección de BTC con los números de la tabla.

#### Scenario: Justificación
- **WHEN** se lee el cierre del benchmark
- **THEN** la justificación cita al menos dos métricas de la tabla (por ejemplo, volatilidad y cantidad de datos 24/7)

### Requirement: Velas de otros activos a la vista
El benchmark SHALL incluir una figura con un tramo de velas diarias de varios activos lado a lado en la misma escala relativa.

#### Scenario: Comparación visual
- **WHEN** se mira la figura
- **THEN** se distinguen las diferencias de tamaño de cuerpos y mechas entre BTC y los otros activos
