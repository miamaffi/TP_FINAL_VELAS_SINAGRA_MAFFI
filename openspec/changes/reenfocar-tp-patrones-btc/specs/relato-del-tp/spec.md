# Spec Delta

## Purpose

Define la pregunta de investigación del TP, el orden en que el notebook cuenta la historia, cómo se justifica cada decisión y cómo se presentan los descartes y las conclusiones.

## ADDED Requirements

### Requirement: Pregunta central exploratoria
El notebook SHALL plantear como pregunta central "¿Qué patrones de velas esconde Bitcoin y qué nos dicen?" y SHALL presentar "¿sirven para predecir?" como sub-pregunta, no como objetivo principal.

#### Scenario: Lectura de la sección 1
- **WHEN** alguien lee la sección de la pregunta de investigación
- **THEN** encuentra la pregunta central exploratoria, la escalera de sub-preguntas (incluida la predictiva) y las hipótesis H0/H1 de la parte predictiva

### Requirement: Velas japonesas, CNN y recurrente en el centro
Los modelos predictivos propios SHALL usar como entrada solo velas japonesas diarias de BTC, y el notebook SHALL incluir una CNN que aprende de imágenes de velas y una red recurrente (LSTM) que las lee como secuencia; cada sección de contexto SHALL cerrar conectando lo que muestra con las velas.

#### Scenario: Entrada de los modelos
- **WHEN** se revisa la entrada de la CNN y de la LSTM
- **THEN** solo contiene información derivada de las velas diarias (OHLC y volumen), sin noticias ni eventos

#### Scenario: Sección de contexto
- **WHEN** termina la sección de contexto, hitos o benchmark
- **THEN** su "¿qué nos llevamos?" dice qué implica eso para las velas o para los modelos

### Requirement: Orden del relato
El notebook SHALL seguir este orden: contexto de Bitcoin → pregunta → entorno → base de velas → historia (hitos, eventos, halvings) → benchmark → anatomía y respuesta de cada día → caza de patrones → ¿sirven para predecir? (baselines, Monte Carlo, CNN, LSTM, pre-entrenados, examen, errores) → descartes → conclusiones → guardado de resultados. La historia y el benchmark van después de la base de velas porque se calculan con ella.

#### Scenario: Índice del notebook
- **WHEN** se regenera el notebook y se listan sus títulos de primer nivel
- **THEN** aparecen en ese orden y no hay secciones de modelos antes de la caza de patrones

### Requirement: Cada decisión justificada
Cada decisión de diseño del trabajo (fuente, temporalidad, target, umbral, split, ventana, arquitectura, métrica) SHALL tener al lado un texto que diga qué se decidió y en base a qué.

#### Scenario: Decisión sin justificar
- **WHEN** se revisa una celda de código que fija un parámetro elegido por nosotras
- **THEN** la celda markdown anterior o la tabla de decisiones explica el porqué

### Requirement: Sección de descartes
El notebook SHALL tener una sección que liste lo que se probó y no funcionó (con su resultado en validación) y qué se hizo en su lugar.

#### Scenario: Mejora que no ayudó
- **WHEN** una variante (acciones, criptos, reentreno, híbrida, ResNet, velas semanales) no mejora en validación
- **THEN** aparece en la sección de descartes con su número y el motivo de descarte, en vez de mezclada con los modelos principales

### Requirement: Solo velas diarias en los modelos
Los modelos y las pruebas estadísticas SHALL usar únicamente velas diarias; las velas semanales SHALL usarse solo para visualizar ciclos.

#### Scenario: Comparación diarias vs semanales
- **WHEN** se busca la comparación de velas diarias contra semanales
- **THEN** figura solo en la sección de descartes, como algo probado y abandonado por pedido de los profesores y por falta de datos

### Requirement: Conclusiones calculadas y coherentes
La sección de conclusiones SHALL responder cada sub-pregunta con los números de la corrida y SHALL contrastar el resultado predictivo con la bibliografía.

#### Scenario: Resultado predictivo nulo
- **WHEN** ningún modelo supera al azar tras la corrección por comparaciones múltiples
- **THEN** la conclusión lo dice explícitamente, lo relaciona con la eficiencia de mercado y la bibliografía, y destaca qué sí se aprendió de los patrones

#### Scenario: Corrida rápida
- **WHEN** el notebook corre con el modo rápido activado
- **THEN** las conclusiones se generan igual, marcadas como provisorias
