# Diseño

## Contexto

- Todo el trabajo vive en `src/generar_notebook.py` (≈2.600 líneas, funciones `md()` y `code()` que van agregando celdas), que genera `TP_Velas_Bitcoin.ipynb`. Se edita el generador y se regenera; el `.ipynb` nunca se toca a mano.
- Se ejecuta solo en Google Colab (GPU T4 para la corrida completa). Claude no corre el notebook: Mia lo corre y manda el zip de `resultados/`.
- Orden actual: 0 contexto · 1 pregunta · 2 entorno · 3 dataset · 4 exploración visual · 5 target/split · 6 baselines · 7 CNN · 8 LSTM · 9 mejoras · 10 pre-entrenados · 11 examen · 12 errores · 13 conclusiones · zip.
- Ya existen y se reutilizan: descarga con yfinance (`bajar_accion`, 9.1), descarga de criptos Bitstamp (`bajar_cripto`, 9.1b), anatomía de vela de la planilla (5.1), `foto_mas_parecida` (base del Shazam), `MODO_RAPIDO`, `guardar_fig` / `guardar_tabla` / `anotar`.
- Ver `proposal.md` (Por qué) y las specs para los requisitos.

## Objetivos / No objetivos

**Objetivos:**
- Reordenar el notebook según el nuevo relato **sin reentrenar distinto**: los modelos, el target, el split y las semillas quedan iguales, así los números de la corrida del 15-nov siguen siendo comparables.
- Agregar las secciones nuevas (línea de tiempo, benchmark, caza de patrones, Monte Carlo, descartes) con costo de cómputo bajo.
- Que cada sección termine con un "¿qué nos llevamos?" corto, en lenguaje claro, que después pueda reutilizar la landing.

**No objetivos:**
- La app/landing (siguiente change).
- Nuevas arquitecturas de redes o nuevos pre-entrenados.
- Optimizar hiperparámetros para "rescatar" algún modelo.

## Decisiones

### 1. Nueva numeración de secciones (como quedó implementada)
0 Contexto · 1 Pregunta · 2 Entorno (constantes y MODO_RAPIDO) · 3 Base de velas · 4 Historia: hitos, eventos, halvings · 5 Benchmark · 6 Anatomía + respuesta + split (ex-5) · 7 Caza de patrones · 8 Baselines + Monte Carlo (ex-6) · 9 CNN (ex-7) · 10 LSTM (ex-8) · 11 Pre-entrenados (ex-10.1-10.3, 10.5) · 12 Examen (ex-11) · 13 Errores (ex-12) · 14 Descartes (ex-7.4b, 9.x, 10.4, 11.7) · 15 Conclusiones · zip. La caza de patrones va después de la sección 6 porque necesita la respuesta de cada día (z, clase) y el split.
- *Por qué el benchmark va después del dataset:* necesita las velas diarias de BTC ya armadas.
- *Por qué la caza de patrones va antes del target:* lo descriptivo no necesita el split, salvo las familias, que usan las fechas del split (las fechas de los halvings y del split se definen como constantes en el entorno, así que se pueden usar antes).
- *Alternativa descartada:* dejar el orden y solo agregar secciones al final. No contesta el pedido de "historia" de los profes.

### 2. Los descartes se siguen ejecutando
Las mejoras (acciones, criptos, reentreno, híbrida), ResNet y diarias vs semanales **se siguen corriendo** (para que sus números sean reales), pero en la sección 10 "Lo que probamos y descartamos", con una tabla: qué se probó, por qué se probó, resultado en validación y por qué se descartó. Al examen final solo entran los "candidatos": baselines, Monte Carlo, CNN, LSTM, Kronos y Chronos-2. Los descartados se evalúan en su propia tabla (14.7), con validación vs base y test por transparencia.
- *Alternativa:* borrarlos. Se pierde la evidencia de "lo intentamos", que los profes pidieron explícitamente.
- Consecuencia: el orden de ejecución ya no coincide con el de lectura (los descartes dependen de funciones de 7). Se resuelve poniendo la sección 10 después del examen y que el examen muestre los descartados solo si existen (`if nombre in PREDICCIONES`).

### 3. Línea de tiempo: lista curada a mano en el generador
~15 eventos con fecha, título y URL de fuente en una lista de Python (Mt. Gox 2014, hackeo de Bitstamp ene-2015, ICOs y pico 2017, COVID mar-2020, Tesla feb-2021, prohibición en China 2021, Luna/3AC/FTX 2022, ETFs spot ene-2024, halvings, pico oct-2025, etc.). Retorno a 7 días después de cada evento contra la distribución de todos los días (percentil).
- *Alternativa:* una API de noticias. Es frágil, necesita clave y es difícil de justificar.
- Mia revisa y aprueba la lista de eventos antes de implementarla (tarea explícita).

### 3b. Hitos: zigzag con umbral "se duplicó o se cortó a la mitad"
- **Detección:** zigzag sobre el log del precio de cierre diario con umbral ln 2. Un tramo se confirma cuando el precio se revierte un factor 2 desde el extremo en curso. Así la cantidad de hitos la decide el dato (con los datos hasta nov-2026 da 17 tramos, de ago-2011 a jun-2026). Va en pura pandas/numpy, sin dependencias.
- **Por qué ln 2:** es simétrico en escala log (+100% y −50% son el mismo tamaño de movimiento), se explica en una frase y funciona igual con $10 que con $100.000.
- **Alternativas descartadas:** un top N fijo (arbitrario, deja afuera tramos como abr-jul 2021 o la caída actual) y un umbral de 40% (da 35 tramos y mete ruido de 2011-2012).
- **Contexto:** diccionario curado (fechas del tramo → texto + URL). El borrador se arma con la lista de Gemini que trajo Mia más los tramos que a esa lista le faltan (abr-2013, rally 2019, abr-jul 2021, caída 2025-26). Ningún texto entra sin una fuente verificada. Mia lo aprueba.
- **Velas previas:** para cada hito se dibujan las 20 velas anteriores con la misma función de dibujo de la CNN (ventana en B/N), para que se vea "lo que vio la red". Si el tramo cae en validación o test, se cruzan con `PREDICCIONES` de la CNN y la LSTM. Como las predicciones existen recién después de entrenar, la galería se muestra en dos partes: en la sección 0 solo las velas, y en el examen la misma galería con lo que dijeron la CNN y la LSTM. Así no se rompe el orden de lectura.
- **Riesgo:** explicar a posteriori es fácil. → Se redacta como "contexto que coincidió", y eso alimenta la conclusión: si las causas fueron noticias imprevisibles, no podían estar en las velas.

### 4. Benchmark con yfinance
Tickers: `BTC-USD` no (BTC sale de Bitstamp), `ETH-USD`, `LTC-USD`, `XRP-USD` (o Bitstamp si ya están bajados en 9.1b), `^GSPC`, `^IXIC`, `GC=F` (oro), `LLY`. Período común: desde 2017-11 (cuando yfinance tiene ETH) hasta el último cierre. Métricas: CAGR, volatilidad anualizada (√365 para cripto, √252 para mercados tradicionales; se aclara), máxima caída, % de días con mecha > 2× cuerpo, cuerpo medio / precio. Cada descarga va en un try/except con aviso, según la spec.

### 5. Caza de patrones
- **Clásicos:** reglas propias en pandas (martillo, estrella fugaz, envolventes, doji) en vez de TA-Lib, porque TA-Lib necesita compilar C en Colab y ya tenemos la anatomía de la vela calculada. Las reglas se documentan en una tabla con la definición del manual de Nison.
- Retorno posterior a 7 días: se compara con el de todos los días mediante una diferencia de medias, con un IC por bootstrap por bloques. Se marca "muestra chica" si hay menos de 30 casos.
- **Planilla del profe:** se reutilizan las columnas de 5.1 y se agrega el "sharpe" de 3 ruedas. LLY se baja con yfinance para el mismo cálculo y se ponen lado a lado.
- **Por fase del halving:** se divide cada ciclo en 4 fases (0-25%, 25-50%, 50-75% y 75-100% de ~1.460 días). Se hace una tabla de contingencia de tipo de vela × fase y una prueba χ². El gráfico de ciclos superpuestos se arma con el precio normalizado a 1 el día del halving.
- **Familias:** KMeans (k elegido por silueta entre 4 y 10) sobre ventanas de 5 velas normalizadas (las mismas features por vela que la LSTM: retorno, cuerpo, mechas en unidades de volatilidad), ajustado solo con train. Se muestra el "centroide" dibujado como velas. Por familia: frecuencia y retorno posterior en train/val/test.
- **Shazam:** se reutiliza `foto_mas_parecida` con las últimas N velas contra todo el historial anterior. Se excluyen los vecinos que se solapan con la consulta.

### 6. Monte Carlo
Paseo aleatorio geométrico: para cada día t, 1.000 trayectorias de 7 días con retornos normales con media 0 y desvío igual a la volatilidad de 30 días hasta t. Se cuenta qué fracción termina en cada clase con el mismo umbral k. Clase predicha = la más probable. Variante con bootstrap de retornos históricos (solo los anteriores a t) para tener colas gordas.
- Con media 0 y el umbral k calibrado (≈⅓ lateral en train), el paseo aleatorio da ≈35% sube / 30% lateral / 35% baja todos los días: su "clase más probable" es una moneda entre sube y baja (corrección encontrada al implementar; antes decía "casi siempre lateral"). Se deja a propósito como ejemplo de modelo que no sabe nada, y se compara la **calibración** (Brier) de Monte Carlo vs CNN/LSTM vs "⅓ fijo".
- Se vectoriza con numpy: 919 días de test × 1.000 × 7 tarda segundos.

### 7. P-valores
- **Desplazamiento circular:** se rota `y_real` por un desplazamiento aleatorio de entre 30 y n−30 días, 1.000 veces, y se recalcula la balanced accuracy.
- **IC 95%:** bootstrap por bloques móviles de 30 días.
- **Holm** sobre los modelos candidatos del examen (no sobre los descartados, que ya se eligieron mirando validación). Se implementa a mano (10 líneas) para no agregar dependencias.
- Se reemplaza `p_valor_permutacion`. La columna "significativo" usa el p-valor corregido.

### 8. `tipo_de`
Se pasa a un diccionario explícito nombre → tipo, con los tipos {Baseline, Simulación, Red propia, Descartado, Pre-entrenado}. Si aparece un nombre que no está en el diccionario, salta un error (en vez de caer en "Red propia" por defecto).

## Riesgos / Trade-offs

- [Reordenar 2.600 líneas puede romper dependencias entre celdas] → Se mueven bloques enteros sin reescribirlos, se corre primero con `MODO_RAPIDO=True` en Colab y se agrega una verificación de sintaxis local (`python3 -c "import ast; ..."` sobre cada celda de código).
- [La caza de patrones va a encontrar "patrones" por azar (muchas pruebas)] → Cada hallazgo se valida fuera de train y se aplica Holm en el resumen. El texto habla de "aparece / se sostiene", no de "funciona".
- [yfinance cambia de formato o limita las descargas] → try/except por activo, cache en CSV y el notebook sigue sin ese activo.
- [La lista de eventos es subjetiva] → Se cita la fuente de cada evento y se la presenta como contexto, no como prueba.
- [Corrida más larga] → Lo nuevo es barato (segundos o minutos). El costo dominante sigue siendo el entrenamiento de las redes.

## Plan de migración

1. Implementar en el generador y regenerar el `.ipynb`.
2. Mia corre en Colab con `MODO_RAPIDO=True` → se corrigen errores.
3. Corrida completa con GPU → zip → Claude analiza y redacta las conclusiones.
4. Actualizar `CLAUDE.md` (pregunta, estructura) y avisarle a Mia qué cambia en sus notas de Word.
- Rollback: el commit anterior (`9b7c265`) tiene el generador viejo.

## Preguntas abiertas

- Qué papers concretos se citan en las conclusiones ("coincide con la bibliografía"). Candidatos: Fama (1970) sobre mercados eficientes, Marshall, Young & Rose (2006) sobre velas sin valor en el DJIA, Jiang, Kelly & Xiu (2023). Se verifican antes de citarlos; no cambia el diseño.
- El k exacto de las familias se decide por silueta en la corrida.
