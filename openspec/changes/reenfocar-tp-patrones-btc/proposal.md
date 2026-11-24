# Propuesta: reenfocar el TP en "qué patrones esconde Bitcoin"

## Por qué

El notebook actual gira alrededor de una pregunta predictiva ("¿el precio de hoy ya estaba en las velas?") cuya respuesta, después de la corrida completa, es "no de forma significativa": ningún modelo supera al azar con p-valores corregidos (Holm). Presentado así, el trabajo parece un fracaso, cuando en realidad coincide con la bibliografía. La 2ª ronda de correcciones (18-nov, Fran Sinagra) además pidió cosas que el relato actual no tiene: línea de tiempo de eventos, benchmark contra otros mercados, comparación con simulaciones Monte Carlo, justificar cada decisión y documentar lo descartado. Hay que cerrar un **trabajo terminado, con conclusiones buenas y coherentes**, antes de pensar en la app (que queda para un change aparte).

## Qué cambia

> **Lo que NO cambia:** el objetivo del trabajo sigue siendo el mismo que pide la consigna: **velas japonesas diarias de BTC** como única fuente de información de los modelos, y **una CNN que aprende de las imágenes de las velas y una red recurrente (LSTM) que las lee como secuencia**. Todo lo nuevo (contexto, hitos, benchmark) está para entender y explicar las velas, no para reemplazarlas.

- **Nueva pregunta central (exploratoria):** "¿Qué patrones de velas esconde Bitcoin y qué nos dicen?". La predicción pasa a ser una sub-pregunta: "¿sirven esos patrones para predecir?", y su respuesta negativa se presenta como hallazgo, no como fracaso.
- **Nuevo orden del notebook (relato):**
  1. Contexto de Bitcoin + **línea de tiempo de eventos** (noticias que pegaron en el precio) + **hitos: cada tramo en que el precio se duplicó o se cortó a la mitad, detectado en los datos, explicado con el contexto de ese momento y con la pregunta "¿las velas (y la CNN / LSTM) lo anunciaban?"**.
  2. **Benchmark:** BTC vs otras criptos, S&P 500 / Nasdaq, oro y LLY (la acción del profe), en velas; por qué elegimos BTC.
  3. Dataset: cómo y por qué se armó (se mantiene lo actual, con cada decisión justificada).
  4. **Caza de patrones** (lo nuevo y central): patrones clásicos en BTC, la planilla del profe replicada en BTC, patrones por fase del halving, familias de velas que encuentra la máquina sola, y el "Shazam" (a qué momento pasado se parece hoy).
  5. **¿Sirven para predecir?** Baselines, **Monte Carlo como estándar de la industria**, CNN vs LSTM (recurrente) y pre-entrenados (Kronos, Chronos-2).
  6. **Lo que se probó y se descartó** y por qué (mejoras 9.x, ResNet, híbrida, velas semanales, etc.).
  7. **Conclusiones** respondidas con los números de la corrida y contrastadas con la bibliografía.
- **Corrección estadística:** reemplazar los p-valores por permutación simple (optimistas) por permutación con desplazamiento circular + IC por bootstrap por bloques + corrección de Holm por comparaciones múltiples.
- **Arreglos:** bug de `tipo_de` ("+ criptos" sale como "Red propia"); la inversión simulada de Kronos (hoy calculada aparte) entra al notebook.
- **Velas semanales fuera de los modelos** (pedido de los profes): la comparación 7.4b pasa a "descartes"; las semanales quedan solo para visualizar ciclos.
- **Fuera de este change:** la app/landing web interactiva (change siguiente), velas de 4h, features de halving como entrada de los modelos más allá de lo que ya existe.

## Capacidades

### Capacidades nuevas
- `relato-del-tp`: pregunta de investigación, orden de las secciones, justificación de decisiones, sección de descartes y conclusiones automáticas del notebook.
- `contexto-y-benchmark`: línea de tiempo de eventos de BTC y comparación visual/estadística de BTC contra otros activos.
- `caza-de-patrones`: análisis exploratorio de patrones de velas en BTC (clásicos, planilla del profe, por fase de halving, familias descubiertas, Shazam).
- `evaluacion-predictiva`: cómo se mide si algo predice: baselines, Monte Carlo, modelos, p-valores robustos y corrección por comparaciones múltiples, sin data leakage.

### Capacidades modificadas
- (ninguna: todavía no hay specs en `openspec/specs/`)

## Impacto

- `src/generar_notebook.py` (≈2.600 líneas): se reordenan secciones, se agregan 4.x (línea de tiempo, benchmark) y la caza de patrones, se agrega Monte Carlo, se reescriben 1 y 13, se mueve 9/10.4/7.4b a "descartes", se cambian los p-valores.
- `TP_Velas_Bitcoin.ipynb`: se regenera (nunca se edita a mano).
- Dependencias nuevas en Colab: ninguna obligatoria (yfinance ya se usa; scikit-learn trae KMeans/PCA para las familias de velas).
- Tiempo de corrida en Colab: sube un poco (Monte Carlo y clustering son baratos; el benchmark baja datos de yfinance).
- `CLAUDE.md` y las notas de Word de Mia: actualizar la pregunta de investigación y la estructura.
