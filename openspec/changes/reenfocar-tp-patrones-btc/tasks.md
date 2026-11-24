# Tareas

> Verificación local en todas las tareas: `python3 src/generar_notebook.py TP_Velas_Bitcoin.ipynb` regenera sin error y cada celda de código del .ipynb pasa por `ast.parse` (lo llamamos "chequeo de sintaxis"). La verificación real es la corrida de Mia en Colab (grupo 8).

## 1. Reordenar el esqueleto

- [x] 1.1 Reescribir la sección 1 con la pregunta central "¿Qué patrones de velas esconde Bitcoin y qué nos dicen?", la escalera de sub-preguntas (la predictiva como una más) y H0/H1; verificar que pase el chequeo de sintaxis y que el texto nombre la pregunta nueva
- [x] 1.2 Mover los bloques de secciones al orden nuevo (0 contexto · 1 pregunta · 2 entorno · 3 dataset · 4 benchmark · 5 caza de patrones · 6 preparación · 7 ¿sirven para predecir? · 8 examen · 9 errores · 10 descartes · 11 conclusiones · zip) y renumerar los títulos y los nombres de los archivos de resultados; verificar que los títulos `# N.` del .ipynb salgan en ese orden
- [x] 1.3 Mover las mejoras (ex-9), ResNet (ex-10.4) y diarias vs semanales (ex-7.4b) a la sección 10 "Lo que probamos y descartamos", con la tabla qué/por qué/resultado en validación/motivo; verificar que no quede ninguna referencia a velas semanales en las secciones 6 a 8
- [x] 1.4 Pasar las constantes que usan varias secciones (fechas de halvings y de split, embargo) a la celda de entorno; verificar con grep que se definan una sola vez y antes de su primer uso

## 2. Contexto, línea de tiempo e hitos

- [x] 2.1 Detectar los hitos con zigzag sobre el log del cierre (umbral ln 2) y guardar la tabla; verificar que con los datos actuales salgan los 17 tramos (ago-2011 → jun-2026), entre ellos −70% abr-2013, −53% abr-jul 2021 y −53% oct-2025 → jun-2026
- [ ] 2.2 Redactar el contexto y la fuente de cada hito y de los ~15 eventos (borrador: lista de Gemini + tramos que le faltan), y mostrárselo a Mia para que lo apruebe; verificar que cada fila tenga una fuente verificada y que ninguna diga "la causa fue"
- [x] 2.3 Agregar a la sección 0 el gráfico de precio log con halvings, eventos e hitos marcados, la tabla del retorno a 7 días por evento con su percentil contra todos los días y la galería "las 20 velas antes de cada hito" (con el mismo dibujo que usa la CNN); guardar en `resultados/`; chequeo de sintaxis
- [x] 2.4 En el examen, completar la galería de hitos con lo que dijeron la CNN y la LSTM en los hitos que caen en validación o test (y la aclaración de "sin predicción fuera de muestra" en los demás); verificar que la caída oct-2025 → jun-2026 muestre las predicciones de test
- [x] 2.5 Cerrar la sección con un "¿qué nos llevamos?" que conecte los hitos con las velas; verificar que lo mencione explícitamente

## 3. Benchmark

- [x] 3.1 Agregar la descarga con yfinance de ETH, LTC, XRP, ^GSPC, ^IXIC, GC=F y LLY, con cache en CSV y try/except con aviso por activo; verificar que el código sigue si falla un ticker (probarlo en la corrida rápida dejando un ticker inválido a propósito, y después sacarlo)
- [x] 3.2 Calcular la tabla comparativa (CAGR, volatilidad anual, máxima caída, % de días con mecha larga, cuerpo medio) en el período común, más la figura de velas lado a lado; guardarlas en resultados
- [x] 3.3 Escribir "¿Por qué Bitcoin?" citando al menos dos métricas de la tabla y mover ahí el panorama de halvings (ex-4); chequeo de sintaxis

## 4. Caza de patrones

- [x] 4.1 Detectores de martillo, estrella fugaz, envolvente alcista/bajista y doji sobre la anatomía de la vela, con la tabla de definiciones (Nison); tabla de frecuencia + retorno a 7 días + % sube/baja/lateral vs todos los días + marca de "muestra chica" si hay menos de 30 casos
- [x] 4.2 Replicar la planilla del profe en BTC ("sharpe" de 3 ruedas, HAMMER, P(ret 3d > 3% | sharpe > 0,9)) y ponerla al lado de LLY; verificar contra `MFO2509.xlsx` que la fórmula del "sharpe" coincida
- [x] 4.3 Ciclos superpuestos (precio normalizado desde el halving, con el día actual marcado) + tabla tipo de vela × fase con χ²
- [x] 4.4 Familias con KMeans ajustado solo con train (k por silueta 4-10), centroides dibujados como velas, retorno posterior por familia en train/val/test; verificar en el código que `fit` reciba solo filas de train
- [x] 4.5 Shazam con las últimas velas: top 5 parecidos sin solapamiento, figura superpuesta, retorno posterior de cada uno
- [x] 4.6 Tabla resumen "qué esconden las velas" (patrón, n, qué pasó después, ¿se sostiene fuera de train?, p corregido por Holm) + párrafo "¿qué nos llevamos?"; guardarla en resultados

## 5. Evaluación predictiva

- [x] 5.1 Reemplazar `p_valor_permutacion` por desplazamiento circular + IC por bootstrap de bloques + Holm (implementado a mano); verificar con un ejemplo sintético dentro del generador (predicción = y real → p≈0; predicción al azar → p no significativo) que corra en la celda
- [x] 5.2 Agregar Monte Carlo (paseo aleatorio con volatilidad de 30d + variante de bootstrap histórico usando solo días anteriores a t), registrado como modelo tipo "Simulación"; verificar que no use datos posteriores a t
- [x] 5.3 Reemplazar `tipo_de` por un diccionario explícito que dé error ante un nombre desconocido; verificar que "+ criptos" salga como "Descartado"
- [x] 5.4 Separar en el examen los candidatos (baselines, Monte Carlo, CNN, LSTM, Kronos, Chronos-2) de los descartados (tabla secundaria) y aplicar Holm solo a los candidatos
- [x] 5.5 Agregar a la inversión simulada a Kronos, Chronos-2 y Monte Carlo; verificar que la tabla incluya a todos los candidatos más buy & hold
- [x] 5.6 Agregar la comparación de calibración (probabilidades de Monte Carlo vs CNN vs LSTM) y el texto CNN vs recurrente; chequeo de sintaxis

## 6. Conclusiones

- [x] 6.1 Reescribir las conclusiones automáticas: responder cada sub-pregunta con números, el bloque "qué esconden las velas" y el bloque "¿sirven para predecir?", con la marca de provisorio si se corrió con `MODO_RAPIDO`
- [x] 6.2 Verificar las citas de la bibliografía (Fama 1970, Marshall-Young-Rose 2006, Jiang-Kelly-Xiu 2023) y agregarlas a las conclusiones; verificar que cada cita tenga autor, año y DOI o URL

## 7. Documentación del proyecto

- [x] 7.1 Actualizar `CLAUDE.md` (pregunta vigente, estructura nueva, decisiones) y preparar la lista de qué cambia en las notas de Word de Mia; verificar que la sección "Decisiones vigentes" no contradiga la pregunta nueva

## 8. Integración en Colab

- [x] 8.1 Mia corre el notebook en Colab con `MODO_RAPIDO=True`; verificar que corra de punta a punta sin errores y que se genere el zip (se saltó: Mia corrió directo la completa el 2026-11-23 y salió de punta a punta sin errores)
- [x] 8.2 Mia hace la corrida completa con GPU T4; verificar que el zip tenga las figuras y tablas nuevas (eventos, benchmark, patrones, familias, Shazam, Monte Carlo, descartes) y que las conclusiones salgan sin marca de provisorio

## Workflow follow-up

- Analizar el zip de la corrida completa y redactar el informe.
- Archivar este change (`/openspec-archive-change`) cuando el informe esté cerrado.
- Abrir el change siguiente: landing web interactiva.
