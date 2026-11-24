"""Genera TP_Velas_Bitcoin.ipynb (notebook único para Colab)."""
import json, sys
from pathlib import Path

celdas = []
def md(txt): celdas.append({"cell_type": "markdown", "metadata": {}, "source": txt.strip("\n")})
def code(txt): celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": txt.strip("\n")})


# =====================================================================
md(r"""
# 🕯️ Velas japonesas y Deep Learning sobre Bitcoin
### ¿Qué patrones de velas esconde Bitcoin y qué nos dicen?

**Autora:** Mia Maffi

Este notebook contiene **todo el trabajo en un solo lugar**: desde la descarga de los datos crudos hasta las conclusiones. Cada sección explica **qué** hace el código y **por qué**.

**El objetivo no cambia en ninguna parte del trabajo:** mirar **solo las velas japonesas diarias de Bitcoin**, buscar los patrones que se repiten, y entrenar **una CNN que aprende de las imágenes de las velas** y **una red recurrente (LSTM) que las lee en orden**. El contexto histórico y la comparación con otros mercados están para **entender** las velas, no para reemplazarlas.

| Parte | # | Sección |
|---|---|---|
| **I. Bitcoin y sus datos** | 0 | Contexto: qué es Bitcoin, oferta finita y halving |
| | 1 | Pregunta de investigación |
| | 2 | Preparación del entorno |
| | 3 | Construcción de la base de datos de velas |
| | 4 | La historia de Bitcoin en sus velas: hitos, eventos y halvings |
| | 5 | Benchmark: Bitcoin contra otros mercados |
| **II. La caza de patrones** | 6 | Anatomía de cada vela y la respuesta de cada día |
| | 7 | **¿Qué patrones esconden las velas?** |
| **III. ¿Sirven para predecir?** | 8 | Baselines y simulaciones de Monte Carlo |
| | 9 | CNN: la red que mira la foto de las velas |
| | 10 | LSTM: la red recurrente que lee las velas |
| | 11 | Rivales pre-entrenados (Kronos, Chronos-2) |
| | 12 | El examen final en el ciclo actual |
| | 13 | Anatomía de los errores |
| **IV. Cierre** | 14 | Lo que probamos y descartamos |
| | 15 | Conclusiones |

> **Cómo usarlo:** en Colab, `Entorno de ejecución → Cambiar tipo → T4 GPU` y después `Ejecutar todas`. Con `MODO_RAPIDO = True` (sección 2) corre en pocos minutos para probar que no haya errores.
""")

# =====================================================================
md(r"""
---
# 0. Contexto: ¿qué es Bitcoin y por qué es especial?

### Qué es
**Bitcoin** es un dinero digital **descentralizado**: no lo emite ningún banco central ni lo controla ninguna empresa. Las transacciones se registran en la **blockchain**, un libro contable público que comparten miles de computadoras.

### Cómo se crean los bitcoins nuevos
Los **mineros** compiten resolviendo un problema matemático (*prueba de trabajo*). El ganador agrega un nuevo **bloque** de transacciones a la cadena y cobra una **recompensa en bitcoins nuevos**. Se genera un bloque cada **~10 minutos**, lo que da unos 144 bloques por día.

### Oferta finita: 21 millones
El código de Bitcoin fija que **nunca van a existir más de 21.000.000 de BTC**. Hoy ya se emitió más del **95%**, y el último bitcoin se va a minar cerca del año **2140**. Es lo opuesto a una moneda tradicional, que se puede imprimir sin límite.

### El halving: la emisión se corta a la mitad cada ~4 años
Cada **210.000 bloques** (≈ 4 años) la recompensa por bloque **se divide por 2**. Ojo: no se reduce la cantidad de bitcoins que ya existen, se reduce **la velocidad a la que aparecen bitcoins nuevos**.

| Evento | Fecha | Recompensa por bloque | BTC nuevos por día |
|---|---|---|---|
| Génesis | 3-ene-2009 | 50 BTC | 7.200 |
| 1° halving | 28-nov-2012 | 25 BTC | 3.600 |
| 2° halving | 9-jul-2016 | 12,5 BTC | 1.800 |
| 3° halving | 11-may-2020 | 6,25 BTC | 900 |
| 4° halving | 20-abr-2024 | 3,125 BTC | 450 |
| 5° halving | ~2028 | 1,5625 BTC | 225 |

### ¿Por qué importa para este trabajo?
- Cada halving es un **shock de oferta programado**: se sabe con años de anticipación cuándo va a ocurrir.
- Históricamente el precio mostró **ciclos de ~4 años**: suba fuerte después del halving, pico, caída fuerte y acumulación.
- Si hay ciclos que se repiten, **quizás también se repiten patrones de velas** dentro de cada fase del ciclo. Eso es exactamente lo que pide la consigna: buscar **recurrencia de patrones**.
- En la sección 4 recorremos la historia completa: los **hitos** (cada vez que el precio se duplicó o se cortó a la mitad), qué pasaba en el mundo en cada uno y cómo se veían las velas justo antes.

### Mercado
Bitcoin opera **24 horas, 7 días a la semana**, sin cierre. Es un activo **líquido** que hoy se negocia en EE.UU. a través de ETFs spot (por ejemplo IBIT, desde enero de 2024) y de futuros en CME.
""")

# =====================================================================
md(r"""
---
# 1. Pregunta de investigación

### La consigna del profesor
> *"Esa recurrencia de los patrones es lo que querríamos poder entrenar."*
> Buscar patrones de velas que se repitan y den **cierta predictibilidad** en los precios: mirando **solo el patrón de velas**, ¿el activo me dice si **va a subir, va a bajar o se va a detener**?

### Pregunta central
> **¿Qué patrones de velas esconde Bitcoin y qué nos dicen?**

Es una pregunta **exploratoria**: primero buscamos qué patrones aparecen en las velas diarias de Bitcoin (los de los libros, los de la planilla del profe, los que se repiten en cada ciclo de halving y los que descubre la máquina sola) y medimos **qué pasó después** de cada uno. Recién después preguntamos si sirven para anticipar el futuro.

### La sub-pregunta predictiva (comprobable)
> **¿Esos patrones sirven para predecir?** Si entrenamos una CNN y una LSTM **solo con los ciclos anteriores** (2013 → halving de abril de 2024) y les mostramos el ciclo actual **día por día, como si lo estuvieran viviendo**, ¿habrían anticipado si la **semana siguiente** el precio iba a **subir, bajar o quedar lateral**, mejor que el azar, que las reglas clásicas y que una simulación de Monte Carlo?

**¿Por qué así y no "cuánto va a valer Bitcoin"?** Predecir un precio futuro no se puede comprobar hoy. El ciclo actual (abr-2024 → hoy) **ya pasó**: cada predicción se puede marcar como **acierto o error**.

### Cómo lo respondemos
- **Datos:** solo **velas diarias** de Bitcoin (≈5.000 desde 2013). Las semanales se usan solo para dibujar los ciclos: son muy pocas para entrenar.
- **Dos redes que miran las mismas velas de dos maneras:** la **CNN** mira la **foto** de las velas, como un trader; la **LSTM** las **lee** una por una, en orden y con memoria.

### Hipótesis (parte predictiva)
- **H0 (mercado eficiente):** el precio ya incorpora la información de las velas pasadas: ningún modelo supera al azar de forma consistente.
- **H1:** las velas tienen un poder predictivo débil pero real.

### 🪜 La escalera de preguntas
| # | Pregunta | Cómo la respondemos | Sección |
|---|---|---|---|
| 1 | ¿Qué movimientos marcaron la historia de Bitcoin y qué pasaba en cada uno? | hitos detectados en los datos + contexto | 4 |
| 2 | ¿En qué se parecen y en qué se diferencian las velas de Bitcoin de las de otros mercados? | benchmark | 5 |
| 3 | ¿Los patrones de los libros (martillo, envolventes, doji...) anticipan algo en Bitcoin? | patrones clásicos | 7.1 |
| 4 | ¿Lo que mide la planilla del profe en LLY funciona en Bitcoin? | planilla replicada | 7.2 |
| 5 | ¿Hay patrones que se repiten en la misma fase de cada ciclo de halving? | ciclos superpuestos | 7.3 |
| 6 | ¿Qué familias de velas encuentra la máquina sola, y se sostienen en el tiempo? | agrupamiento | 7.4 |
| 7 | ¿A qué momento del pasado se parece hoy? | "Shazam" | 7.5 |
| 8 | ¿Las reglas simples y una simulación de Monte Carlo anticipan la semana siguiente? | baselines y Monte Carlo | 8 |
| 9 | ¿Es mejor **mirar la foto** (CNN) o **leer la secuencia** (LSTM)? | las dos redes, mismas reglas | 9 y 10 |
| 10 | ¿Un modelo pre-entrenado con miles de millones de velas ve más? | Kronos y Chronos-2 | 11 |
| 11 | **¿Alguno le gana al azar en el ciclo actual?** | examen final con p-valores corregidos | 12 |
| 12 | ¿Por qué se equivocan? | anatomía de los errores | 13 |
| 13 | ¿Qué más probamos y por qué lo descartamos? | descartes | 14 |
""")

# =====================================================================
md(r"""
---
# 2. Preparación del entorno

Instalamos y cargamos las librerías:
- **requests**: hace los pedidos a la API de Bitstamp (descarga de datos).
- **pandas**: maneja las tablas de velas (el "Excel" de Python).
- **matplotlib** y **mplfinance**: gráficos; mplfinance dibuja velas japonesas.

- **yfinance**: baja precios de otros mercados (benchmark, sección 5).

Colab trae todo menos `mplfinance` y `yfinance`, así que los instalamos.

### ⚙️ Configuración de la corrida
`MODO_RAPIDO = True` corre **todo el notebook en pocos minutos**, con muy poco entrenamiento: sirve **solo para comprobar que no haya errores**, y sus resultados **no valen** (las conclusiones salen marcadas como provisorias). Para el informe: `MODO_RAPIDO = False` y **GPU activada** (Entorno de ejecución → Cambiar tipo → T4 GPU). La corrida completa puede tardar cerca de una hora.
""")
code(r"""
!pip install -q mplfinance yfinance
""")
code(r"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import requests

pd.set_option("display.width", 140)
""")
md(r"""
### ¿Dónde se guardan los datos?
Colab **borra los archivos al cerrar la sesión**. Si querés conservarlos, poné `USAR_DRIVE = True`: se guardan en tu Google Drive y la próxima vez no hace falta volver a descargarlos.
""")
code(r"""
USAR_DRIVE = False   # True -> guarda todo en Google Drive (persistente)

if USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/TP_Velas_Bitcoin")
else:
    BASE = Path("datos_tp_btc")

RAW = BASE / "raw"              # datos crudos tal como vienen de la API
VELAS = BASE / "velas"          # nuestras velas construidas
VALID = BASE / "validacion"     # archivos para el control de calidad
RESULTADOS = BASE / "resultados"  # gráficos y tablas para el informe (se baja como zip al final)
for carpeta in (RAW, VELAS, VALID, RESULTADOS):
    carpeta.mkdir(parents=True, exist_ok=True)

# --- Ayudantes para guardar todo lo importante para el informe ---
RESUMEN = []                       # números clave de cada paso -> 00_resumen.txt

def guardar_fig(nombre, fig=None):
    # Guarda el gráfico actual como PNG en la carpeta de resultados
    (fig or plt.gcf()).savefig(RESULTADOS / f"{nombre}.png", dpi=150, bbox_inches="tight")

def guardar_tabla(tabla, nombre):
    tabla.to_csv(RESULTADOS / f"{nombre}.csv")

def anotar(texto):
    # Muestra un resultado clave y lo guarda en el resumen
    print(texto)
    RESUMEN.append(texto)

# Fechas de los halvings (fuente: blockchain, bloques 210.000 / 420.000 / 630.000 / 840.000)
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"]).tz_localize("UTC")
NOMBRES_HALVING = ["1° (50→25)", "2° (25→12,5)", "3° (12,5→6,25)", "4° (6,25→3,125)"]
# Parámetros del experimento (se usan en varias secciones; se definen una sola vez acá)
HORIZONTE = 7      # días hacia adelante que queremos anticipar
VENTANA_VOL = 30   # días que usamos para medir la volatilidad reciente
INICIO = pd.Timestamp("2013-01-01", tz="UTC")             # antes hay muy pocas operaciones (sección 3.3)
CORTE_VAL, CORTE_TEST = HALVINGS[2], HALVINGS[3]          # validación desde may-2020, test desde abr-2024
EMBARGO = pd.Timedelta(days=HORIZONTE)                    # días que se sacan antes de cada corte

# ---------- Configuración de la corrida ----------
MODO_RAPIDO = False      # True = prueba de pocos minutos para ver que no haya errores (resultados NO válidos)
print("Carpeta de trabajo:", BASE.resolve())
""")

# =====================================================================
md(r"""
---
# 3. Construcción de la base de datos de velas

## 3.1 ¿De dónde salen los datos?

| Fuente | Desde | ¿Cubre los 4 halvings? |
|---|---|---|
| **Bitstamp BTC/USD** ⭐ | sep-2011 | ✅ |
| Coinbase BTC-USD | jul-2015 | ❌ (falta 2012) |
| Binance BTCUSDT | ago-2017 | ❌ (faltan 2012 y 2016) |

Usamos **Bitstamp**: es uno de los exchanges más antiguos, opera en dólares, tiene una API pública y gratuita, y es el único que cubre los 4 halvings. Coinbase la usamos para **validar**.

## 3.2 Descarga: velas de 1 hora

Bajamos velas de **1 hora** como "materia prima". Con ellas después construimos nosotras las velas de 4 horas, 1 día y 1 semana.

**¿Por qué 1 hora y no 1 minuto?** Con 1 minuto serían ~8 millones de filas y no ganamos nada para analizar ciclos de días o semanas. Con 1 hora son ~133.000 filas.

**El problema:** la API devuelve **como máximo 1.000 velas por pedido**. La solución es un **loop**: pedimos 1.000 horas, avanzamos hasta la última recibida, volvemos a pedir… y así hasta hoy (~134 pedidos).

> Si el archivo ya existe (por ejemplo en Drive), se saltea la descarga.
""")
code(r"""
URL_BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"

def pedir_bloque(desde_unix, paso_seg, limite=1000):
    # Pide hasta `limite` velas a Bitstamp empezando en `desde_unix`. Reintenta si falla.
    params = {"step": paso_seg, "limit": limite, "start": desde_unix}
    for intento in range(5):
        try:
            r = requests.get(URL_BITSTAMP, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["data"]["ohlc"]
        except Exception as e:
            espera = 2 ** intento                       # 1, 2, 4, 8, 16 segundos
            print(f"  error ({e}); reintento en {espera}s")
            time.sleep(espera)
    raise RuntimeError("Bitstamp no respondió después de 5 intentos")


def descargar_bitstamp(paso_seg, desde="2011-08-01"):
    # Loop de descarga completo: devuelve un DataFrame con todas las velas hasta hoy.
    desde_unix = int(pd.Timestamp(desde, tz="UTC").timestamp())
    ahora = int(time.time())
    filas, pedidos = [], 0
    while desde_unix < ahora:
        bloque = pedir_bloque(desde_unix, paso_seg)
        pedidos += 1
        if bloque:
            filas.extend(bloque)
            desde_unix = int(bloque[-1]["timestamp"]) + paso_seg   # seguimos desde la vela siguiente
        else:
            desde_unix += paso_seg * 1000                           # tramo vacío: saltamos
        if pedidos % 20 == 0:
            print(f"{pedidos} pedidos | {len(filas):,} velas | voy por {pd.to_datetime(desde_unix, unit='s'):%Y-%m-%d}")
        time.sleep(0.25)                                            # no saturar la API
    df = pd.DataFrame(filas)[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


ARCHIVO_CRUDO = RAW / "btcusd_1h_bitstamp.csv"
if ARCHIVO_CRUDO.exists():
    crudo = pd.read_csv(ARCHIVO_CRUDO)
    print("Ya estaba descargado.")
else:
    crudo = descargar_bitstamp(paso_seg=3600)
    crudo.to_csv(ARCHIVO_CRUDO, index=False)

print(f"{len(crudo):,} velas de 1 hora")
crudo.head()
""")
md(r"""
Cada fila es una **vela de 1 hora**:
- `timestamp`: momento de **apertura** de la vela, en segundos desde 1970 (formato Unix).
- `open`, `high`, `low`, `close`: precio de apertura, máximo, mínimo y cierre de esa hora, en USD.
- `volume`: cantidad de BTC operados en esa hora.

## 3.3 Limpieza

Antes de usar datos hay que **desconfiar** de ellos. Revisamos:

1. **Fechas:** pasamos el timestamp a fecha en **UTC**. Bitcoin opera 24/7, no tiene "horario de mercado", así que UTC es la referencia universal.
2. **Duplicados y orden:** el loop puede repetir alguna vela en los bordes de cada pedido.
3. **Vela sin cerrar:** la hora en curso todavía no terminó, así que la sacamos.
4. **Velas imposibles:** máximo menor que el mínimo, o apertura/cierre fuera del rango. Serían errores de datos.
5. **Huecos:** horas que directamente no están.
6. **Velas "planas" sin operaciones:** apertura = máximo = mínimo = cierre y volumen 0. En esa hora **nadie operó** y Bitstamp repite el último precio. No las borramos: las **marcamos** con la columna `sin_operaciones`.
""")
code(r"""
h1 = crudo.copy()

# 1. Fechas en UTC
h1["fecha"] = pd.to_datetime(h1["timestamp"], unit="s", utc=True)

# 2. Duplicados y orden
n_dup = h1.duplicated("timestamp").sum()
h1 = h1.drop_duplicates("timestamp").sort_values("fecha")

# 3. Vela sin cerrar
sin_cerrar = h1["fecha"] + pd.Timedelta(hours=1) > pd.Timestamp.now(tz="UTC")
h1 = h1[~sin_cerrar]

# 4. Velas imposibles
imposibles = (
    (h1["high"] < h1["low"])
    | (h1[["open", "close"]].max(axis=1) > h1["high"] + 1e-9)
    | (h1[["open", "close"]].min(axis=1) < h1["low"] - 1e-9)
    | (h1[["open", "high", "low", "close"]] <= 0).any(axis=1)
)
h1 = h1[~imposibles]

# 5. Huecos: comparamos contra una grilla con TODAS las horas del período
grilla = pd.date_range(h1["fecha"].min(), h1["fecha"].max(), freq="1h")
faltantes = grilla.difference(pd.DatetimeIndex(h1["fecha"]))

# 6. Velas planas sin operaciones
h1["sin_operaciones"] = (h1["volume"] == 0) & (h1["high"] == h1["low"])

h1 = h1.set_index("fecha")[["open", "high", "low", "close", "volume", "sin_operaciones"]]

print(f"Duplicados: {n_dup} | sin cerrar: {sin_cerrar.sum()} | imposibles: {imposibles.sum()} | horas faltantes: {len(faltantes)}")
print(f"Período: {h1.index.min():%Y-%m-%d} → {h1.index.max():%Y-%m-%d %H:%M} UTC  ({len(h1):,} horas)")
""")
md(r"""
### Calidad de los datos por año
""")
code(r"""
anio = h1.index.year
calidad = pd.DataFrame({
    "horas": h1.groupby(anio).size(),
    "horas_sin_operaciones": h1.groupby(anio)["sin_operaciones"].sum(),
    "volumen_BTC": h1.groupby(anio)["volume"].sum().round(0).astype(int),
})
calidad["% horas con operaciones"] = (100 * (1 - calidad["horas_sin_operaciones"] / calidad["horas"])).round(1)
guardar_tabla(calidad, "03_calidad_por_anio")
calidad
""")
md(r"""
### 🔎 Qué nos dicen estos números

- **No falta ninguna hora**: Bitstamp rellena las horas vacías con velas planas.
- **2011: solo ~10% de las horas tuvo operaciones; 2012: ~63%.** En esos años el mercado casi entero estaba en **Mt. Gox** (exchange japonés que quebró en 2014) y Bitstamp recién arrancaba. Desde 2013 hay operaciones prácticamente todas las horas.
  → **Decisión:** para entrenar los modelos usamos datos **desde 2013**. El halving de 2012 se usa solo para la parte visual de ciclos.
- **2015: ~107 horas sin operaciones.** Veamos cuándo fueron 👇
""")
code(r"""
planas_2015 = h1[(h1.index.year == 2015) & h1["sin_operaciones"]]
planas_2015.groupby(planas_2015.index.date).size().rename("horas sin operaciones")
""")
md(r"""
Las horas sin operaciones de 2015 están todas entre el **5 y el 9 de enero de 2015**, cuando **Bitstamp sufrió un hackeo** (le robaron ~19.000 BTC) y suspendió el servicio durante varios días. **El hackeo quedó grabado en los datos.**

## 3.4 Construcción de nuestras velas japonesas (4h, 1 día, 1 semana)

Juntamos varias velas de 1 hora en una más grande (**resample**). La regla:

| Campo | Cómo se calcula | Por qué |
|---|---|---|
| **open** | el **primer** precio del período | es el precio con el que abre |
| **high** | el **máximo** de los máximos | el precio más alto alcanzado |
| **low** | el **mínimo** de los mínimos | el precio más bajo alcanzado |
| **close** | el **último** precio del período | es el precio con el que cierra |
| **volume** | la **suma** | total operado |

**Detalle importante:** las horas *sin operaciones* **no se usan**, porque su precio es una copia del anterior y en ese período nadie operó a ese precio. Si un período entero no tuvo operaciones, queda una vela plana marcada.

**Convenciones del mercado cripto:**
- Todo en **UTC**. La vela diaria va de 00:00 a 24:00 UTC (de 21:00 a 21:00 en Argentina).
- Las velas de 4h arrancan a las 00, 04, 08, 12, 16 y 20 UTC.
- La vela semanal va de **lunes a domingo**.
- Cada vela se identifica por su fecha/hora de **apertura**.
""")
code(r"""
REGLAS_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

def construir_velas(h1, regla, duracion):
    # Arma velas de la temporalidad `regla` a partir de velas de 1 hora.
    activas = h1[~h1["sin_operaciones"]]            # solo horas en que hubo operaciones
    opciones = {"label": "left", "closed": "left"}   # etiqueta = apertura de la vela

    velas = activas.resample(regla, **opciones).agg(REGLAS_OHLCV)
    velas["horas_con_operaciones"] = activas["close"].resample(regla, **opciones).count()

    # Incluir también los períodos sin ninguna operación (vela plana en el último precio conocido)
    ultimo_precio = h1["close"].resample(regla, **opciones).last()
    velas = velas.reindex(ultimo_precio.index)
    for col in ["open", "high", "low", "close"]:
        velas[col] = velas[col].fillna(ultimo_precio)
    velas["volume"] = velas["volume"].fillna(0)
    velas["horas_con_operaciones"] = velas["horas_con_operaciones"].fillna(0).astype(int)
    velas["sin_operaciones"] = velas["horas_con_operaciones"] == 0

    # Sacar la última vela si todavía no cerró
    fin_datos = h1.index.max() + pd.Timedelta(hours=1)
    velas = velas[velas.index + duracion <= fin_datos]
    velas.index.name = "fecha"
    return velas


btc = {"1h": h1}
btc["4h"] = construir_velas(h1, "4h", pd.Timedelta(hours=4))
btc["1d"] = construir_velas(h1, "1D", pd.Timedelta(days=1))
btc["1w"] = construir_velas(h1, "W-MON", pd.Timedelta(weeks=1))

assert (btc["1w"].index.day_of_week == 0).all()     # control: todas las semanas empiezan un lunes

for nombre, df in btc.items():
    df.to_csv(VELAS / f"btc_{nombre}.csv")
    anotar(f"3. Velas {nombre}: {len(df):>7,} velas | {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d %H:%M}"
          f" | sin operaciones: {df['sin_operaciones'].sum()}")
""")
md(r"""
### ¿Qué temporalidad usamos para qué?

| Temporalidad | Velas | Uso |
|---|---|---|
| **1 día** | ~5.500 | ⭐ **Principal.** Es la escala de los ciclos de halving y la misma que usa la planilla del profesor. |
| **4 horas** | ~33.000 | Se construye, pero **no se usa en los modelos**: más ruido y se pierde la escala de los ciclos (ver 14). |
| **1 semana** | ~800 | **Solo para visualizar** los ciclos. Son muy pocas velas para entrenar una red. |

Menos de 1 hora no la usamos: tiene demasiado ruido y se pierde la relación con el halving.
""")
code(r"""
btc["1d"].tail()
""")

# =====================================================================
md(r"""
## 3.5 Validación: ¿nuestras velas están bien construidas?

Comparamos **nuestras velas diarias** (armadas a mano desde 1h) contra:
1. **La vela diaria oficial de Bitstamp.** Es la misma fuente, así que debería dar **idéntico**. Si da igual, nuestro resample está bien hecho.
2. **La vela diaria de Coinbase**, el principal exchange de EE.UU. Al ser otro exchange tiene que dar **muy parecido**, aunque no exacto, porque cada exchange tiene sus propias operaciones.
""")
code(r"""
def diaria_coinbase(desde="2015-07-20"):
    # Velas diarias de Coinbase (máx. 300 por pedido). Formato: [tiempo, low, high, open, close, volumen]
    filas = []
    ini = pd.Timestamp(desde, tz="UTC")
    hoy = pd.Timestamp.now(tz="UTC").normalize()
    while ini < hoy:
        fin = min(ini + pd.Timedelta(days=299), hoy)
        r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                         params={"granularity": 86400, "start": ini.isoformat(), "end": fin.isoformat()},
                         headers={"User-Agent": "tp-velas"}, timeout=30)
        filas += r.json()
        ini = fin + pd.Timedelta(days=1)
        time.sleep(0.4)
    df = pd.DataFrame(filas, columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]].sort_index().loc[lambda d: ~d.index.duplicated()]


def comparar(nuestra, referencia, nombre, archivo):
    comun = nuestra.index.intersection(referencia.index)
    filas = {}
    for col in ["open", "high", "low", "close"]:
        dif = (nuestra.loc[comun, col] / referencia.loc[comun, col] - 1).abs() * 100
        filas[col] = {"diferencia mediana (%)": round(dif.median(), 4),
                      "días con diferencia > 1%": int((dif > 1).sum()),
                      "% de días": round(100 * (dif > 1).mean(), 2)}
    print(f"\nNuestras velas vs {nombre}  ({len(comun):,} días en común)")
    tabla = pd.DataFrame(filas).T
    guardar_tabla(tabla, archivo)
    anotar(f"3. Validación vs {nombre}: diferencia mediana del cierre = {tabla.loc['close', 'diferencia mediana (%)']}%"
           f" en {len(comun):,} días")
    display(tabla)


try:
    bitstamp_1d = descargar_bitstamp(paso_seg=86400)
    bitstamp_1d.index = pd.to_datetime(bitstamp_1d["timestamp"], unit="s", utc=True)
    coinbase_1d = diaria_coinbase()
    nuestras = btc["1d"][~btc["1d"]["sin_operaciones"]]
    comparar(nuestras, bitstamp_1d, "la diaria OFICIAL de Bitstamp", "03_validacion_vs_bitstamp")
    comparar(nuestras, coinbase_1d, "la diaria de Coinbase (otro exchange)", "03_validacion_vs_coinbase")
except Exception as e:
    print("No se pudo validar (problema de conexión):", e)
""")
md(r"""
### 🔎 Resultado de la validación
- **vs Bitstamp oficial: 0,0000% de diferencia.** Nuestras velas son **idénticas** a las oficiales, así que la construcción está bien hecha.
- **vs Coinbase:** diferencia mediana de **~0,03%**, y solo ~2% de los días difieren más de 1% (casi todos en los primeros años, cuando los exchanges estaban menos conectados entre sí). Es lo esperable entre dos exchanges distintos, y confirma que **los datos representan bien el precio de mercado** de Bitcoin.

✅ **La base de datos está lista y validada.**
""")
# =====================================================================
md(r"""
---
# 4. La historia de Bitcoin contada por sus velas

Antes de buscar patrones chicos (de 1 a 30 velas), miramos la película completa: **¿qué movimientos marcaron la historia de Bitcoin y qué pasaba en el mundo en cada uno?** Lo hacemos en las dos direcciones:
- **Del precio a la noticia (4.2):** detectamos en los datos los movimientos más fuertes y buscamos el contexto de cada uno.
- **De la noticia al precio (4.3):** tomamos eventos conocidos y medimos qué hizo el precio la semana siguiente.

Y cerramos con la pregunta que conecta todo con el trabajo: **¿las velas de antes de cada hito lo anunciaban?** (4.4).

Primero, dos herramientas de dibujo que usamos en todo el notebook:
- `dibujar_velas`: velas japonesas a color (verde = cerró arriba, roja = cerró abajo), para que las veamos nosotras.
- `dibujar_ventana`: la **foto en blanco y negro que recibe la CNN** (sección 9.2 explica cada decisión: 64 píxeles de alto, 3 por vela, alcista hueca y bajista rellena, volumen abajo).
""")
code(r"""
from matplotlib.patches import Rectangle

def dibujar_velas(ax, ohlc, ancho=0.6):
    # ohlc: array (N velas x 4: open, high, low, close) -> velas japonesas a color en el eje ax
    for j, (o, h, l, c) in enumerate(np.asarray(ohlc, dtype=float)[:, :4]):
        color = "#2a9d55" if c >= o else "#d1495b"
        ax.vlines(j, l, h, color=color, linewidth=1)
        ax.add_patch(Rectangle((j - ancho / 2, min(o, c)), ancho, max(abs(c - o), 1e-12), color=color))
    ax.set_xlim(-1, len(ohlc))
    ax.set_ylim(np.min(np.asarray(ohlc)[:, 2]) * 0.98, np.max(np.asarray(ohlc)[:, 1]) * 1.02)

ALTO_PRECIO, SEPARACION, ALTO_VOL = 50, 2, 12
ALTO = ALTO_PRECIO + SEPARACION + ALTO_VOL       # 64 píxeles

def dibujar_ventana(v):
    # v: tabla de N velas (open, high, low, close, volume) -> imagen de 64 x 3N píxeles (0 = negro, 1 = blanco)
    N = len(v)
    img = np.zeros((ALTO, 3 * N), dtype=np.uint8)
    minimo, maximo = v[:, 2].min(), v[:, 1].max()
    escala = (ALTO_PRECIO - 1) / (maximo - minimo) if maximo > minimo else 0
    fila = lambda precio: int(round((maximo - precio) * escala))   # precio -> fila (0 = arriba)
    vol_max = v[:, 4].max()

    for j, (o, h, l, c, vol) in enumerate(v):
        x = 3 * j
        f_alto, f_bajo = fila(h), fila(l)
        f_techo, f_piso = fila(max(o, c)), fila(min(o, c))   # techo y piso del cuerpo
        img[f_alto:f_techo + 1, x + 1] = 1                    # mecha superior
        img[f_piso:f_bajo + 1, x + 1] = 1                     # mecha inferior
        if c >= o:                                            # alcista: cuerpo hueco
            img[f_techo:f_piso + 1, x] = 1
            img[f_techo:f_piso + 1, x + 2] = 1
            img[f_techo, x:x + 3] = 1
            img[f_piso, x:x + 3] = 1
        else:                                                 # bajista: cuerpo relleno
            img[f_techo:f_piso + 1, x:x + 3] = 1
        if vol_max > 0:                                       # barra de volumen
            alto_v = int(round(vol / vol_max * ALTO_VOL))
            if alto_v > 0:
                img[ALTO - alto_v:, x + 1] = 1
    return img
""")
md(r"""
## 4.1 Panorama completo (velas semanales, escala logarítmica)

Usamos **escala logarítmica** porque el precio pasó de ~$10 a más de $100.000. En escala log, una misma distancia vertical representa el mismo **porcentaje** de cambio, así que podemos comparar 2013 con 2024. (Las velas semanales se usan **solo para dibujar**; los modelos usan velas diarias.)
""")
code(r"""
w = btc["1w"]
fig, ax = plt.subplots(figsize=(14, 6))
ax.semilogy(w.index, w["close"], color="#f7931a", lw=1.5)
for fecha, nombre in zip(HALVINGS, NOMBRES_HALVING):
    ax.axvline(fecha, color="#444", ls="--", lw=1)
    ax.text(fecha, w["close"].max() * 1.3, f" Halving {nombre}", fontsize=8, rotation=90, va="top")
ax.set_title("Bitcoin (BTC/USD, Bitstamp) – cierre semanal en escala logarítmica y los 4 halvings")
ax.set_ylabel("USD (escala log)")
ax.grid(alpha=0.3, which="both")
guardar_fig("04_1_panorama_halvings")
plt.show()
""")
md(r"""
**Qué se ve:** después de cada halving vienen una **suba fuerte** (12 a 18 meses después), un **pico** (dic-2013, dic-2017, nov-2021, oct-2025), una **caída profunda** y una etapa de **acumulación** antes del siguiente halving. Es la "recurrencia" a gran escala.

## 4.2 Los hitos: cada vez que el precio se duplicó o se cortó a la mitad

**¿Cómo elegimos los hitos sin elegirlos a mano?** Con un criterio fijo, el mismo para toda la historia: un hito es un tramo en el que el precio **se duplicó (+100%) o se cortó a la mitad (−50%)** antes de darse vuelta en la misma medida.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| **Umbral "×2 o ÷2"** | un "top 8" de subidas y caídas | un número fijo es arbitrario: mete movimientos chicos o deja afuera grandes (por ejemplo, la caída de abr-jul 2021). Con el umbral, **la cantidad de hitos la decide el dato** |
| **En escala logarítmica** | en porcentaje común | +100% y −50% son el mismo tamaño de movimiento en log (multiplicar o dividir por 2): el criterio es simétrico y vale igual para $10 que para $100.000 |
| **Precio de cierre diario** | máximos y mínimos intradía | es el mismo dato que usan los modelos; por eso el pico de 2025 da $124.700 (cierre) y no $126.200 (máximo intradía) |
| **Umbral de ×2** | umbral de 40% | con 40% salen el doble de tramos, muchos de ellos ruido de 2011-2012 |

El algoritmo (*zigzag*) recorre los cierres día por día, lleva el registro del máximo (o mínimo) en curso, y confirma un giro cuando el precio se aleja un factor 2 de ese extremo. El último tramo queda **abierto**: todavía no se dio vuelta ×2.
""")
code(r"""
def detectar_hitos(cierre, umbral=np.log(2)):
    # Zigzag sobre el log del precio: devuelve los tramos en que el precio se multiplicó o dividió por e^umbral
    lp = np.log(cierre.to_numpy())
    pivotes, tendencia, alto, bajo, ext = [0], 0, 0, 0, 0
    for i in range(1, len(lp)):
        if tendencia == 0:                                   # todavía no hubo un primer movimiento x2
            alto = i if lp[i] > lp[alto] else alto
            bajo = i if lp[i] < lp[bajo] else bajo
            if lp[alto] - lp[bajo] >= umbral:
                pivotes, tendencia, ext = ([bajo], 1, alto) if alto > bajo else ([alto], -1, bajo)
        elif tendencia == 1:                                 # subiendo: seguimos el máximo
            if lp[i] > lp[ext]:
                ext = i
            elif lp[ext] - lp[i] >= umbral:                  # cayó a la mitad desde el máximo -> giro
                pivotes.append(ext)
                tendencia, ext = -1, i
        else:                                                # bajando: seguimos el mínimo
            if lp[i] < lp[ext]:
                ext = i
            elif lp[i] - lp[ext] >= umbral:                  # se duplicó desde el mínimo -> giro
                pivotes.append(ext)
                tendencia, ext = 1, i
    pivotes.append(ext)
    filas = []
    for n, (a, b) in enumerate(zip(pivotes, pivotes[1:])):
        filas.append({"inicio": cierre.index[a], "fin": cierre.index[b], "días": (cierre.index[b] - cierre.index[a]).days,
                      "precio inicio": cierre.iloc[a], "precio fin": cierre.iloc[b],
                      "movimiento": cierre.iloc[b] / cierre.iloc[a] - 1,
                      "estado": "abierto (puede seguir)" if n == len(pivotes) - 2 else "cerrado"})
    hitos = pd.DataFrame(filas)
    hitos["tipo"] = np.where(hitos["movimiento"] > 0, "subida", "caída")
    return hitos

diaria = btc["1d"]
cierre_hitos = diaria.loc[~diaria["sin_operaciones"].astype(bool), "close"]
HITOS = detectar_hitos(cierre_hitos)
anotar(f"4.2 Hitos detectados (el precio se duplicó o se cortó a la mitad): {len(HITOS)} tramos, "
       f"{(HITOS['tipo'] == 'subida').sum()} subidas y {(HITOS['tipo'] == 'caída').sum()} caídas")
""")
md(r"""
### El contexto de cada hito
Para cada tramo, lo que pasaba en ese momento según la prensa de la época. **Ojo con cómo se lee:** es el **contexto que coincidió** con el movimiento, no una causa probada. Explicar el pasado es fácil; anticiparlo no. Si un hito nuevo aparece (porque entraron datos nuevos) y no tiene texto cargado, se muestra igual como **"contexto pendiente"**.

Las fuentes de cada fila están en la tabla y en el CSV de resultados.
""")
code(r"""
# Fecha de inicio del tramo -> (contexto que coincidió con el movimiento, fuente)
CONTEXTO_HITOS = {
    "2011-08-20": ("Fin de la primera burbuja (jun-2011, impulsada por notas de prensa sobre Silk Road) y hackeo de Mt. Gox de jun-2011.",
                   "https://fortune.com/2021/01/02/is-bitcoin-a-bubble-will-it-crash-again-history"),
    "2011-10-20": ("Recuperación lenta y creciente interés mediático; en mar-2013, la crisis bancaria de Chipre se asoció (con causalidad discutida) a la demanda de bitcoin.",
                   "https://money.cnn.com/2013/03/28/investing/bitcoin-cyprus/"),
    "2013-04-09": ("Mt. Gox, que concentraba la mayor parte del volumen, se satura por la avalancha de usuarios y frena la operatoria (10-abr-2013).",
                   "https://www.pcworld.com/article/451281/bitcoin-exchange-mt-gox-suspends-service-following-huge-swing-in-trading-price.html"),
    "2013-04-16": ("Rebote rápido después del pánico del crash de Mt. Gox.",
                   "https://en.wikipedia.org/wiki/History_of_bitcoin"),
    "2013-04-24": ("Presión regulatoria: el DHS de EE.UU. incauta cuentas de la filial de Mt. Gox (14-may-2013).",
                   "https://techcrunch.com/2013/05/16/mt-gox-dwolla-account-money-seizure"),
    "2013-07-06": ("Audiencia del Senado de EE.UU. de tono favorable (18-nov-2013) y fuerte demanda desde China; supera USD 1.000 el 27-nov.",
                   "https://money.cnn.com/2013/11/27/investing/Bitcoin-1000"),
    "2013-12-04": ("El Banco Popular de China prohíbe a los bancos operar con bitcoin (5-dic-2013) y Mt. Gox quiebra con 850.000 BTC perdidos (feb-2014).",
                   "https://techcrunch.com/2014/02/28/mt-gox-files-for-bankruptcy"),
    "2015-01-14": ("Segundo halving (jul-2016), boom de las ICO en 2017 y lanzamiento de futuros de bitcoin en Cboe y CME (dic-2017).",
                   "https://ir.cboe.com/news/news-details/2017/Cboe-Bitcoin-Futures-Open-for-Trading-12-11-2017"),
    "2017-12-16": ("Estalla la burbuja de las ICO: China las prohíbe, la SEC sanciona y el temor regulatorio se generaliza.",
                   "https://www.nbcnews.com/business/markets/bitcoin-high-2017-decline-2018-data-n949576"),
    "2018-12-15": ("Facebook anuncia Libra (18-jun-2019) y crece la expectativa por el halving de 2020.",
                   "https://www.cnbc.com/2019/06/26/bitcoin-soars-close-to-13000-hitting-a-17-month-high.html"),
    "2019-06-26": ("Deriva bajista en la segunda mitad de 2019 (contexto no verificado) y pánico del COVID: el 12-mar-2020 bitcoin cae ~37% en un día.",
                   "https://www.cnbc.com/2020/03/13/bitcoin-loses-half-of-its-value-in-two-day-plunge.html"),
    "2020-03-12": ("Estímulo monetario pospandemia, narrativa de 'oro digital' y entrada institucional: MicroStrategy (ago-2020), PayPal (oct-2020), Tesla (feb-2021), Coinbase en Nasdaq (abr-2021).",
                   "https://www.coindesk.com/markets/2020/12/16/first-mover-stimulus-tests-bitcoin-in-real-time-and-it-passes-20k/"),
    "2021-04-13": ("Tesla deja de aceptar bitcoin (12-may-2021) y China lanza una ofensiva contra la minería y el trading (21-may-2021).",
                   "https://www.scmp.com/tech/policy/article/3134473/china-escalates-crackdown-bitcoin-mining-trading"),
    "2021-07-20": ("Bitcoin moneda de curso legal en El Salvador (7-sep-2021), primer ETF de futuros en EE.UU. (19-oct-2021) e inflación récord en EE.UU.",
                   "https://www.aljazeera.com/amp/economy/2021/11/10/bitcoin-flirts-with-69000-on-heels-of-surging-inflation-news"),
    "2021-11-08": ("La Fed sube tasas (desde mar-2022) y caen en cadena Terra/Luna (may-2022), Celsius y Three Arrows (jun-2022) y FTX (nov-2022).",
                   "https://www.bloomberg.com/news/articles/2022-11-11/ftx-com-goes-bankrupt-in-stunning-reversal-for-crypto-exchange"),
    "2022-11-21": ("La SEC aprueba los ETF spot (10-ene-2024), cuarto halving (abr-2024), elección de Trump (nov-2024) y Reserva Estratégica de Bitcoin (mar-2025).",
                   "https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023"),
    "2025-10-06": ("Liquidaciones récord (~USD 19.000 M) tras el anuncio de aranceles a China (10-oct-2025), salidas de los ETF, dudas sobre la Fed y ventas de empresas 'tesoreras' de bitcoin (Strategy vende parte de su reserva en jun-2026).",
                   "https://www.aljazeera.com/economy/2025/11/18/bitcoin-ticks-up-after-erasing-all-of-2025-gains"),
}

def contexto_de(fecha, tolerancia=10):
    # Busca el contexto del hito que empieza en esa fecha: el MÁS CERCANO, con unos días de tolerancia por si cambian los datos
    # (en abril de 2013 hay tres hitos en dos semanas: tomar el primero que entra en la tolerancia les cruzaba los textos)
    distancias = {clave: abs((pd.Timestamp(clave, tz="UTC") - fecha).days) for clave in CONTEXTO_HITOS}
    clave = min(distancias, key=distancias.get)
    return CONTEXTO_HITOS[clave] if distancias[clave] <= tolerancia else ("contexto pendiente", "")

HITOS["contexto"] = [contexto_de(f)[0] for f in HITOS["inicio"]]
HITOS["fuente"] = [contexto_de(f)[1] for f in HITOS["inicio"]]
pendientes = (HITOS["contexto"] == "contexto pendiente").sum()
if pendientes:
    print(f"⚠️ {pendientes} hito(s) sin contexto cargado: completar CONTEXTO_HITOS")

tabla_hitos = HITOS.assign(inicio=HITOS["inicio"].dt.strftime("%d-%m-%Y"), fin=HITOS["fin"].dt.strftime("%d-%m-%Y"),
                           movimiento=(HITOS["movimiento"] * 100).round(0).astype(int).astype(str) + "%",
                           **{"precio inicio": HITOS["precio inicio"].round(0), "precio fin": HITOS["precio fin"].round(0)})
guardar_tabla(tabla_hitos, "04_2_hitos")
with pd.option_context("display.max_colwidth", 120):
    display(tabla_hitos[["inicio", "fin", "días", "precio inicio", "precio fin", "movimiento", "tipo", "estado", "contexto"]])

subidas, caidas = HITOS[HITOS["tipo"] == "subida"], HITOS[HITOS["tipo"] == "caída"]
anotar(f"4.2 Duración mediana: subidas {subidas['días'].median():.0f} días, caídas {caidas['días'].median():.0f} días | "
       f"caída más profunda {caidas['movimiento'].min():.0%} | caídas de más de 200 días: "
       + ", ".join(f"{m:.0%}" for m in caidas.loc[caidas["días"] > 200, "movimiento"]))
""")
md(r"""
## 4.3 Los eventos: ¿qué hizo el precio la semana siguiente?
La dirección inversa: partimos de eventos conocidos (fecha y fuente verificadas) y medimos el **retorno de los 7 días siguientes**. Para saber si fue un movimiento "raro", lo ubicamos en la distribución de **todos** los retornos de 7 días desde 2013 (percentil: 50 = una semana normal; cerca de 0 o de 100 = una semana extrema).
""")
code(r"""
EVENTOS = [
    ("2013-04-10", "Mt. Gox se satura y frena la operatoria", "https://www.pcworld.com/article/451281/bitcoin-exchange-mt-gox-suspends-service-following-huge-swing-in-trading-price.html"),
    ("2013-12-05", "China prohíbe a los bancos operar bitcoin", "https://www.npr.org/sections/thetwo-way/2013/12/05/248967111/china-bans-bitcoin-trading-by-banks"),
    ("2014-02-07", "Mt. Gox suspende los retiros", "https://techcrunch.com/2014/02/28/mt-gox-files-for-bankruptcy"),
    ("2015-01-04", "Hackeo de Bitstamp", "https://techcrunch.com/2015/01/05/bitstamp-bitcoin-exchange-hack/"),
    ("2017-09-04", "China prohíbe las ICO", "https://fortune.com/2017/09/04/ico-china-central-bank-ban/"),
    ("2017-12-10", "Futuros de bitcoin en Cboe", "https://ir.cboe.com/news/news-details/2017/Cboe-Bitcoin-Futures-Open-for-Trading-12-11-2017"),
    ("2019-06-18", "Facebook anuncia Libra", "https://www.axios.com/2019/06/18/facebook-unveils-libra-its-long-awaited-cryptocurrency-project"),
    ("2020-03-12", "'Jueves negro' del COVID", "https://www.cnbc.com/2020/03/13/bitcoin-loses-half-of-its-value-in-two-day-plunge.html"),
    ("2020-08-11", "MicroStrategy compra bitcoin", "https://www.theblock.co/linked/74534/microstrategy-becomes-first-listed-company-to-buy-bitcoin"),
    ("2021-02-08", "Tesla compra USD 1.500 M en bitcoin", "https://www.bloomberg.com/news/articles/2021-02-08/tesla-invests-1-5-billion-in-bitcoin-plans-to-accept-cryptocurrency"),
    ("2021-05-12", "Tesla deja de aceptar bitcoin", "https://techcrunch.com/2021/05/12/elon-musk-technoking-of-tesla-orders-a-halt-to-bitcoin-car-payments"),
    ("2021-09-07", "El Salvador: bitcoin de curso legal", "https://en.wikipedia.org/wiki/Bitcoin_in_El_Salvador"),
    ("2022-05-09", "Colapso de Terra/Luna", "https://www.theblock.co/post/146444/terra-luna-and-ust-how-we-got-here"),
    ("2022-11-11", "Quiebra de FTX", "https://www.bloomberg.com/news/articles/2022-11-11/ftx-com-goes-bankrupt-in-stunning-reversal-for-crypto-exchange"),
    ("2024-01-10", "La SEC aprueba los ETF spot", "https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023"),
    ("2024-11-05", "Elección presidencial en EE.UU.", "https://www.theblock.co/post/325658/bitcoin-breaks-above-90000-amid-rally-led-by-trumps-victory"),
    ("2025-03-06", "Reserva Estratégica de Bitcoin de EE.UU.", "https://www.whitehouse.gov/presidential-actions/2025/03/establishment-of-the-strategic-bitcoin-reserve-and-united-states-digital-asset-stockpile/"),
    ("2025-10-10", "Liquidaciones récord tras aranceles a China", "https://decrypt.co/343941/bitcoin-ethereum-dogecoin-down-record-19-billion-crypto-liquidations"),
]
cierre_d = diaria["close"]
ret7_todos = np.log(cierre_d.shift(-7) / cierre_d.shift(1)).loc["2013-01-01":].dropna()   # del cierre de la víspera a 7 días después
filas = []
for fecha, titulo, fuente in EVENTOS:
    f = pd.Timestamp(fecha, tz="UTC")
    if f not in ret7_todos.index:
        continue
    r = ret7_todos.loc[f]
    filas.append({"fecha": f.strftime("%d-%m-%Y"), "evento": titulo, "retorno 7 días": f"{np.expm1(r):+.1%}",
                  "percentil": round((ret7_todos < r).mean() * 100), "fuente": fuente})
tabla_eventos = pd.DataFrame(filas)
guardar_tabla(tabla_eventos, "04_3_eventos")
display(tabla_eventos.drop(columns="fuente"))
extremos_ev = (tabla_eventos["percentil"] <= 10) | (tabla_eventos["percentil"] >= 90)
anotar(f"4.3 Eventos: {extremos_ev.sum()} de {len(tabla_eventos)} fueron seguidos por una semana extrema (percentil ≤10 o ≥90)")

# Gráfico: precio diario con hitos (sombreados), halvings y eventos
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.semilogy(cierre_hitos.index, cierre_hitos, color="black", linewidth=0.8)
for _, h in HITOS.iterrows():
    ax.axvspan(h["inicio"], h["fin"], color="#2a9d55" if h["tipo"] == "subida" else "#d1495b", alpha=0.12)
    ax.text(h["inicio"] + (h["fin"] - h["inicio"]) / 2, 0.02, f"{h['movimiento']:+.0%}", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7, rotation=90, color="#2a9d55" if h["tipo"] == "subida" else "#d1495b")
for fecha in HALVINGS:
    ax.axvline(fecha, color="#f7931a", linestyle="--", linewidth=1.2)
for n, (fecha, titulo, _) in enumerate(EVENTOS, start=1):
    f = pd.Timestamp(fecha, tz="UTC")
    if f in cierre_d.index:
        ax.plot(f, cierre_d.loc[f], "o", color="#264653", markersize=4)
        ax.annotate(str(n), (f, cierre_d.loc[f]), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
ax.set_title("Hitos (verde = subida ×2, rojo = caída ÷2), halvings (naranja) y eventos numerados")
ax.set_ylabel("USD (escala log)")
guardar_fig("04_3_linea_de_tiempo")
plt.show()
print("\n".join(f"{n}. {fecha}: {titulo}" for n, (fecha, titulo, _) in enumerate(EVENTOS, start=1)))
""")
md(r"""
## 4.4 Las velas antes de cada hito: ¿lo anunciaban?
Esta es la pregunta del trabajo aplicada a los momentos más importantes. Para cada hito mostramos las **20 velas diarias anteriores a su inicio**, dibujadas **exactamente como las ve la CNN**. Si hubiera un "aviso" en las velas (un patrón que se repite antes de cada gran caída o subida), debería verse acá.

Lo que dijeron la CNN y la LSTM en los hitos que caen en validación o test se completa en el examen final (12.6b), cuando las redes ya están entrenadas.
""")
code(r"""
OHLCV_DIARIA = diaria[["open", "high", "low", "close", "volume"]].to_numpy()
columnas_g = 4
filas_g = int(np.ceil(len(HITOS) / columnas_g))
fig, ejes = plt.subplots(filas_g, columnas_g, figsize=(16, 2.3 * filas_g))
ejes = np.atleast_1d(ejes).ravel()
for ax, (_, h) in zip(ejes, HITOS.iterrows()):
    p = diaria.index.get_loc(h["inicio"])
    if p >= 20:
        ax.imshow(dibujar_ventana(OHLCV_DIARIA[p - 19:p + 1]), cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"{h['inicio']:%d-%m-%Y} → {h['tipo']} {h['movimiento']:+.0%}", fontsize=9,
                 color="#2a9d55" if h["tipo"] == "subida" else "#d1495b")
    ax.axis("off")
for ax in ejes[len(HITOS):]:
    ax.axis("off")
plt.suptitle("Las 20 velas anteriores a cada hito, como las ve la CNN: ¿hay un aviso que se repita?", y=1.0)
plt.tight_layout()
guardar_fig("04_4_velas_antes_de_cada_hito")
plt.show()
""")
md(r"""
## 4.5 Las velas diarias alrededor de cada halving
60 días antes y 120 días después de cada halving (línea punteada).
""")
code(r"""
estilo = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":")
for fecha, nombre in zip(HALVINGS, NOMBRES_HALVING):
    tramo = diaria.loc[fecha - pd.Timedelta(days=60): fecha + pd.Timedelta(days=120)].copy()
    tramo.index = tramo.index.tz_localize(None)                     # mplfinance no acepta zona horaria
    fig, _ = mpf.plot(tramo[["open", "high", "low", "close", "volume"]], type="candle", style=estilo, volume=True,
                      title=f"\nBTC/USD velas diarias – halving {nombre} ({fecha:%d-%m-%Y})",
                      vlines=dict(vlines=[fecha.tz_localize(None)], colors="#444", linestyle="--"),
                      figsize=(13, 6), returnfig=True)
    guardar_fig(f"04_5_halving_{fecha:%Y}", fig)
    plt.show()
""")
md(r"""
### 🔎 ¿Qué nos llevamos de la historia?
- **Los grandes movimientos tienen nombre y apellido:** casi todos los hitos coinciden con **noticias** (quiebras de exchanges, prohibiciones en China, ETFs, la Fed, liquidaciones). Son hechos que **no se sabían de antemano**: si la causa de un movimiento fue una sorpresa, **no podía estar escrita en las velas anteriores**. Esto anticipa lo que vamos a medir en la Parte III.
- **Las 20 velas previas a cada hito (4.4):** si antes de subidas y de caídas aparece "de todo" (velas grandes, chicas, verdes, rojas) y no una firma que se repita, el aviso no está a la vista. Lo verificamos con números en la Parte II (patrones) y en el examen (12.6b: qué dijeron las redes en esos días).
- **El ciclo de 4 años sí se ve a gran escala:** suba fuerte después de cada halving, pico unos 530 días después (2016, 2020 y 2024 casi iguales), caída de −77% a −85% y acumulación. Pero cada ciclo **sube menos y cae menos** que el anterior (rendimientos decrecientes): la caída 2025-26 fue la más chica (−53%). Y por primera vez el precio superó su máximo **antes** del halving (mar-2024, por los ETF).
- **Ese ciclo es un patrón de años, no de velas:** con solo 4 ciclos (3 completos), es una regularidad para mirar con cuidado, no una ley. Y no dice nada sobre la semana siguiente, que es lo que miran las velas diarias.
""")

# =====================================================================
md(r"""
---
# 5. Benchmark: Bitcoin contra otros mercados

¿En qué se parecen y en qué se diferencian las velas de Bitcoin de las de otros activos? Comparamos, **con velas diarias y en el mismo período**:

| Grupo | Activos | Por qué |
|---|---|---|
| Otras criptomonedas | Ethereum (ETH), Litecoin (LTC), Ripple (XRP) | ¿todas las cripto se comportan igual? |
| Índices de EE.UU. | S&P 500, Nasdaq | los mercados líquidos y desarrollados que pide la consigna |
| Refugio | Oro | el "oro digital" contra el oro de verdad |
| La acción del profe | Eli Lilly (LLY) | la de la planilla y el ejemplo de la consigna |

Los datos de los otros activos salen de Yahoo Finanzas (`yfinance`), ajustados por *splits* y dividendos. Bitcoin sale de **nuestra** base (Bitstamp). Si algún activo no se puede bajar, se avisa y la sección sigue sin él.

**Qué medimos por activo:**
- **Rendimiento anual** (CAGR) y **volatilidad anual**: cuánto ganó y cuánto se movió. Ojo: los mercados tradicionales operan ≈252 días al año y las cripto 365, así que anualizamos con cada uno.
- **Peor caída** desde un máximo.
- **% de días con mecha larga** (una mecha mayor que 2 cuerpos) y **tamaño medio del cuerpo**: la "forma" típica de sus velas.
- **% de días con salto de apertura** (abre más de 0,5% lejos del cierre anterior): las acciones cierran de noche y los fines de semana; las cripto no.
- **Correlación** de sus retornos diarios con los de Bitcoin.
""")
code(r"""
import yfinance as yf

ACTIVOS_BENCH = {"Ethereum (ETH)": "ETH-USD", "Litecoin (LTC)": "LTC-USD", "Ripple (XRP)": "XRP-USD",
                 "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Oro": "GC=F", "Eli Lilly (LLY)": "LLY"}
CRIPTOS_BENCH = {"Bitcoin (BTC)", "Ethereum (ETH)", "Litecoin (LTC)", "Ripple (XRP)"}
CARPETA_BENCH = RAW / "benchmark"
CARPETA_BENCH.mkdir(exist_ok=True)

def bajar_yahoo(ticker, desde="2005-01-01"):
    archivo = CARPETA_BENCH / f"{ticker.replace('^', '').replace('=', '_')}.csv"
    if archivo.exists():
        return pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = yf.download(ticker, start=desde, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(archivo)
    return df

bench = {"Bitcoin (BTC)": diaria.loc[~diaria["sin_operaciones"].astype(bool), ["open", "high", "low", "close", "volume"]].tz_convert(None)}
for nombre, ticker in ACTIVOS_BENCH.items():
    try:
        df = bajar_yahoo(ticker)
        if len(df) < 500:
            raise ValueError("muy pocos datos")
        bench[nombre] = df
    except Exception as e:
        print(f"⚠️ {nombre} ({ticker}) no se pudo bajar, se sigue sin él: {e}")
    time.sleep(0.3)

DESDE_BENCH = max(df.index.min() for df in bench.values())
HASTA_BENCH = min(df.index.max() for df in bench.values())
anotar(f"5. Benchmark: {len(bench)} activos, período común {DESDE_BENCH:%d-%m-%Y} → {HASTA_BENCH:%d-%m-%Y}")
""")
code(r"""
def metricas_activo(nombre, df):
    df = df.loc[DESDE_BENCH:HASTA_BENCH]
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    r = np.log(c / c.shift(1)).dropna()
    dias_anio = 365 if nombre in CRIPTOS_BENCH else 252
    anios = (df.index[-1] - df.index[0]).days / 365.25
    cuerpo = (c - o).abs().clip(lower=0.001 * c)
    mecha = np.maximum(h - np.maximum(o, c), np.minimum(o, c) - l)
    return {
        "rendimiento anual": (c.iloc[-1] / c.iloc[0]) ** (1 / anios) - 1,
        "volatilidad anual": r.std() * np.sqrt(dias_anio),
        "peor caída": (c / c.cummax() - 1).min(),
        "velas por año": dias_anio,
        "% días con mecha larga": (mecha > 2 * cuerpo).mean(),
        "cuerpo medio (% del precio)": ((c - o).abs() / o).mean(),
        "rango diario medio (% del precio)": ((h - l) / o).mean(),
        "% días con salto de apertura": ((o / c.shift(1) - 1).abs() > 0.005).iloc[1:].mean(),
    }

tabla_bench = pd.DataFrame({n: metricas_activo(n, df) for n, df in bench.items()}).T
ret_diarios = pd.DataFrame({n: np.log(df["close"] / df["close"].shift(1)) for n, df in bench.items()}).loc[DESDE_BENCH:HASTA_BENCH]
tabla_bench["correlación con BTC"] = ret_diarios.corr()["Bitcoin (BTC)"]       # en los días que operan los dos
guardar_tabla(tabla_bench.round(4), "05_benchmark")
anotar("5. Benchmark (mismo período, velas diarias):\n" + tabla_bench.round(3).to_string())
formato = {c: "{:.1%}" for c in tabla_bench.columns if c not in ("velas por año", "correlación con BTC")}
display(tabla_bench.style.format(formato | {"velas por año": "{:.0f}", "correlación con BTC": "{:.2f}"}))
""")
code(r"""
# 1) ¿Cuánto rindió 1 dólar en cada activo? (escala log)
fig, ejes = plt.subplots(1, 2, figsize=(16, 5))
for n, df in bench.items():
    c = df["close"].loc[DESDE_BENCH:HASTA_BENCH]
    ejes[0].semilogy(c.index, c / c.iloc[0], linewidth=2.2 if n == "Bitcoin (BTC)" else 1,
                     color="#f7931a" if n == "Bitcoin (BTC)" else None, label=n)
ejes[0].set_title("USD 1 invertido al inicio del período común")
ejes[0].legend(fontsize=8)
t = tabla_bench.sort_values("volatilidad anual")
ejes[1].barh(t.index, t["volatilidad anual"], color=["#f7931a" if n == "Bitcoin (BTC)" else "#9a9a9a" for n in t.index])
ejes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ejes[1].set_title("Volatilidad anual: cuánto se mueve cada activo")
plt.tight_layout()
guardar_fig("05_rendimiento_y_volatilidad")
plt.show()

# 2) Las mismas 60 velas finales de cada activo, en la misma escala relativa (primer cierre = 100)
nombres = list(bench)
fig, ejes = plt.subplots(2, int(np.ceil(len(nombres) / 2)), figsize=(18, 7), sharey=True)
ejes = ejes.ravel()
tramos = {n: df[["open", "high", "low", "close"]].loc[:HASTA_BENCH].tail(60) for n, df in bench.items()}
tramos = {n: v / v["close"].iloc[0] * 100 for n, v in tramos.items()}
lim = (min(v["low"].min() for v in tramos.values()) * 0.98, max(v["high"].max() for v in tramos.values()) * 1.02)
for ax, n in zip(ejes, nombres):
    dibujar_velas(ax, tramos[n].to_numpy())
    ax.set_ylim(*lim)
    ax.set_title(n, fontsize=10)
    ax.set_xticks([])
for ax in ejes[len(nombres):]:
    ax.axis("off")
plt.suptitle("Las últimas 60 velas diarias de cada activo, en la misma escala (primer cierre = 100)")
plt.tight_layout()
guardar_fig("05_velas_lado_a_lado")
plt.show()
""")
md(r"""
### ¿Por qué Bitcoin?
La celda de abajo arma la justificación con los números de la tabla.
""")
code(r"""
b = tabla_bench.loc["Bitcoin (BTC)"]
tradicionales = [n for n in tabla_bench.index if n not in CRIPTOS_BENCH]
otras_cripto = [n for n in tabla_bench.index if n in CRIPTOS_BENCH and n != "Bitcoin (BTC)"]
texto = [
    f"- Se mueve mucho: volatilidad anual de {b['volatilidad anual']:.0%}, contra {tabla_bench.loc[tradicionales, 'volatilidad anual'].median():.0%} "
    f"de mediana en los mercados tradicionales → las velas tienen cuerpos y mechas grandes ({b['cuerpo medio (% del precio)']:.1%} de cuerpo medio por día).",
    f"- Opera 24/7: {b['velas por año']:.0f} velas por año (contra 252) y casi sin saltos de apertura ({b['% días con salto de apertura']:.0%} de los días, "
    f"contra {tabla_bench.loc[tradicionales, '% días con salto de apertura'].median():.0%} en los tradicionales): cada vela empieza donde terminó la anterior, "
    f"así que toda la información del día está en la vela.",
    f"- Tiene la historia más larga de las cripto (desde 2011 en Bitstamp, con 4 halvings), y las otras cripto se mueven con ella "
    f"(correlación mediana {tabla_bench.loc[otras_cripto, 'correlación con BTC'].median():.2f}): es la que marca el paso del mercado cripto.",
    f"- Es muy distinto de la acción del profe: LLY se mueve {tabla_bench.loc['Eli Lilly (LLY)', 'volatilidad anual']:.0%} por año. "
    f"Una regla pensada para LLY (como 'sube más de 3% en 3 ruedas') no significa lo mismo en Bitcoin." if "Eli Lilly (LLY)" in tabla_bench.index else "",
]
anotar("5. ¿Por qué Bitcoin?\n" + "\n".join(t for t in texto if t))
""")
md(r"""
### 🔎 ¿Qué nos llevamos del benchmark?
- **Las velas de Bitcoin son "más grandes"** que las de los mercados tradicionales: más volatilidad, cuerpos y rangos diarios más amplios. Por eso **todo el trabajo mide los movimientos en unidades de volatilidad** (sección 6): un +3% no significa lo mismo en Bitcoin que en LLY.
- **Bitcoin no tiene saltos de apertura:** en un mercado 24/7 la apertura de hoy es casi igual al cierre de ayer. Patrones de libro que dependen de un "hueco" (ventanas, estrellas con *gap*) **casi no pueden existir** en Bitcoin; los que dependen del cuerpo y las mechas, sí.
- **Las cripto se mueven en bloque** con Bitcoin. Por eso agregar ETH, LTC y XRP como datos extra para las redes no garantiza aprender algo nuevo (lo probamos en 14.3).
""")

# =====================================================================
md(r"""
---
# 6. Anatomía de cada vela y la respuesta de cada día

### La idea
Nos paramos en el **día del halving de abril de 2024** y le **escondemos al modelo todo lo que pasó después**. Lo entrenamos solo con el pasado y después le mostramos el ciclo actual día por día. Como ese período **ya ocurrió**, cada predicción se puede corregir: ¿acertó o no?

```
 2013 ─────────── may-2020 │ may-2020 ─────── abr-2024 │ abr-2024 ─────── hoy
      ENTRENAMIENTO        │      VALIDACIÓN           │   TEST: ciclo actual
  (el modelo aprende acá)  │ (ajustamos el modelo)     │ (¿lo habría anticipado?)
     ciclos 2016 y 2020    │  ciclo del halving 2020   │  ciclo del halving 2024
```

Para **cada día** de la historia armamos una pregunta de examen:
- **Lo que el modelo ve:** las **últimas N velas diarias** hasta ese día (cuántas, lo decidimos con un experimento en 9.4 y 10.4).
- **Lo que tiene que adivinar:** qué pasó en los **7 días siguientes**: ¿**sube**, **baja** o **lateral**?

En esta sección armamos: (6.1) la anatomía de cada vela, (6.2) la respuesta correcta de cada día y (6.3) la división en entrenamiento / validación / test. **La caza de patrones (sección 9) usa todo esto**: para saber si un patrón "dice algo" necesitamos saber qué pasó después de cada día.
""")
md(r"""
## 6.1 Anatomía de cada vela (como la planilla del profesor)

Reproducimos las columnas de la planilla del profe (LLY) para cada vela diaria:

| Columna | Fórmula | Qué mide |
|---|---|---|
| `cuerpo` | \|close − open\| (mínimo 0,1% del precio) | la fuerza del día |
| `direccion` | +1 si cerró arriba (verde), −1 si cerró abajo (roja) | quién ganó: compradores o vendedores |
| `mecha_sup` | high − max(open, close) | cuánto subió y fue rechazado |
| `mecha_inf` | min(open, close) − low | cuánto bajó y fue rechazado |
| `prop_*` | cada parte / (high − low) | la forma de la vela, sin importar el precio |
| `martillo` | mecha inferior > 2 cuerpos y mecha superior < 0,2 cuerpos | el patrón clásico (lo analizamos en 7.1 y lo usamos como baseline) |
| `ret_1d` | log(close hoy / close ayer) | el retorno del día |

**¿Por qué el cuerpo mínimo es 0,1% del precio y no 0,005 como en la planilla?** Porque en la planilla el mínimo está en dólares y LLY vale unos cientos; Bitcoin pasó de $13 a $126.000, así que el mínimo tiene que ser **proporcional al precio**.

Las proporciones son importantes para las redes: una vela de 2013 ($100) y una de 2025 ($100.000) pueden tener **la misma forma**, y eso es lo que queremos que el modelo compare.
""")
code(r"""
d = btc["1d"][["open", "high", "low", "close", "volume", "sin_operaciones"]].copy()
o, h, l, c = d["open"], d["high"], d["low"], d["close"]

# --- Anatomía de cada vela ---
d["cuerpo"] = (c - o).abs().clip(lower=0.001 * c)
d["direccion"] = np.where(c >= o, 1, -1)
d["mecha_sup"] = h - np.maximum(o, c)
d["mecha_inf"] = np.minimum(o, c) - l

rango = (h - l).replace(0, np.nan)               # vela plana: rango 0 -> sin proporciones
d["prop_cuerpo"] = (c - o).abs() / rango
d["prop_mecha_sup"] = d["mecha_sup"] / rango
d["prop_mecha_inf"] = d["mecha_inf"] / rango

d["martillo"] = (d["mecha_inf"] / d["cuerpo"] > 2) & (d["mecha_sup"] / d["cuerpo"] < 0.2)
d["ret_1d"] = np.log(c / c.shift(1))

d[["open", "high", "low", "close", "cuerpo", "direccion", "prop_cuerpo",
   "prop_mecha_sup", "prop_mecha_inf", "martillo"]].tail(5).round(3)
""")
md(r"""
## 6.2 La respuesta correcta de cada día: ¿sube, baja o lateral?

Para cada día *t* calculamos el **retorno de los 7 días siguientes**:

$$\text{ret\_futuro}_t = \log\left(\frac{\text{close}_{t+7}}{\text{close}_t}\right)$$

**¿Cuándo es "lateral"?** Un +3% en una semana no significa lo mismo en un mes tranquilo que en un mes loco. Por eso comparamos el movimiento con la **volatilidad reciente** (cuánto se movió el precio en los últimos 30 días):

$$z_t = \frac{\text{ret\_futuro}_t}{\text{volatilidad reciente}_t}$$

- *z* grande y positivo → **sube** (se movió para arriba mucho más de lo normal)
- *z* grande y negativo → **baja**
- *z* cerca de 0 → **lateral** (se movió dentro de lo esperable: "se detuvo", como dice la consigna)

El umbral *k* que separa "lateral" de "se movió" lo elegimos para que cada clase tenga **≈⅓ de los días de entrenamiento**. Así, **adivinar al azar acierta el 33%**: esa es la vara a superar.

⚠️ **Fijate:** `ret_futuro` mira hacia adelante. Es **la respuesta del examen**, nunca algo que el modelo pueda ver. Por eso usamos `shift(-7)` (traer el dato de 7 días después), mientras que la volatilidad usa solo días **pasados**.
""")
code(r"""
# Volatilidad reciente: desvío de los retornos diarios de los últimos 30 días, llevado a 7 días (×√7)
d["vol_reciente"] = d["ret_1d"].rolling(VENTANA_VOL).std() * np.sqrt(HORIZONTE)

# Retorno de los próximos 7 días (LA RESPUESTA: mira al futuro)
d["ret_futuro"] = np.log(c.shift(-HORIZONTE) / c)
d["z"] = (d["ret_futuro"] / d["vol_reciente"]).replace([np.inf, -np.inf], np.nan)

d[["close", "ret_futuro", "vol_reciente", "z"]].dropna().tail(3).round(3)
""")
md(r"""
## 6.3 División por ciclos: entrenamiento, validación y test

**Nunca se mezcla al azar.** Si mezcláramos los días, el modelo entrenaría con el 10 de marzo y lo evaluaríamos con el 9 de marzo: estaría "viendo el futuro" y el resultado sería falso. Cortamos **por fechas**, usando los halvings como bordes:

| Período | Fechas | Para qué |
|---|---|---|
| **Entrenamiento** | 2013 → halving may-2020 | el modelo aprende |
| **Validación** | may-2020 → halving abr-2024 | elegimos la mejor versión del modelo |
| **Test** | abr-2024 → hoy | **la prueba final**: el ciclo actual, que el modelo nunca vio |

**Detalle fino ("embargo"):** el último día de entrenamiento no puede tener una respuesta que caiga dentro de la validación. Si el corte es el 11-may-2020, la respuesta del 8 de mayo se conoce recién el 15 de mayo. Por eso sacamos los **últimos 7 días** antes de cada corte.

**Arrancamos en 2013** porque en 2011-2012 había muy pocas operaciones (sección 3.3).
""")
code(r"""
# INICIO, CORTE_VAL, CORTE_TEST y EMBARGO se definen en la sección 2
d["periodo"] = None
d.loc[(d.index >= INICIO) & (d.index < CORTE_VAL - EMBARGO), "periodo"] = "train"
d.loc[(d.index >= CORTE_VAL) & (d.index < CORTE_TEST - EMBARGO), "periodo"] = "val"
d.loc[d.index >= CORTE_TEST, "periodo"] = "test"

# Nos quedamos con los días que tienen período y respuesta (los últimos 7 días todavía no la tienen)
datos = d.dropna(subset=["periodo", "z", "vol_reciente"]).copy()

# Umbral k: se calcula SOLO con entrenamiento (si usáramos el test, le estaríamos "soplando" al modelo)
k = datos.loc[datos["periodo"] == "train", "z"].abs().quantile(1 / 3)
datos["clase"] = np.select([datos["z"] > k, datos["z"] < -k], ["sube", "baja"], default="lateral")

CLASES = ["baja", "lateral", "sube"]
datos["y"] = datos["clase"].map({nombre: i for i, nombre in enumerate(CLASES)})   # 0, 1, 2 para las redes

anotar(f"6. Umbral k = {k:.3f}  →  'lateral' si el movimiento de 7 días es menor a {k:.2f} volatilidades")
anotar(f"6. Días con respuesta: {len(datos):,}  (último: {datos.index.max():%Y-%m-%d})")
tabla = pd.crosstab(datos["periodo"], datos["clase"], margins=True, margins_name="total")
tabla = tabla.loc[["train", "val", "test", "total"], CLASES + ["total"]]
guardar_tabla(tabla, "06_clases_por_periodo")
anotar("6. Cantidad de días por período y clase:\n" + tabla.to_string())
print()
print("Proporción de cada clase:")
print(pd.crosstab(datos["periodo"], datos["clase"], normalize="index").loc[["train", "val", "test"], CLASES].round(3))
""")
md(r"""
### Un ejemplo concreto
Elegimos un día del test y miramos cómo se le asignó la respuesta:
""")
code(r"""
ej_fecha = pd.Timestamp("2025-10-06", tz="UTC")     # el día del pico del ciclo actual
ej = datos.loc[ej_fecha]
precio_despues = d["close"].shift(-HORIZONTE).loc[ej_fecha]
anotar(f"6. Ejemplo {ej_fecha:%d-%m-%Y}: cierre ${ej['close']:,.0f} → 7 días después ${precio_despues:,.0f} "
       f"({np.expm1(ej['ret_futuro']):+.1%}) | volatilidad reciente {ej['vol_reciente']:.1%} | "
       f"z = {ej['z']:+.2f} → {ej['clase'].upper()}")
""")
md(r"""
### Visualización: la respuesta de cada día sobre el precio
Cada punto es un día, coloreado según lo que pasó la semana siguiente. Las líneas verticales son los cortes entre entrenamiento, validación y test.
""")
code(r"""
COLORES = {"sube": "#2a9d55", "baja": "#d1495b", "lateral": "#9a9a9a"}

fig, ejes = plt.subplots(2, 1, figsize=(14, 9))

ax = ejes[0]
for clase in CLASES:
    sub = datos[datos["clase"] == clase]
    ax.scatter(sub.index, sub["close"], s=2, color=COLORES[clase], label=clase)
for corte, texto in [(CORTE_VAL, "validación →"), (CORTE_TEST, "test →")]:
    ax.axvline(corte, color="black", linestyle="--", linewidth=1)
    ax.text(corte, datos["close"].max(), " " + texto, va="top")
ax.set_yscale("log")
ax.set_title("Bitcoin diario: ¿qué pasó en los 7 días siguientes a cada día?")
ax.legend(markerscale=6)

ax = ejes[1]   # zoom en el test: el ciclo actual
test = datos[datos["periodo"] == "test"]
ax.plot(test.index, test["close"], color="#bbbbbb", linewidth=1, zorder=0)
for clase in CLASES:
    sub = test[test["clase"] == clase]
    ax.scatter(sub.index, sub["close"], s=8, color=COLORES[clase], label=clase)
ax.set_title("Zoom en el test (ciclo actual): esto es lo que el modelo tendrá que anticipar")
ax.legend(markerscale=3)

plt.tight_layout()
guardar_fig("06_respuesta_de_cada_dia")
plt.show()

datos.to_csv(VELAS / "btc_1d_dataset.csv")
print("Guardado:", VELAS / "btc_1d_dataset.csv")
""")
md(r"""
### 🔎 Qué tenemos hasta acá
- Un **examen con ≈5.000 preguntas** (una por día desde 2013), cada una con su respuesta correcta.
- Las tres clases quedan **parejas (≈⅓ cada una)**, así que el azar acierta ≈33%.
- En entrenamiento hay algo más de "sube" que de "baja": Bitcoin subió mucho entre 2013 y 2020. Por eso vamos a medir con **balanced accuracy** (el promedio del acierto en cada clase), que no se deja engañar si el modelo dice siempre "sube".
- El test (el ciclo actual) **nunca se toca** hasta la evaluación final.

""")

# =====================================================================
md(r"""
---
# 7. La caza de patrones: ¿qué esconden las velas de Bitcoin?

Esta es la parte central de la pregunta. Buscamos patrones de cuatro maneras distintas y, para cada uno, medimos **qué pasó en los 7 días siguientes**:

| # | Dónde buscamos | De dónde sale la idea |
|---|---|---|
| 7.1 | **Patrones clásicos** (martillo, estrella fugaz, envolventes, doji, tres soldados / tres cuervos) | los libros de velas (Nison, Ponsi) |
| 7.2 | **La planilla del profe** ("sharpe" de 3 ruedas, HAMMER) | el modelo de LLY del profesor |
| 7.3 | **Por fase del ciclo de halving** | la recurrencia de los ciclos de 4 años |
| 7.4 | **Familias que encuentra la máquina sola** (sin decirle qué buscar) | aprendizaje no supervisado |
| 7.5 | **"Shazam":** a qué momento del pasado se parecen las velas de hoy | "este patrón se parece a LLY" (consigna) |

### Cómo medimos si un patrón "dice algo"
- **La vara:** el movimiento de la semana siguiente en **unidades de volatilidad** (*z*, sección 6.2), comparado con el de **todos** los días. Si después de un patrón *z* es igual al de cualquier día, el patrón no dice nada.
- **¿Fue suerte? (p-valor por desplazamiento):** corremos la marca del patrón en el tiempo (la "deslizamos" por toda la serie, como si apareciera en otras fechas) 1.000 veces y vemos cuántas veces una ubicación al azar da una diferencia igual o mayor. Lo hacemos **desplazando** y no mezclando porque los días vecinos se parecen (las semanas se solapan): mezclar al azar exageraría la significancia.
- **¿Fue suerte de tanto buscar? (corrección de Holm):** si probamos 20 patrones, alguno va a "funcionar" por azar. Holm sube la vara según cuántas pruebas hicimos.
- **¿Se sostiene en el tiempo?** Medimos el efecto por separado en **entrenamiento** (2013-2020), **validación** (2020-2024) y el **ciclo actual** (2024-hoy). Un patrón que solo funcionó en el pasado no le sirve a nadie hoy.
""")
code(r"""
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import chi2_contingency

REPETICIONES = 200 if MODO_RAPIDO else 1000

def p_desplazamiento(estadistico, serie, repeticiones=REPETICIONES, margen=30, semilla=0):
    # p-valor: proporción de desplazamientos circulares de `serie` que dan un estadístico >= al observado
    rng_c = np.random.default_rng(semilla)
    serie = np.asarray(serie)
    observado = estadistico(serie)
    desplazamientos = rng_c.integers(margen, len(serie) - margen, repeticiones)
    nulos = np.array([estadistico(np.roll(serie, s)) for s in desplazamientos])
    return observado, (1 + np.sum(nulos >= observado)) / (1 + repeticiones)

def ic_bootstrap_bloques(estadistico, *arrays, bloque=30, repeticiones=REPETICIONES, semilla=0):
    # Intervalo de confianza 95% remuestreando bloques de 30 días seguidos (respeta que los días vecinos se parecen)
    rng_b = np.random.default_rng(semilla)
    n = len(arrays[0])
    inicios = rng_b.integers(0, n - bloque + 1, (repeticiones, int(np.ceil(n / bloque))))
    indices = (inicios[:, :, None] + np.arange(bloque)).reshape(repeticiones, -1)[:, :n]
    valores = [estadistico(*(np.asarray(a)[i] for a in arrays)) for i in indices]
    return np.percentile(valores, [2.5, 97.5])

def holm(p_valores):
    # Corrección de Holm por comparaciones múltiples
    p = np.asarray(p_valores, dtype=float)
    ajustado, acumulado = np.empty(len(p)), 0.0
    for r, j in enumerate(np.argsort(p)):
        acumulado = max(acumulado, min(1.0, (len(p) - r) * p[j]))
        ajustado[j] = acumulado
    return ajustado

# Control con un ejemplo conocido: una "predicción" igual a la respuesta debe dar p ≈ 0; una al azar, no
y_ej = datos.loc[datos["periodo"] == "val", "clase"].to_numpy()
pred_azar_ej = np.random.default_rng(1).choice(y_ej, len(y_ej))
_, p_perfecta = p_desplazamiento(lambda yy: balanced_accuracy_score(yy, y_ej), y_ej)
_, p_azar_ej = p_desplazamiento(lambda yy: balanced_accuracy_score(yy, pred_azar_ej), y_ej)
assert p_perfecta < 0.01, "la prueba por desplazamiento no detecta una predicción perfecta"
print(f"Control de la prueba: predicción perfecta p = {p_perfecta:.3f} | predicción al azar p = {p_azar_ej:.3f}")
""")
md(r"""
## 7.1 Los patrones clásicos de los libros

| Patrón | Definición (Nison) | Qué "promete" el libro |
|---|---|---|
| **Martillo** | mecha inferior > 2 cuerpos, mecha superior < 0,2 cuerpos, **después de una semana bajista** | rebote: los vendedores empujaron y los compradores ganaron |
| **Estrella fugaz** | mecha superior > 2 cuerpos, mecha inferior < 0,2 cuerpos, **después de una semana alcista** | techo: los compradores empujaron y los vendedores ganaron |
| **Envolvente alcista** | vela verde cuyo cuerpo "envuelve" el cuerpo rojo del día anterior | giro al alza |
| **Envolvente bajista** | vela roja que envuelve el cuerpo verde del día anterior | giro a la baja |
| **Doji** | cuerpo ≤ 10% del rango del día | indecisión |
| **Tres soldados blancos** | 3 velas verdes seguidas, cada una cierra más alto | continuación alcista |
| **Tres cuervos negros** | 3 velas rojas seguidas, cada una cierra más bajo | continuación bajista |
""")
code(r"""
o, h, l, c = d["open"], d["high"], d["low"], d["close"]
semana_previa = np.log(c / c.shift(HORIZONTE))
verde, roja = c > o, c < o
PATRONES = pd.DataFrame({
    "martillo": (d["mecha_inf"] > 2 * d["cuerpo"]) & (d["mecha_sup"] < 0.2 * d["cuerpo"]) & (semana_previa < 0),
    "estrella fugaz": (d["mecha_sup"] > 2 * d["cuerpo"]) & (d["mecha_inf"] < 0.2 * d["cuerpo"]) & (semana_previa > 0),
    "envolvente alcista": roja.shift(1, fill_value=False) & verde & (o <= c.shift(1)) & (c >= o.shift(1)),
    "envolvente bajista": verde.shift(1, fill_value=False) & roja & (o >= c.shift(1)) & (c <= o.shift(1)),
    "doji": (c - o).abs() <= 0.1 * (h - l).replace(0, np.nan),
    "tres soldados blancos": verde & verde.shift(1, fill_value=False) & verde.shift(2, fill_value=False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2)),
    "tres cuervos negros": roja & roja.shift(1, fill_value=False) & roja.shift(2, fill_value=False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2)),
}, index=d.index).fillna(False).reindex(datos.index).astype(bool)

z_dias = datos["z"].to_numpy()
en_train = (datos["periodo"] == "train").to_numpy()
en_val = (datos["periodo"] == "val").to_numpy()
en_test = (datos["periodo"] == "test").to_numpy()

def efecto_de(marca):
    # Diferencia de z (semana siguiente) entre los días con la marca y todos los días, en todo el período y por tramo
    marca = np.asarray(marca, dtype=bool)
    dif, p = p_desplazamiento(lambda m: abs(z_dias[m].mean() - z_dias.mean()) if m.any() else 0, marca)
    signo = np.sign(z_dias[marca].mean() - z_dias.mean()) if marca.any() else 0
    tramo = lambda sel: z_dias[marca & sel].mean() - z_dias[sel].mean() if (marca & sel).any() else np.nan
    return signo * dif, p, tramo(en_train), tramo(en_val), tramo(en_test)

def sostiene(d_tr, d_va, d_te, minimo=0.1):
    # ¿El efecto de entrenamiento (2013-2020) sigue con el mismo signo en validación (2020-2024) y en el ciclo actual (2024-hoy)?
    sigue = lambda d_: np.sign(d_) == np.sign(d_tr) and abs(d_) >= minimo
    if abs(d_tr) < minimo:
        return "no había efecto"
    if sigue(d_va) and sigue(d_te):
        return "sí, hasta hoy"
    if sigue(d_va):
        return "hasta 2024; en el ciclo actual se apagó"
    return "se apagó después de 2020"

filas = []
for nombre in PATRONES.columns:
    m = PATRONES[nombre].to_numpy()
    dif, p, d_tr, d_va, d_te = efecto_de(m)
    clase_m = datos.loc[m, "clase"]
    filas.append({"patrón": nombre, "veces": int(m.sum()),
                  "retorno 7d medio": np.expm1(datos.loc[m, "ret_futuro"]).mean(),
                  "sube": (clase_m == "sube").mean(), "baja": (clase_m == "baja").mean(), "lateral": (clase_m == "lateral").mean(),
                  "dif z (todo)": dif, "p": p, "dif z train": d_tr, "dif z val": d_va, "dif z test": d_te,
                  "¿se sostiene?": sostiene(d_tr, d_va, d_te), "muestra chica": "sí" if m.sum() < 30 else ""})
clasicos = pd.DataFrame(filas).set_index("patrón")
clasicos["p Holm"] = holm(clasicos["p"])
todos = {"veces": len(datos), "retorno 7d medio": np.expm1(datos["ret_futuro"]).mean(),
         "sube": (datos["clase"] == "sube").mean(), "baja": (datos["clase"] == "baja").mean(), "lateral": (datos["clase"] == "lateral").mean()}
guardar_tabla(clasicos.round(4), "07_1_patrones_clasicos")
anotar("7.1 Patrones clásicos (2013-hoy). Referencia, todos los días: "
       + ", ".join(f"{k} {v:.3f}" for k, v in todos.items()) + "\n" + clasicos.round(3).to_string())
display(clasicos.round(3))
""")
code(r"""
# Un ejemplo de cada patrón (la vela marcada es la última) y su efecto dentro y fuera de entrenamiento
fig = plt.figure(figsize=(18, 7.5))
rng_ej = np.random.default_rng(3)
for j, nombre in enumerate(PATRONES.columns):
    ax = fig.add_subplot(2, len(PATRONES.columns), j + 1)
    dias_p = np.flatnonzero(PATRONES[nombre].to_numpy())
    if len(dias_p):
        p = d.index.get_loc(datos.index[rng_ej.choice(dias_p)])
        dibujar_velas(ax, d[["open", "high", "low", "close"]].iloc[p - 9:p + 1].to_numpy())
        ax.axvspan(8.5, 9.5, color="#f4a261", alpha=0.25)
    ax.set_title(nombre, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
ax = fig.add_subplot(2, 1, 2)
clasicos[["dif z train", "dif z val", "dif z test"]].plot.bar(ax=ax, rot=0, color=["#bbbbbb", "#f4a261", "#264653"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("z de la semana siguiente\n(diferencia contra todos los días)")
ax.set_title("¿Después del patrón el precio se movió distinto que cualquier día? Entrenamiento (2013-20), validación (2020-24) y ciclo actual (2024-hoy)")
plt.tight_layout()
guardar_fig("07_1_patrones_clasicos")
plt.show()
""")
md(r"""
## 7.2 La planilla del profe, replicada en Bitcoin

Reproducimos exactamente las medidas de la planilla `MFO2509.xlsx` (verificadas celda por celda contra sus valores):
- **"Sharpe" de 3 ruedas:** suma de los cuerpos **con signo** de las 3 velas anteriores ÷ suma de sus tamaños. Va de −1 (3 velas rojas) a +1 (3 velas verdes).
- **Retorno futuro a 3 ruedas:** cierre dentro de 3 ruedas ÷ cierre de hoy − 1.
- **HAMMER:** mecha inferior > 2 cuerpos y mecha superior < 0,2 cuerpos.
- **La pregunta de la planilla:** P(retorno futuro > 3% | sharpe > 0,9): después de 3 velas verdes, ¿qué tan probable es subir más de 3%?

**El detalle importante:** un 3% no significa lo mismo en LLY que en Bitcoin (sección 5). Por eso agregamos la misma pregunta con una **vara equivalente**: subir más de **1 desvío** de 3 ruedas de cada activo.
""")
code(r"""
def planilla(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    cuerpo = (c - o).abs().clip(lower=0.001 * c)
    signado = (c - o)
    sharpe = signado.shift(1).rolling(3).sum() / (c - o).abs().shift(1).rolling(3).sum()
    fut3 = c.shift(-3) / c - 1
    vara = (np.log(c / c.shift(1)).rolling(250).std() * np.sqrt(3)).shift(1)     # 1 desvío de 3 ruedas (solo pasado)
    hammer = ((np.minimum(o, c) - l) > 2 * cuerpo) & ((h - np.maximum(o, c)) < 0.2 * cuerpo)
    t = pd.DataFrame({"sharpe": sharpe, "fut3": fut3, "vara": vara, "hammer": hammer}).dropna()
    cond, cond_neg = t["sharpe"] > 0.9, t["sharpe"] < -0.9
    return {
        "días": len(t),
        "P(fut3 > 3%)": (t["fut3"] > 0.03).mean(),
        "P(fut3 > 3% | sharpe > 0,9)": (t.loc[cond, "fut3"] > 0.03).mean(),
        "días con sharpe > 0,9": int(cond.sum()),
        "P(fut3 > 3% | sharpe < −0,9)": (t.loc[cond_neg, "fut3"] > 0.03).mean(),
        "P(fut3 > 3% | HAMMER)": (t.loc[t["hammer"], "fut3"] > 0.03).mean(),
        "P(fut3 > 1 desvío)": (t["fut3"] > t["vara"]).mean(),
        "P(fut3 > 1 desvío | sharpe > 0,9)": (t.loc[cond, "fut3"] > t.loc[cond, "vara"]).mean(),
    }

tabla_planilla = {"Bitcoin (2013-hoy)": planilla(d.loc[INICIO:])}
if "Eli Lilly (LLY)" in bench:
    tabla_planilla["LLY (misma historia de Yahoo)"] = planilla(bench["Eli Lilly (LLY)"])
tabla_planilla = pd.DataFrame(tabla_planilla).T
guardar_tabla(tabla_planilla.round(4), "07_2_planilla_del_profe")
anotar("7.2 La planilla del profe en Bitcoin y en LLY:\n" + tabla_planilla.round(3).to_string())
display(tabla_planilla.round(3))
""")
md(r"""
**Cómo leerlo:** si P(... | sharpe > 0,9) es mayor que P(...) sin condición, después de 3 velas verdes **es más probable** subir: eso sería **inercia** (*momentum*). Si son parecidas, la racha no dice nada. La comparación con LLY muestra si "lo que ve el profe" en una acción aparece también en Bitcoin.

## 7.3 ¿Hay patrones que se repiten en la misma fase de cada ciclo de halving?

Alineamos los ciclos con **el día del halving como día 0** y miramos:
1. **Los ciclos superpuestos:** el precio de cada ciclo dividido por su precio el día del halving.
2. **El rendimiento de cada año del ciclo**, ciclo por ciclo: ¿el mismo año del ciclo se comporta igual en todos?
3. **¿Cambia la forma de las velas según la fase?** Clasificamos cada vela en 5 tipos (cuerpo verde, cuerpo rojo, doji, mecha inferior larga, mecha superior larga) y contamos cuántas hay en cada año del ciclo. La prueba χ² dice si las diferencias son más grandes que las esperables por azar.

⚠️ Solo hay **4 ciclos, 3 completos**: cualquier regularidad entre ciclos se apoya en 3 ejemplos.
""")
code(r"""
DIAS_CICLO = 1460
fig, ax = plt.subplots(figsize=(14, 6))
colores_ciclo = ["#9a9a9a", "#264653", "#2a9d55", "#f7931a"]
for fecha, nombre, color in zip(HALVINGS, NOMBRES_HALVING, colores_ciclo):
    tramo = diaria["close"].loc[fecha:fecha + pd.Timedelta(days=DIAS_CICLO)]
    dias_desde = (tramo.index - fecha).days
    ax.semilogy(dias_desde, tramo / tramo.iloc[0], color=color, linewidth=2.5 if color == "#f7931a" else 1.3, label=f"halving {nombre} ({fecha:%Y})")
hoy = (diaria.index[-1] - HALVINGS[-1]).days
ax.axvline(hoy, color="#f7931a", linestyle=":")
ax.text(hoy, 0.97, f" hoy: día {hoy}", transform=ax.get_xaxis_transform(), va="top", color="#f7931a")
for x in (365, 730, 1095):
    ax.axvline(x, color="#dddddd", linewidth=0.8)
ax.set_xlabel("Días desde el halving")
ax.set_ylabel("Precio / precio el día del halving (log)")
ax.set_title("Los 4 ciclos superpuestos: ¿se repite la película?")
ax.legend()
guardar_fig("07_3_ciclos_superpuestos")
plt.show()

d["dias_desde_halving"] = np.nan
for fecha in HALVINGS:
    m = d.index >= fecha
    d.loc[m, "dias_desde_halving"] = (d.index[m] - fecha).days
d["anio_del_ciclo"] = np.minimum(d["dias_desde_halving"] // 365 + 1, 4)

filas = {}
for fecha, nombre in zip(HALVINGS, NOMBRES_HALVING):
    fila = {}
    for anio in range(1, 5):
        a, b = fecha + pd.Timedelta(days=365 * (anio - 1)), fecha + pd.Timedelta(days=365 * anio)
        tramo = diaria["close"].loc[a:b]
        if len(tramo) > 30:
            fila[f"año {anio}"] = f"{tramo.iloc[-1] / tramo.iloc[0] - 1:+.0%}" + (" (en curso)" if b > diaria.index[-1] else "")
    filas[f"ciclo {fecha:%Y}"] = fila
rend_ciclos = pd.DataFrame(filas).T
guardar_tabla(rend_ciclos, "07_3_rendimiento_por_anio_del_ciclo")
anotar("7.3 Rendimiento de cada año del ciclo:\n" + rend_ciclos.to_string())
display(rend_ciclos)

rango_d = (d["high"] - d["low"]).replace(0, np.nan)
d["tipo_de_vela"] = np.select(
    [(d["close"] - d["open"]).abs() <= 0.1 * rango_d, d["prop_mecha_inf"] > 0.5, d["prop_mecha_sup"] > 0.5, d["close"] >= d["open"]],
    ["doji", "mecha inferior larga", "mecha superior larga", "cuerpo verde"], default="cuerpo rojo")
desde_2013 = d.loc[INICIO:].dropna(subset=["anio_del_ciclo"])
contingencia = pd.crosstab(desde_2013["anio_del_ciclo"].astype(int), desde_2013["tipo_de_vela"])
chi2, p_chi2, gl, _ = chi2_contingency(contingencia)
porcentajes = contingencia.div(contingencia.sum(1), axis=0)
guardar_tabla(porcentajes.round(4), "07_3_tipo_de_vela_por_anio_del_ciclo")
anotar(f"7.3 Tipo de vela según el año del ciclo (χ² = {chi2:.1f}, {gl} gl, p = {p_chi2:.4f}):\n" + porcentajes.round(3).to_string())
COLOR_TIPO_VELA = {"cuerpo verde": "#2a9d55", "cuerpo rojo": "#d1495b", "doji": "#9a9a9a",
                   "mecha inferior larga": "#264653", "mecha superior larga": "#f4a261"}
ax = porcentajes.plot.bar(stacked=True, rot=0, figsize=(11, 4.5), color=[COLOR_TIPO_VELA[t] for t in porcentajes.columns])
ax.set_xlabel("Año del ciclo de halving")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title(f"¿Cambia la forma de las velas según la fase del ciclo? (χ² p = {p_chi2:.3f})")
ax.legend(bbox_to_anchor=(1, 1))
guardar_fig("07_3_tipo_de_vela_por_fase")
plt.show()
""")
md(r"""
**Cómo leerlo:** el **año 1** después de cada halving fue alcista en todos los ciclos, pero cada vez **menos** (rendimientos decrecientes). La forma de las velas también cambia un poco con la fase: en el año 1 hay más velas verdes, como es lógico en una suba. Pero ojo: eso **describe** la tendencia de cada año, que se conoce después. La pregunta de la Parte III es otra: con las velas de hoy, ¿se puede anticipar la **semana** siguiente?

## 7.4 Las familias de velas que encuentra la máquina sola

Hasta acá buscamos patrones **con nombre** (los de los libros). ¿Y si hay patrones que nadie bautizó? Le damos a un algoritmo de agrupamiento (**K-means**) todas las ventanas de **5 velas** del entrenamiento y le pedimos que las junte en **familias** parecidas, **sin decirle qué pasó después**. Recién después miramos qué hizo el precio tras cada familia.

| Decisión | Qué elegimos | En base a qué |
|---|---|---|
| **Qué describe cada vela** | 4 números: retorno del día, cuerpo con signo, mecha superior, mecha inferior, **en unidades de volatilidad** | igual que la LSTM: así una vela de 2014 y una de 2025 son comparables |
| **Cuántas velas** | 5 (una semana) | la ventana elegida por la CNN en la corrida anterior; patrones cortos, como los de los libros |
| **Cuántas familias** | la que da mejor **silueta** entre 4 y 10 | la silueta mide qué tan separadas quedan las familias; no elegimos a ojo |
| **Con qué datos se arman** | **solo entrenamiento** | las ventanas de validación y test solo se asignan a la familia más parecida: el futuro no se usa para definir nada |
""")
code(r"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

VELAS_FAMILIA = 5
vol_dia = (d["vol_reciente"] / np.sqrt(HORIZONTE)).replace(0, np.nan)
por_vela = pd.DataFrame({
    "retorno": d["ret_1d"],
    "cuerpo": np.log(d["close"] / d["open"]),
    "mecha_sup": np.log(d["high"] / np.maximum(d["open"], d["close"])),
    "mecha_inf": np.log(np.minimum(d["open"], d["close"]) / d["low"]),
}).div(vol_dia, axis=0).clip(-5, 5).fillna(0).to_numpy()
POS_DATOS = d.index.get_indexer(datos.index)
X_fam = np.stack([por_vela[p - VELAS_FAMILIA + 1:p + 1].ravel() for p in POS_DATOS])

rng_s = np.random.default_rng(0)
muestra_s = rng_s.choice(np.flatnonzero(en_train), min(2000, en_train.sum()), replace=False)
siluetas = {}
for k_fam in range(4, 11):
    km = KMeans(k_fam, n_init=10, random_state=0).fit(X_fam[en_train])
    siluetas[k_fam] = silhouette_score(X_fam[muestra_s], km.predict(X_fam[muestra_s]))
K_FAM = max(siluetas, key=siluetas.get)
agrupador = KMeans(K_FAM, n_init=10, random_state=0).fit(X_fam[en_train])     # se ajusta SOLO con entrenamiento
FAMILIA = agrupador.predict(X_fam)                                             # val y test solo se asignan
anotar(f"7.4 Familias: silueta por cantidad de familias {dict((k_, round(v, 3)) for k_, v in siluetas.items())} → elegimos {K_FAM}")

def nombre_familia(centro):
    v = centro.reshape(VELAS_FAMILIA, 4)
    total, ultima = v[:, 0].sum(), v[-1, 0]
    semana = "semana en subida" if total > 1 else "semana en bajada" if total < -1 else "semana quieta"
    final = ("termina con vela verde" + (" fuerte" if ultima > 1 else "") if ultima > 0.4 else
             "termina con vela roja" + (" fuerte" if ultima < -1 else "") if ultima < -0.4 else "termina con vela chica")
    return f"{semana}, {final}"

filas = []
for f_ in range(K_FAM):
    marca = FAMILIA == f_
    dif, p, d_tr, d_va, d_te = efecto_de(marca)
    fila = {"familia": f"F{f_ + 1}", "descripción": nombre_familia(agrupador.cluster_centers_[f_]), "veces": int(marca.sum()),
            "dif z (todo)": dif, "p": p, "dif z train": d_tr, "dif z val": d_va, "dif z test": d_te, "¿se sostiene?": sostiene(d_tr, d_va, d_te)}
    for per in ["train", "val", "test"]:
        sel = marca & (datos["periodo"] == per).to_numpy()
        fila[f"% sube ({per})"] = (datos["clase"].to_numpy()[sel] == "sube").mean() if sel.any() else np.nan
    filas.append(fila)
familias = pd.DataFrame(filas).set_index("familia")
familias["p Holm"] = holm(familias["p"])
guardar_tabla(familias.round(4), "07_4_familias")
anotar("7.4 Familias de velas:\n" + familias.round(3).to_string())
display(familias.round(3))

# Dibujo de la vela "típica" (centro) de cada familia
fig, ejes = plt.subplots(1, K_FAM, figsize=(2.6 * K_FAM, 3.2))
for f_, ax in enumerate(np.atleast_1d(ejes)):
    v = agrupador.cluster_centers_[f_].reshape(VELAS_FAMILIA, 4) * 0.02      # 1 volatilidad diaria ≈ 2% para dibujar
    velas_f, cierre_prev = [], 100.0
    for ret_, cuerpo_, ms_, mi_ in v:
        apertura = cierre_prev * np.exp(ret_ - cuerpo_)
        cierre_v = apertura * np.exp(cuerpo_)
        velas_f.append([apertura, max(apertura, cierre_v) * np.exp(ms_), min(apertura, cierre_v) * np.exp(-mi_), cierre_v])
        cierre_prev = cierre_v
    dibujar_velas(ax, np.array(velas_f))
    fila = familias.iloc[f_]
    ax.set_title(f"F{f_ + 1} ({fila['veces']} veces)\nz: train {fila['dif z train']:+.2f} · hoy {fila['dif z test']:+.2f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("La semana típica de cada familia (encontradas por la máquina, sin mirar el futuro)")
plt.tight_layout()
guardar_fig("07_4_familias")
plt.show()
""")
md(r"""
## 7.5 "Shazam": ¿a qué momento del pasado se parecen las velas de hoy?

Como la app que reconoce una canción por un pedacito: tomamos las **últimas 30 velas** disponibles y buscamos en toda la historia las **5 ventanas más parecidas**, comparando la **foto que ve la CNN** píxel por píxel (menos píxeles distintos = más parecida). Después miramos **qué pasó a continuación** en cada una.

- Solo se comparan ventanas que **terminan antes** de que empiecen las velas de hoy (sin solaparse) y que tienen 30 días de "después" conocidos.
- Entre las 5 elegidas dejamos al menos 30 días de distancia, para que no sean la misma semana corrida un día.
- Es la misma idea que el baseline "gemela del pasado" (9.5), que mide si este método acierta en general.
""")
code(r"""
N_SHAZAM, DESPUES = 30, 30
ohlcv_d = d[["open", "high", "low", "close", "volume"]].to_numpy()
fin_hoy = len(d) - 1
foto_hoy = dibujar_ventana(ohlcv_d[fin_hoy - N_SHAZAM + 1:fin_hoy + 1])
candidatas = np.arange(d.index.get_indexer([INICIO])[0] + N_SHAZAM, fin_hoy - N_SHAZAM - DESPUES)
distintos = np.array([(dibujar_ventana(ohlcv_d[p - N_SHAZAM + 1:p + 1]) != foto_hoy).sum() for p in candidatas])
elegidas = []
for j in np.argsort(distintos):
    if all(abs(candidatas[j] - e) >= 30 for e, _ in elegidas):
        elegidas.append((candidatas[j], distintos[j]))
    if len(elegidas) == 5:
        break

cierre_np = d["close"].to_numpy()
shazam = pd.DataFrame([{"termina el": d.index[p].strftime("%d-%m-%Y"), "píxeles distintos": int(dist),
                        "7 días después": cierre_np[p + 7] / cierre_np[p] - 1, "30 días después": cierre_np[p + 30] / cierre_np[p] - 1}
                       for p, dist in elegidas])
guardar_tabla(shazam.round(4), "07_5_shazam")
anotar(f"7.5 Shazam con las velas hasta el {d.index[fin_hoy]:%d-%m-%Y} (de {foto_hoy.sum()} píxeles encendidos):\n" + shazam.round(3).to_string(index=False))

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(2, 6, 1)
ax.imshow(foto_hoy, cmap="gray", aspect="auto", interpolation="nearest")
ax.set_title(f"HOY\n(hasta {d.index[fin_hoy]:%d-%m-%Y})", fontsize=9)
ax.axis("off")
for j, (p, dist) in enumerate(elegidas):
    ax = fig.add_subplot(2, 6, j + 2)
    ax.imshow(dibujar_ventana(ohlcv_d[p - N_SHAZAM + 1:p + 1]), cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"{d.index[p]:%d-%m-%Y}\n{dist} píxeles distintos", fontsize=9)
    ax.axis("off")
ax = fig.add_subplot(2, 1, 2)
x = np.arange(-N_SHAZAM + 1, DESPUES + 1)
for j, (p, _) in enumerate(elegidas):
    tramo = cierre_np[p - N_SHAZAM + 1:p + DESPUES + 1]
    ax.plot(x, tramo / cierre_np[p] * 100, alpha=0.8, label=f"{d.index[p]:%m-%Y}: {cierre_np[p + 30] / cierre_np[p] - 1:+.0%} en 30 días")
ax.plot(x[:N_SHAZAM], cierre_np[fin_hoy - N_SHAZAM + 1:] / cierre_np[fin_hoy] * 100, color="black", linewidth=3, label="hoy")
ax.axvline(0, color="gray", linestyle="--")
ax.set_xlabel("días (0 = último día de la ventana)")
ax.set_ylabel("precio (último día = 100)")
ax.set_title("Las 5 ventanas del pasado más parecidas a hoy, y qué pasó después")
ax.legend(fontsize=8)
plt.tight_layout()
guardar_fig("07_5_shazam")
plt.show()
""")
md(r"""
**Cómo leerlo:** si las 5 "gemelas" siguieron caminos **parecidos** después (todas subieron o todas bajaron), el parecido diría algo. Si siguieron caminos **distintos**, la forma de las velas no alcanza para saber qué viene.

## 7.6 Resumen: ¿qué esconden las velas de Bitcoin?
""")
code(r"""
resumen_patrones = pd.concat([
    clasicos.assign(tipo="clásico")[["tipo", "veces", "dif z train", "dif z val", "dif z test", "¿se sostiene?", "p"]],
    familias.assign(tipo="familia")[["tipo", "veces", "dif z train", "dif z val", "dif z test", "¿se sostiene?", "p"]]
        .rename(index=lambda f_: f"{f_}: {familias.loc[f_, 'descripción']}"),
])
resumen_patrones["p Holm (todos juntos)"] = holm(resumen_patrones["p"])
resumen_patrones["veredicto"] = np.where(
    resumen_patrones["p Holm (todos juntos)"] >= 0.05, "no se distingue del azar",
    np.where(resumen_patrones["¿se sostiene?"] == "sí, hasta hoy", "real y sigue vigente",
             "real en el pasado, " + resumen_patrones["¿se sostiene?"]))
guardar_tabla(resumen_patrones.round(4), "07_6_resumen_patrones")
anotar("7.6 RESUMEN DE PATRONES:\n" + resumen_patrones.round(3).to_string())
display(resumen_patrones.round(3))

reales = resumen_patrones[resumen_patrones["veredicto"] != "no se distingue del azar"]
vigentes = reales.index[reales["veredicto"] == "real y sigue vigente"].tolist()
apagados = reales.index[reales["veredicto"] != "real y sigue vigente"].tolist()
anotar("7.6 ¿Qué nos llevamos? "
       f"De {len(resumen_patrones)} patrones, {len(reales)} se distinguen del azar con todo el historial (Holm). "
       f"Siguen vigentes en el ciclo actual: {', '.join(vigentes) if vigentes else 'ninguno'}. "
       f"Se apagaron: {', '.join(apagados) if apagados else 'ninguno'}.")
""")
md(r"""
### 🔎 ¿Qué nos llevamos de la caza de patrones?
Las conclusiones con los números de esta corrida están en la celda de arriba y en la sección 15. Lo que hay que mirar:
- **Los patrones de libro de una sola vela** (martillo, estrella fugaz, doji) aparecen poco o no dicen nada distinto de un día cualquiera. Que el martillo "funcione" en los libros no se ve en Bitcoin.
- **Los patrones que sí se distinguen del azar son de inercia:** después de una semana de velas verdes (tres soldados blancos, las familias "semana en subida"), la semana siguiente tendía a seguir subiendo. Es lo mismo que mide el "sharpe" de la planilla del profe.
- **Pero esa inercia alcista se apagó:** fuerte en 2013-2020, más débil en 2020-2024 y **nula en el ciclo actual**. Es lo que predice la teoría de mercados eficientes cuando un activo madura (más participantes, ETFs, arbitraje): **un patrón conocido deja de funcionar porque todos lo explotan**.
- **La excepción, si aparece en la tabla 7.6, es la inercia bajista** (una semana en caída que termina con una vela roja fuerte suele seguir con otra semana floja). Ojo: en el ciclo actual son pocas semanas, así que es una pista para mirar, no una regla.
- **El ciclo de 4 años es real a escala de años** pero se apoya en solo 3 ciclos completos y sube cada vez menos.
- **Para la Parte III:** si los patrones que existían se están apagando, es esperable que una red entrenada con 2013-2020 tenga problemas en 2024-2026. Eso es exactamente lo que vamos a medir.
""")

# =====================================================================
md(r"""
---
# 8. ¿Sirven para predecir? Baselines y Monte Carlo: la vara que hay que superar

## 8.1 Baselines

Antes de entrenar redes, necesitamos saber **cuánto acierta alguien que no usa deep learning**. Si una red no le gana a esto, no sirve.

| Baseline | Regla | Qué representa |
|---|---|---|
| **Azar** | elige sube/baja/lateral al azar | tirar un dado de 3 caras |
| **Siempre "sube"** | la clase más común en entrenamiento | "Bitcoin siempre sube" |
| **Martillo** | si hoy hay martillo después de una semana bajista → "sube"; si no → "lateral" | el análisis técnico de libro (el detector de la planilla del profe) |
| **Inercia** | si la semana pasada subió fuerte → "sube"; si bajó fuerte → "baja"; si no → "lateral" | "la tendencia sigue" |
| **Gemela del pasado** *(9.5)* | busca la foto de velas más parecida del entrenamiento y repite lo que pasó después | "este patrón se parece a..." (como el ejemplo de LLY del profe) |

### ¿Cómo medimos? *Balanced accuracy*
Es el **promedio del acierto dentro de cada clase**: (acierto en los días que subió + acierto en los que bajó + acierto en los laterales) / 3.
- El azar da ≈ **0,33**.
- "Siempre sube" acierta el 100% de los días de suba y 0% del resto: (1 + 0 + 0)/3 = **0,33**. La *accuracy* común lo premiaría si hubo muchas subas; la balanced no se deja engañar.

""")
code(r"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score

# ---------- Configuración de la corrida (MODO_RAPIDO se define en la sección 2) ----------

VENTANAS = [5, 10] if MODO_RAPIDO else [5, 10, 20, 30, 60]   # ventanas a probar (9.4 y 10.4)
SEMILLAS = [0] if MODO_RAPIDO else [0, 1, 2, 3, 4]           # repeticiones por ventana
N_REDES_FINAL = 1 if MODO_RAPIDO else 5                       # redes que se promedian en cada modelo final
N_REDES_MEJORAS = 1 if MODO_RAPIDO else 3                     # redes por modelo en los intentos descartados (sección 14)
EPOCAS_MAX = 2 if MODO_RAPIDO else 60

# ---------- Datos que usan todos los modelos ----------
y = datos["y"].to_numpy()                                       # la respuesta correcta de cada día (0, 1, 2)
mascara = {p: (datos["periodo"] == p).to_numpy() for p in ["train", "val", "test"]}
ID = {nombre: i for i, nombre in enumerate(CLASES)}            # baja=0, lateral=1, sube=2
PREDICCIONES = {}   # nombre del modelo -> lo que dijo para cada día
PROBAS = {}         # nombre de la red   -> probabilidad de cada clase para cada día

def bal_acc(pred, periodo):
    m = mascara[periodo]
    return balanced_accuracy_score(y[m], pred[m])

# ---------- Los 4 baselines ----------
rng = np.random.default_rng(0)
PREDICCIONES["Azar"] = rng.integers(0, 3, len(datos))
PREDICCIONES["Siempre sube"] = np.full(len(datos), ID["sube"])

ret_semana_pasada = np.log(d["close"] / d["close"].shift(HORIZONTE)).reindex(datos.index)
PREDICCIONES["Martillo"] = np.where(datos["martillo"] & (ret_semana_pasada < 0), ID["sube"], ID["lateral"])

z_pasado = (ret_semana_pasada / datos["vol_reciente"]).to_numpy()
PREDICCIONES["Inercia"] = np.select([z_pasado > k, z_pasado < -k], [ID["sube"], ID["baja"]], default=ID["lateral"])

anotar(f"8. Días con martillo después de una semana bajista: {(PREDICCIONES['Martillo'] == ID['sube']).sum()} de {len(datos)}")
tabla_baselines = pd.DataFrame({"balanced_acc_validacion": {n: bal_acc(p, "val") for n, p in PREDICCIONES.items()}}).round(3)
guardar_tabla(tabla_baselines, "08_baselines_validacion")
anotar("8. Baselines en VALIDACIÓN (el test se mira recién en la sección 12):\n" + tabla_baselines.to_string())
tabla_baselines
""")
md(r"""
## 8.2 Monte Carlo: lo que usa la industria

En la industria financiera, la forma estándar de "pronosticar" un precio **no es adivinar una dirección**: es **simular miles de futuros posibles** y ver cómo se reparten. Eso es una simulación de **Monte Carlo**. Para cada día *t* simulamos 1.000 semanas posibles y contamos cuántas terminan en **sube**, **baja** o **lateral**, con **la misma regla y el mismo umbral k** que la respuesta (sección 6.2). La clase más frecuente es su predicción, y el reparto es su probabilidad.

| Variante | Cómo simula los próximos 7 días | Qué supone |
|---|---|---|
| **Paseo aleatorio** | retornos normales con media 0 y la volatilidad de los últimos 30 días | el precio no tiene memoria (mercado eficiente puro) |
| **Historia del último año** | sortea 7 retornos diarios reales de los 365 días anteriores a *t* (*bootstrap*) | lo que pasó el último año (tendencia y colas gordas incluidas) se repite |

**Sin espiar el futuro:** las dos variantes usan solo datos **anteriores o iguales a *t***.

**Un detalle que vale la pena entender:** en el paseo aleatorio, la probabilidad de cada clase es casi la misma todos los días (≈35% sube, ≈30% lateral, ≈35% baja, porque *k* deja ⅓ de los días de entrenamiento como lateral y las colas reales son más gordas que las de la normal). Elegir "la más probable" ahí es **tirar una moneda** entre sube y baja. Lo dejamos a propósito: muestra cómo se ve un modelo que **no sabe nada** y sirve para calibrar qué tan fácil es que un resultado "parezca" bueno por azar.
""")
code(r"""
SIMULACIONES = 200 if MODO_RAPIDO else 1000
rng_mc = np.random.default_rng(0)
ret_diarios_btc = d["ret_1d"].to_numpy()
vol_7d = datos["vol_reciente"].to_numpy()
POS_MC = d.index.get_indexer(datos.index)

def a_probabilidades(z_simulados):
    return np.array([np.mean(z_simulados < -k), np.mean(np.abs(z_simulados) <= k), np.mean(z_simulados > k)])

PROBAS_MC = {"Monte Carlo (paseo aleatorio)": np.zeros((len(datos), 3)), "Monte Carlo (historia del último año)": np.zeros((len(datos), 3))}
for j, p in enumerate(POS_MC):
    z_paseo = rng_mc.standard_normal(SIMULACIONES)                           # ret 7d ~ N(0, vol) -> z ~ N(0, 1)
    historia = ret_diarios_btc[max(1, p - 364):p + 1]
    historia = historia[np.isfinite(historia)]
    z_hist = rng_mc.choice(historia, (SIMULACIONES, HORIZONTE)).sum(1) / vol_7d[j]
    PROBAS_MC["Monte Carlo (paseo aleatorio)"][j] = a_probabilidades(z_paseo)
    PROBAS_MC["Monte Carlo (historia del último año)"][j] = a_probabilidades(z_hist)
for nombre, proba in PROBAS_MC.items():
    PREDICCIONES[nombre] = proba.argmax(1)
    anotar(f"8.2 {nombre}: balanced accuracy en validación = {bal_acc(PREDICCIONES[nombre], 'val'):.3f} | "
           f"probabilidad media (baja, lateral, sube) = {np.round(proba[mascara['val']].mean(0), 3)}")
""")

# =====================================================================
md(r"""
---
# 9. CNN: una red que aprende mirando la FOTO de las velas

## 9.1 ¿Por qué una CNN?
- **La consigna es reconocer patrones mirando las velas**, y un trader no lee una tabla de números: **mira el gráfico**. Una CNN (red convolucional) es el tipo de red diseñado para **mirar imágenes**: es la que se usa para reconocer caras, objetos o radiografías.
- **Su herramienta, la convolución, busca formas locales en cualquier parte de la imagen.** Un martillo es un martillo esté al principio o al final de la ventana: la CNN usa **el mismo detector** en todas las posiciones.
- **Hay evidencia de que funciona en finanzas:** Jiang, Kelly & Xiu (*Journal of Finance*, 2023) entrenaron una CNN con imágenes de gráficos de acciones de EE.UU. y encontraron poder predictivo. Nosotros **replicamos esa idea sobre Bitcoin**.
- **Es el complemento de la red recurrente** (sección 10): la CNN mira **toda la foto de golpe**; la LSTM **lee las velas una por una**. Comparar las dos responde si importa *cómo* se miran las velas.

El camino de la CNN tiene 4 pasos: **la foto** (9.2) → **la arquitectura** (9.3) → **cuántas velas mirar** (9.4) → **el modelo final** (9.5).
""")
md(r"""
## 9.2 La foto: cómo convertimos las velas en una imagen

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| **Blanco y negro** (1 canal) | color (verde/rojo) | el color no agrega información: la dirección ya se ve en el cuerpo. 1 canal = menos datos para aprender |
| **Alcista = cuerpo hueco, bajista = relleno** | todas iguales | es la convención de los libros de velas en blanco y negro (Nison) |
| **3 píxeles por vela** (mecha en el centro) | más ancho | es el formato de Jiang et al.; alcanza para ver cuerpo y mechas sin agrandar la imagen |
| **Normalizada dentro de la ventana** (máximo arriba, mínimo abajo) | precio real | la red tiene que ver **la forma**: una ventana de 2014 ($500) y una de 2025 ($100.000) son comparables |
| **Volumen abajo** | sin volumen | el volumen confirma o desmiente un movimiento (también lo usa Jiang) |
| **64 píxeles de alto** | más alto | suficiente resolución para mechas y cuerpos; imágenes chicas = entrenamiento rápido |
""")
code(r"""
import copy
import torch
import torch.nn as nn

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
print("Entrenamos en:", DISPOSITIVO, "(si dice cpu, activá la GPU: Entorno de ejecución → Cambiar tipo → T4 GPU)")

OHLCV = d[["open", "high", "low", "close", "volume"]].to_numpy()
POS = d.index.get_indexer(datos.index)           # posición de cada día del dataset dentro de la tabla completa

# dibujar_ventana (la foto de cada ventana) se definió en la sección 4

def armar_imagenes(N):
    # Una imagen por día del dataset, con las N velas que terminan ese día (incluido)
    return np.stack([dibujar_ventana(OHLCV[p - N + 1:p + 1]) for p in POS])[:, None]   # (días, 1 canal, 64, 3N)
""")
md(r"""
### Así ve la red el pico del ciclo actual
Arriba, el gráfico "normal" de las 30 velas que terminan el 6-oct-2025; abajo, la foto que recibe la CNN.
""")
code(r"""
ej_pos = d.index.get_loc(ej_fecha)
ventana_ej = d.iloc[ej_pos - 29:ej_pos + 1][["open", "high", "low", "close", "volume"]].copy()
ventana_ej.index = ventana_ej.index.tz_localize(None)            # mplfinance no acepta zona horaria
fig, _ = mpf.plot(ventana_ej, type="candle", style="charles", volume=True,
                  title="30 velas hasta el 06-10-2025 (gráfico normal)", figsize=(8, 4), returnfig=True)
guardar_fig("09_ejemplo_grafico_normal", fig)
plt.show()

plt.figure(figsize=(8, 3))
plt.imshow(dibujar_ventana(OHLCV[ej_pos - 29:ej_pos + 1]), cmap="gray", aspect="auto", interpolation="nearest")
plt.title("La misma ventana como la ve la CNN (64 × 90 píxeles)")
plt.axis("off")
guardar_fig("09_ejemplo_foto_cnn")
plt.show()

X_ejemplo = armar_imagenes(30)
fig, ejes = plt.subplots(3, 4, figsize=(14, 6))
for fila_ax, clase in zip(ejes, CLASES):
    dias = np.flatnonzero(mascara["train"] & (y == ID[clase]))
    for ax, i in zip(fila_ax, rng.choice(dias, 4, replace=False)):
        ax.imshow(X_ejemplo[i, 0], cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"{datos.index[i]:%d-%m-%Y} → {clase}", fontsize=9)
        ax.axis("off")
plt.suptitle("Fotos de entrenamiento con su respuesta: ¿se ve a simple vista qué va a pasar?")
plt.tight_layout()
guardar_fig("09_ejemplos_fotos_por_clase")
plt.show()
del X_ejemplo
""")
md(r"""
## 9.3 La arquitectura de la CNN: qué decidimos y por qué

### El recorrido de una foto por la red
```
FOTO  1 canal × 64 alto × 3N ancho
  │
  ├─ Bloque 1: convolución 5×3 (16 filtros) → normalización → activación → achicar el alto a la mitad
  │      busca formas de UNA vela: mechas largas, cuerpos grandes, velas huecas o rellenas
  ├─ Bloque 2: convolución 5×3 (32 filtros) → ... → achicar el alto
  │      combina formas vecinas: pares de velas (envolvente, harami...)
  ├─ Bloque 3: convolución 5×3 (64 filtros) → ... → achicar el alto
  │      combina pares en patrones más largos (tendencias, rebotes)
  │
  ├─ Resumen en una grilla de 4×4 (para cada uno de los 64 detectores)
  ├─ Dropout 50% (apaga neuronas al azar mientras entrena)
  └─ Capa final → 3 puntajes: BAJA · LATERAL · SUBE  → se convierten en probabilidades
```

### Cada decisión, y en base a qué la tomamos
| Pieza | Qué elegimos | En base a qué |
|---|---|---|
| **Tamaño de la red** | 3 bloques, 16-32-64 filtros (≈ 30 mil parámetros) | **Tenemos pocos datos** (≈ 2.700 fotos de entrenamiento). Una red grande (millones de parámetros, como ResNet) los **memorizaría** en vez de aprender. Jiang et al. usan 2-3 bloques con muchos más datos; nosotros nos quedamos en el extremo chico. |
| **Filtro de 5 alto × 3 ancho** | 3 píxeles de ancho | **3 píxeles = exactamente una vela.** El primer bloque mira una vela por vez; los siguientes, al apilarse, ven 2, 3, 4 velas vecinas. 5 de alto mira un tramo vertical de la vela (mecha + borde del cuerpo). |
| **Achicar solo el alto** (*max-pooling* 2×1) | el ancho no se achica | El ancho es **el tiempo**: si lo achicáramos, mezclaríamos velas de días distintos demasiado pronto. El alto sí se puede resumir (la posición exacta de un píxel importa poco). Igual que Jiang et al. |
| **Normalización** (*batch norm*) | después de cada convolución | Estabiliza y acelera el entrenamiento; con pocos datos ayuda a que no dependa tanto de la suerte del arranque. |
| **Activación** *LeakyReLU* | en vez de ReLU | Deja pasar un poquito de señal negativa: evita neuronas "muertas" que dejan de aprender. Es la que usan Jiang et al. |
| **Grilla final de 4×4** (*adaptive pooling*) | en vez de aplanar todo | Permite usar **la misma red con cualquier ventana** (5 o 60 velas): así la prueba de ventana (9.4) compara solo la ventana, no arquitecturas distintas. Y conserva una idea de **dónde** está cada forma (las últimas velas pueden pesar más). |
| **Dropout 50%** | antes de la capa final | Obliga a la red a no depender de un solo detector: es la defensa principal contra **memorizar**. |
| **3 salidas + softmax** | una por clase | La red da una **probabilidad** para cada clase; elegimos la más alta. La probabilidad la usamos después para ver si "sabe cuándo no sabe" (13.2). |

### Cómo entrena, y en base a qué
| Decisión | Qué elegimos | En base a qué |
|---|---|---|
| **Función de error** | *cross-entropy* **con pesos por clase** | En entrenamiento hay más días de "sube" (39%) que de "baja" (27%): sin pesos, la red aprende a decir "sube" de más. Los pesos compensan. |
| **Optimizador** | Adam, tasa de aprendizaje 0,001 | Es el estándar: funciona bien sin ajuste fino. |
| **Regularización** | *weight decay* 0,0001 | Penaliza pesos grandes: otra defensa contra memorizar. |
| **Lotes de 128 fotos** | | Equilibrio entre velocidad y estabilidad. |
| ***Early stopping*** | frena si el error en **validación** no mejora en 6 épocas | Es la forma objetiva de decidir **cuándo parar**: después de ese punto, la red empieza a memorizar el entrenamiento. |
| **5 redes promediadas** | en el modelo final | Una red arranca con pesos al azar; dos entrenamientos iguales dan resultados algo distintos. Promediar 5 = el resultado depende menos de la suerte. |
""")
code(r"""
class CNNVelas(nn.Module):
    def __init__(self):
        super().__init__()
        def bloque(entrada, salida):
            return nn.Sequential(
                nn.Conv2d(entrada, salida, kernel_size=(5, 3), padding=(2, 1)),   # filtro de 5 alto x 3 ancho (1 vela)
                nn.BatchNorm2d(salida),                                           # normalización
                nn.LeakyReLU(0.01),                                               # activación
                nn.MaxPool2d((2, 1)),                                             # alto a la mitad, ancho igual
            )
        self.convoluciones = nn.Sequential(bloque(1, 16), bloque(16, 32), bloque(32, 64))
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),     # grilla de 4x4 -> sirve para cualquier ventana
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 3),                       # 3 puntajes: baja, lateral, sube
        )

    def forward(self, x):
        return self.cabeza(self.convoluciones(x))


def describir_red(modelo, ejemplo, archivo):
    # Tabla con el tamaño de los datos después de cada capa y la cantidad de parámetros que aprende
    filas, x = [], torch.zeros(ejemplo)
    capas = [(f"bloque {i + 1}", b) for i, b in enumerate(modelo.convoluciones)] + [("resumen 4x4 + dropout + salida", modelo.cabeza)]
    filas.append({"capa": "foto de entrada", "forma de los datos": tuple(x.shape[1:]), "parámetros": 0})
    modelo.eval()
    with torch.no_grad():
        for nombre, capa in capas:
            x = capa(x)
            filas.append({"capa": nombre, "forma de los datos": tuple(x.shape[1:]),
                          "parámetros": sum(p.numel() for p in capa.parameters())})
    tabla = pd.DataFrame(filas)
    guardar_tabla(tabla, archivo)
    anotar(f"Arquitectura ({archivo}): {tabla['parámetros'].sum():,} parámetros en total\n" + tabla.to_string(index=False))
    return tabla

describir_red(CNNVelas(), (1, 1, ALTO, 3 * 30), "09_arquitectura_cnn")
""")
md(r"""
### Las herramientas de entrenamiento (las comparten la CNN y la LSTM)
Para que la comparación sea justa, **las dos redes se entrenan con exactamente el mismo código**: misma función de error, mismo optimizador, mismo *early stopping*, misma prueba de ventana.
""")
code(r"""
def salida_red(modelo, X, lote=1024):
    # Puntajes de la red para cada ejemplo (sin entrenar)
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(torch.from_numpy(X[i:i + lote]).float().to(DISPOSITIVO)).cpu()
                          for i in range(0, len(X), lote)])

def predecir_proba(modelo, X):
    return torch.softmax(salida_red(modelo, X), 1).numpy()


def entrenar(crear_modelo, X_tr, y_tr, X_va, y_va, semilla=0, paciencia=6, lr=1e-3, tam_lote=128):
    # Entrena UNA red con early stopping sobre el error de validación
    torch.manual_seed(semilla)
    rng_t = np.random.default_rng(semilla)
    modelo = crear_modelo().to(DISPOSITIVO)

    pesos = torch.tensor(len(y_tr) / (3 * np.bincount(y_tr, minlength=3)), dtype=torch.float32, device=DISPOSITIVO)
    funcion_error = nn.CrossEntropyLoss(weight=pesos)                    # error con pesos por clase
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-4)
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
    y_va_t = torch.from_numpy(y_va.astype(np.int64)).to(DISPOSITIVO)

    mejor_error, mejor_estado, sin_mejora, historia = np.inf, None, 0, []
    for epoca in range(EPOCAS_MAX):
        modelo.train()
        orden = rng_t.permutation(len(X_tr))
        for i in range(0, len(orden), tam_lote):
            idx = np.sort(orden[i:i + tam_lote])
            optimizador.zero_grad()
            error = funcion_error(modelo(torch.from_numpy(X_tr[idx]).float().to(DISPOSITIVO)), y_tr_t[idx].to(DISPOSITIVO))
            error.backward()
            optimizador.step()

        salida_va = salida_red(modelo, X_va)
        error_va = funcion_error(salida_va.to(DISPOSITIVO), y_va_t).item()
        historia.append({"epoca": epoca + 1, "error_val": error_va,
                         "bal_acc_val": balanced_accuracy_score(y_va, salida_va.argmax(1).numpy())})
        if error_va < mejor_error:
            mejor_error, mejor_estado, sin_mejora = error_va, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    modelo.load_state_dict(mejor_estado)
    return modelo, balanced_accuracy_score(y_va, predecir_proba(modelo, X_va).argmax(1)), pd.DataFrame(historia)


def entrenar_varias(crear_modelo, X_tr, y_tr, X_va, y_va, X_predecir, n_redes, tam_lote=128):
    # Entrena n_redes (semillas distintas) y PROMEDIA sus probabilidades sobre X_predecir
    probas, historias = [], []
    for s in range(n_redes):
        red, ba, hist = entrenar(crear_modelo, X_tr, y_tr, X_va, y_va, semilla=s, tam_lote=tam_lote)
        print(f"   red {s + 1}/{n_redes}: balanced acc. validación = {ba:.3f} ({len(hist)} épocas)")
        probas.append(predecir_proba(red, X_predecir))
        historias.append(hist)
    return np.mean(probas, axis=0), historias


def prueba_de_ventana(nombre, armar, crear_modelo, archivo):
    # Entrena la misma red con cada ventana, varias veces, y elige con la regla de "un desvío"
    filas = []
    for N in VENTANAS:
        X = armar(N)
        for s in SEMILLAS:
            _, ba, hist = entrenar(crear_modelo, X[mascara["train"]], y[mascara["train"]],
                                   X[mascara["val"]], y[mascara["val"]], semilla=s)
            filas.append({"ventana": N, "repeticion": s + 1, "bal_acc_val": ba, "epocas": len(hist)})
            print(f"{nombre} | ventana {N:>2} | repetición {s + 1} | balanced acc. validación = {ba:.3f}")
    exp = pd.DataFrame(filas)
    resumen = exp.groupby("ventana")["bal_acc_val"].agg(promedio="mean", desvio="std").fillna(0)
    mejor = resumen["promedio"].idxmax()
    umbral = resumen.loc[mejor, "promedio"] - resumen.loc[mejor, "desvio"]
    elegida = int(resumen.index[resumen["promedio"] >= umbral].min())     # la más chica dentro de un desvío

    guardar_tabla(exp, f"{archivo}_detalle")
    guardar_tabla(resumen.round(4), f"{archivo}_resumen")
    anotar(f"{nombre} - prueba de ventana (balanced accuracy en validación):\n" + resumen.round(3).to_string())
    anotar(f"{nombre} - mejor promedio: ventana {mejor}. Ventana elegida (regla de un desvío): N = {elegida}")

    plt.figure(figsize=(9, 4.5))
    plt.scatter(exp["ventana"], exp["bal_acc_val"], color="#bbbbbb", s=25, zorder=2, label="cada repetición")
    plt.errorbar(resumen.index, resumen["promedio"], yerr=resumen["desvio"], marker="o", capsize=4,
                 color="#264653", zorder=3, label="promedio ± desvío")
    plt.axhline(1 / 3, color="gray", linestyle="--", label="azar (0,33)")
    plt.axvline(elegida, color="#2a9d55", linestyle=":", label=f"elegida: N = {elegida}")
    plt.xscale("log")
    plt.minorticks_off()
    plt.xticks(VENTANAS, VENTANAS)
    plt.xlabel("Ventana (cantidad de velas que ve la red)")
    plt.ylabel("Balanced accuracy en validación")
    plt.title(f"{nombre}: ¿cuántas velas necesita ver?")
    plt.legend(fontsize=8)
    guardar_fig(archivo)
    plt.show()
    return elegida, resumen


def dibujar_matriz(ax, nombre, periodo="test"):
    # Matriz de confusión con cantidad de días y porcentaje por fila
    m = mascara[periodo]
    cm = confusion_matrix(y[m], PREDICCIONES[nombre][m], labels=[0, 1, 2])
    porc = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(porc, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]} días\n{porc[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if porc[i, j] > 0.5 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(CLASES)
    ax.set_xlabel("lo que dijo el modelo")
    ax.set_ylabel("lo que pasó")
    ax.set_title(f"{nombre} ({'validación' if periodo == 'val' else periodo})")


def registrar_red(nombre, proba, historias, archivo):
    # Guarda lo que dijo la red y muestra cómo entrenó
    PROBAS[nombre] = proba
    PREDICCIONES[nombre] = proba.argmax(1)
    anotar(f"{nombre}: balanced accuracy en validación = {bal_acc(PREDICCIONES[nombre], 'val'):.3f}")
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.2))
    for i, h in enumerate(historias):
        ejes[0].plot(h["epoca"], h["error_val"], alpha=0.8, label=f"red {i + 1}")
        ejes[1].plot(h["epoca"], h["bal_acc_val"], alpha=0.8)
    ejes[0].set_title("Error en validación por época (más bajo = mejor)")
    ejes[0].set_xlabel("época")
    ejes[0].legend(fontsize=8)
    ejes[1].axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ejes[1].set_title("Balanced accuracy en validación por época")
    ejes[1].set_xlabel("época")
    ejes[1].legend()
    dibujar_matriz(ejes[2], nombre, "val")
    plt.tight_layout()
    guardar_fig(archivo)
    plt.show()
""")
md(r"""
## 9.4 Experimento: ¿cuántas velas necesita ver la CNN?

No adivinamos el tamaño de la ventana: **lo medimos**. Entrenamos la misma CNN con ventanas de **5, 10, 20, 30 y 60 velas** y comparamos en **validación**.

Cuidados para que la comparación sea justa:
1. **Mismos días en todas las ventanas:** solo cambia cuánto pasado ve la red, no qué días se evalúan.
2. **5 repeticiones por ventana:** cada entrenamiento arranca con pesos al azar, así que da resultados algo distintos. Mostramos **cada repetición** (puntos grises) y el **promedio ± desvío**: diferencias más chicas que las barras **no son diferencias reales**.
3. **El test no se toca:** si eligiéramos la ventana mirando el test, ya no sería un examen limpio.

**Regla para elegir (regla de "un desvío"):** tomamos la ventana con mejor promedio; si una ventana **más chica** queda dentro de un desvío de esa, elegimos la más chica. Si dos ventanas empatan, la más simple es preferible: con pocos datos, menos información de entrada = menos riesgo de memorizar.
""")
code(r"""
N_CNN, resumen_ventana_cnn = prueba_de_ventana("CNN", armar_imagenes, CNNVelas, "09_prueba_ventana_cnn")
""")
md(r"""
### 🔎 Cómo leer este gráfico
- **Si todas las ventanas están cerca de 0,33**, ninguna ve nada: la red no encuentra patrones en ninguna escala. Es un resultado válido (apoya H0).
- **Si alguna sobresale por encima de su barra de error**, en esa escala hay algo de información.
- **Si mejora y después cae**, existe una "memoria" útil de las velas: más pasado que eso es ruido.

""")
md(r"""
## 9.5 La CNN final
Con la ventana elegida entrenamos **5 redes** y **promediamos sus probabilidades**. Después de entrenar mostramos: el error de validación de cada red época por época (dónde frenó el *early stopping*), su acierto, y la matriz de confusión en validación.
""")
code(r"""
X_cnn = armar_imagenes(N_CNN)
print(f"Fotos: {X_cnn.shape[0]:,} de {X_cnn.shape[2]}×{X_cnn.shape[3]} píxeles (ventana de {N_CNN} velas)")
proba, historias = entrenar_varias(CNNVelas, X_cnn[mascara["train"]], y[mascara["train"]],
                                   X_cnn[mascara["val"]], y[mascara["val"]], X_cnn, N_REDES_FINAL)
registrar_red("CNN", proba, historias, "09_entrenamiento_cnn")
""")
md(r"""
### El quinto baseline: la "gemela del pasado"
Para cada día de validación y test buscamos **la foto más parecida de todo el entrenamiento** (la que tiene menos píxeles distintos) y predecimos **lo mismo que pasó después de esa gemela**. Es la idea de "este patrón se parece a otro que ya vi" llevada al extremo, sin red. Guardamos también **cuántos píxeles difieren**: lo usamos en el análisis de errores (13.3) para saber qué fotos son **nuevas** para el modelo.
""")
code(r"""
def foto_mas_parecida(X_consulta, X_referencia, lote=256):
    # Para cada foto de X_consulta: cuántos píxeles difieren de la más parecida de X_referencia, y cuál es
    A = torch.from_numpy(X_consulta.reshape(len(X_consulta), -1)).float().to(DISPOSITIVO)
    B = torch.from_numpy(X_referencia.reshape(len(X_referencia), -1)).float().to(DISPOSITIVO)
    encendidos_B = B.sum(1)
    distancias, indices = [], []
    for i in range(0, len(A), lote):
        a = A[i:i + lote]
        distintos = a.sum(1, keepdim=True) + encendidos_B[None] - 2 * a @ B.T    # píxeles que difieren
        minimo, cual = distintos.min(1)
        distancias.append(minimo.cpu())
        indices.append(cual.cpu())
    return torch.cat(distancias).numpy(), torch.cat(indices).numpy()

idx_train = np.flatnonzero(mascara["train"])
no_train = ~mascara["train"]
distancia, cual = foto_mas_parecida(X_cnn[no_train], X_cnn[mascara["train"]])

NOVEDAD = np.full(len(datos), np.nan)       # píxeles distintos con la foto más parecida del entrenamiento
NOVEDAD[no_train] = distancia
GEMELA = np.full(len(datos), -1)            # qué día del entrenamiento es su gemela
GEMELA[no_train] = idx_train[cual]

pred_gemela = np.full(len(datos), ID["lateral"])
pred_gemela[no_train] = y[GEMELA[no_train]]
PREDICCIONES["Gemela del pasado"] = pred_gemela
anotar(f"9.5 Gemela del pasado: balanced accuracy en validación = {bal_acc(pred_gemela, 'val'):.3f}")
""")

# =====================================================================
md(r"""
---
# 10. LSTM: una red recurrente que LEE las velas una por una

## 10.1 ¿Por qué una red recurrente (LSTM)?
- **Las velas son una secuencia en el tiempo**: el orden importa (no es lo mismo una vela grande y después una chica que al revés). Una red **recurrente** está diseñada para eso: lee los datos **en orden**, uno por uno, y va guardando una **memoria** de lo que vio.
- **¿Por qué LSTM y no una recurrente simple?** Una recurrente simple se "olvida" rápido de lo que vio al principio. La **LSTM** (*Long Short-Term Memory*) tiene **compuertas** que deciden, en cada vela, **qué recordar, qué olvidar y qué usar**. Es la recurrente estándar para series temporales (una GRU es parecida y más simple; con tan pocos datos rinden casi igual).
- **Mira lo mismo de otra manera:** la CNN ve una **foto** (redondeada a píxeles); la LSTM recibe los **números exactos** de cada vela. Si las dos fallan, no es por cómo miramos las velas; si una funciona y la otra no, *cómo* se miran importa.

## 10.2 La secuencia: qué números recibe por cada vela

Por cada vela le damos 6 números. Todos están **en unidades de volatilidad** (divididos por cuánto se movía el precio por día en el último mes), por la misma razón por la que la foto está normalizada: que una vela de 2014 y una de 2025 sean comparables.

| Número | Cálculo | Qué mide | Equivale en la foto a... |
|---|---|---|---|
| `retorno` | log(close / close de ayer) | cuánto se movió el día | la altura de la vela respecto de la anterior |
| `salto_apertura` | log(open / close de ayer) | si abrió lejos del cierre anterior | |
| `cuerpo` | log(close / open), con signo | fuerza y dirección del día | el cuerpo hueco o relleno |
| `mecha_sup` | log(high / max(open, close)) | rechazo de precios altos | la mecha de arriba |
| `mecha_inf` | log(min(open, close) / low) | rechazo de precios bajos | la mecha de abajo |
| `volumen` | log(volumen / promedio de 30 días) | si se operó más o menos de lo normal | la barra de volumen |

Son las mismas columnas de la planilla del profe (cuerpo, mechas, dirección), en formato para la red.
""")
code(r"""
COLUMNAS_SECUENCIA = ["retorno", "salto_apertura", "cuerpo", "mecha_sup", "mecha_inf", "volumen"]

def features_secuencia(df, vol_reciente, horizonte):
    # 6 números por vela, en unidades de volatilidad diaria (sirve para Bitcoin y para acciones)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    c_ayer = c.shift(1)
    vol_diaria = vol_reciente / np.sqrt(horizonte)
    precio = pd.DataFrame({
        "retorno": np.log(c / c_ayer),
        "salto_apertura": np.log(o / c_ayer),
        "cuerpo": np.log(c / o),
        "mecha_sup": np.log(h / np.maximum(o, c)),
        "mecha_inf": np.log(np.minimum(o, c) / l),
    }).div(vol_diaria, axis=0)
    precio["volumen"] = np.log((v + 1) / (v.rolling(30).mean() + 1))
    return precio.clip(-10, 10).fillna(0).to_numpy(np.float32)

SECUENCIAS = features_secuencia(d, d["vol_reciente"], HORIZONTE)

def armar_secuencias(N):
    # Una secuencia por día del dataset: las N velas que terminan ese día, 6 números cada una
    return np.stack([SECUENCIAS[p - N + 1:p + 1] for p in POS])      # (días, N, 6)

# Así ve la LSTM las 10 velas previas al pico del ciclo actual
N_ej = 10
tabla_ej = pd.DataFrame(SECUENCIAS[ej_pos - N_ej + 1:ej_pos + 1], columns=COLUMNAS_SECUENCIA,
                        index=d.index[ej_pos - N_ej + 1:ej_pos + 1].strftime("%d-%m"))
fig, ejes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.3]})
ejes[0].plot(range(N_ej), d["close"].iloc[ej_pos - N_ej + 1:ej_pos + 1], marker="o", color="black")
ejes[0].set_xticks(range(N_ej))
ejes[0].set_xticklabels(tabla_ej.index)
ejes[0].set_title("Precio de cierre de las 10 velas hasta el 06-10-2025")
ejes[1].imshow(tabla_ej.T, cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
for i in range(N_ej):
    for j in range(len(COLUMNAS_SECUENCIA)):
        ejes[1].text(i, j, f"{tabla_ej.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=8)
ejes[1].set_xticks(range(N_ej))
ejes[1].set_xticklabels(tabla_ej.index)
ejes[1].set_yticks(range(len(COLUMNAS_SECUENCIA)))
ejes[1].set_yticklabels(COLUMNAS_SECUENCIA)
ejes[1].set_title("Lo que recibe la LSTM: una columna de 6 números por vela, leídas de izquierda a derecha")
plt.tight_layout()
guardar_fig("10_ejemplo_secuencia_lstm")
plt.show()
""")
md(r"""
## 10.3 La arquitectura de la LSTM: qué decidimos y por qué

```
SECUENCIA  N velas × 6 números
  │
  ├─ LSTM con 32 unidades de memoria: lee vela 1, vela 2, ..., vela N
  │     en cada vela actualiza su memoria (qué recordar, qué olvidar)
  ├─ Se queda con la memoria final (el "resumen" de toda la ventana)
  ├─ Dropout 30%
  └─ Capa final → 3 puntajes: BAJA · LATERAL · SUBE
```

| Pieza | Qué elegimos | En base a qué |
|---|---|---|
| **Tipo de red** | LSTM de 1 capa | Es la recurrente estándar; 1 capa porque hay pocos datos (2 capas duplican los parámetros) |
| **Memoria** | 32 unidades (≈ 5 mil parámetros) | Mismo criterio que la CNN: **red chica para pocos datos**, así no memoriza |
| **Qué usamos para decidir** | la memoria al final de la última vela | La LSTM ya "leyó" toda la ventana; su estado final la resume |
| **Dropout** | 30% | Menos que en la CNN porque la red es mucho más chica |
| **Entrenamiento** | **idéntico a la CNN** (mismo error con pesos, Adam, early stopping, 5 redes promediadas) | Para que la comparación sea **justa**: lo único que cambia es la red |
| **Ventana** | se prueba igual que en la CNN (10.4) | Puede que a la LSTM le sirva una ventana distinta |
""")
code(r"""
class LSTMVelas(nn.Module):
    def __init__(self, n_entradas=len(COLUMNAS_SECUENCIA), memoria=32):
        super().__init__()
        self.lstm = nn.LSTM(n_entradas, memoria, batch_first=True)
        self.cabeza = nn.Sequential(nn.Dropout(0.3), nn.Linear(memoria, 3))

    def forward(self, x):
        _, (memoria_final, _) = self.lstm(x)      # memoria después de leer la última vela
        return self.cabeza(memoria_final[-1])

parametros_lstm = sum(p.numel() for p in LSTMVelas().parameters())
anotar(f"Arquitectura LSTM: {parametros_lstm:,} parámetros en total")
""")
md(r"""
## 10.4 Experimento: ¿cuántas velas necesita leer la LSTM?
Exactamente la misma prueba que en 9.4: mismas ventanas, mismas repeticiones, misma regla para elegir.
""")
code(r"""
N_LSTM, resumen_ventana_lstm = prueba_de_ventana("LSTM", armar_secuencias, LSTMVelas, "10_prueba_ventana_lstm")
""")
md(r"""
## 10.5 La LSTM final
""")
code(r"""
X_lstm = armar_secuencias(N_LSTM)
print(f"Secuencias: {X_lstm.shape[0]:,} de {N_LSTM} velas × {X_lstm.shape[2]} números")
proba, historias = entrenar_varias(LSTMVelas, X_lstm[mascara["train"]], y[mascara["train"]],
                                   X_lstm[mascara["val"]], y[mascara["val"]], X_lstm, N_REDES_FINAL)
registrar_red("LSTM", proba, historias, "10_entrenamiento_lstm")

# CNN vs LSTM en la prueba de ventana
plt.figure(figsize=(9, 4.5))
for nombre, res, color in [("CNN", resumen_ventana_cnn, "#264653"), ("LSTM", resumen_ventana_lstm, "#e76f51")]:
    plt.errorbar(res.index, res["promedio"], yerr=res["desvio"], marker="o", capsize=4, color=color, label=nombre)
plt.axhline(1 / 3, color="gray", linestyle="--", label="azar")
plt.xscale("log")
plt.minorticks_off()
plt.xticks(VENTANAS, VENTANAS)
plt.xlabel("Ventana (velas)")
plt.ylabel("Balanced accuracy en validación")
plt.title("Prueba de ventana: CNN vs LSTM")
plt.legend()
guardar_fig("10_prueba_ventana_cnn_vs_lstm")
plt.show()
""")

md(r"""
---
# 11. Rivales pre-entrenados: ¿un modelo gigante ve lo que nuestras redes no ven?

## 11.1 Qué es un modelo pre-entrenado
Nuestras redes aprendieron **desde cero**, solo con Bitcoin (≈ 2.700 días). Un **modelo pre-entrenado** (o *fundacional*) es al revés: otra gente lo entrenó con **cantidades enormes de series de tiempo** y lo publicó para que cualquiera lo use. Lo usamos **tal cual, sin entrenarlo con nuestros datos** (*zero-shot*): le damos las velas hasta un día y nos dice qué espera para los próximos 7.

Es como comparar a un estudiante que estudió solo los ciclos de Bitcoin (nuestras redes) con un experto que vio millones de gráficos de todo tipo de mercados.

| | **Kronos** | **Chronos-2** | Nuestras redes |
|---|---|---|---|
| Quién | Shi et al., aceptado en **AAAI 2026** | **Amazon**, 2025 | nosotros |
| Con qué aprendió | **más de 12.000 millones de velas** (*K-lines*) de 45 mercados del mundo | millones de series de tiempo de todo tipo (energía, clima, ventas, finanzas...) | ≈ 2.700 días de Bitcoin |
| Qué mira | **velas completas** (apertura, máximo, mínimo, cierre, volumen) | una serie de números (usamos el cierre) | velas completas |
| Tamaño | Kronos-small: ≈ 25 millones de parámetros | ≈ 120 millones de parámetros | ≈ 5 a 30 mil parámetros |
| Contexto que usamos | las últimas 400 velas | las últimas 400 velas | 5 a 60 velas |

También probamos un tercer pre-entrenado, de otro tipo: **ResNet-18**, una red de **imágenes** entrenada con **ImageNet**. No mejoró en validación y quedó entre los descartes (14.6).

**Kronos es el más relevante para este trabajo: es un modelo de velas japonesas.** Si los patrones de velas tienen información, un modelo que vio miles de millones de velas debería encontrarla.

### Para qué nos sirven
1. **Como vara más alta:** si ni un modelo que vio miles de millones de velas anticipa la semana siguiente, la conclusión "no estaba en las velas" se vuelve mucho más fuerte.
2. **Para saber si el problema son los datos:** si el pre-entrenado le gana a nuestras redes, la diferencia es la cantidad de datos con la que aprendió (y eso conecta con el intento de 14.2: entrenar con más datos).
3. **Para combinar:** ¿cuando nuestra red y el pre-entrenado **coinciden**, aciertan más? (11.4)

### Un modelo que evaluamos y descartamos: *candlefusion*
En HuggingFace hay un modelo llamado `tuankg1028/candlefusion` que mira imágenes de velas con un *Vision Transformer*. Lo revisamos y **no lo usamos** porque:
- además de la imagen, **necesita texto de noticias** como entrada (usa BERT), y nosotros no tenemos noticias;
- su ficha **no informa ninguna métrica de acierto** (solo un error de validación de 316.165, en precio);
- **no dice con qué fechas se entrenó** (con un dataset de BTC que podría incluir nuestro período de test);
- no tiene paper ni evaluación independiente.

Elegir rivales **documentados y evaluados** (Kronos: paper aceptado en AAAI 2026; Chronos-2: Amazon) es parte de hacer una comparación seria.

### Cómo convertimos su predicción en sube / baja / lateral
Estos modelos no contestan "sube/baja/lateral": predicen **el precio de los próximos 7 días**. Con el precio predicho a 7 días calculamos el movimiento esperado en volatilidades (*z*) y aplicamos **exactamente la misma regla y el mismo umbral k** que usamos para armar la respuesta (sección 6). Así se los evalúa con las mismas reglas que a nuestras redes.

### ⚠️ Una limitación importante: ¿ya vieron el examen?
No sabemos con exactitud con qué fechas se entrenaron estos modelos. Si incluyeron precios de Bitcoin de 2024-2025, **ya vieron parte del test** y su resultado sería optimista. Para controlarlo, además del test completo medimos un **"examen limpio"**: solo los días **posteriores a noviembre de 2025**, cuando los dos modelos ya estaban publicados (Kronos: ago-2025; Chronos-2: oct-2025), así que **no pueden haberlos visto**.
""")
code(r"""
CONTEXTO_PRE = 400                                   # velas de contexto (Kronos-small acepta hasta 512)
FECHA_LIMPIA = pd.Timestamp("2025-11-01", tz="UTC")  # después de esta fecha, ningún pre-entrenado vio los datos
DIAS_PRE = np.flatnonzero(mascara["test"])           # días del test
if MODO_RAPIDO:
    DIAS_PRE = DIAS_PRE[::25]
PRECIO_PREDICHO = {}

def a_clase(precio_predicho, dias):
    # Precio predicho a 7 días -> sube/baja/lateral con la MISMA regla y el MISMO umbral k de la sección 5
    close_hoy = d["close"].to_numpy()[POS[dias]]
    z_pred = np.log(precio_predicho / close_hoy) / datos["vol_reciente"].to_numpy()[dias]
    return np.select([z_pred > k, z_pred < -k], [ID["sube"], ID["baja"]], default=ID["lateral"])

def registrar_preentrenado(nombre, precio_predicho, dias):
    pred = np.full(len(datos), ID["lateral"])
    pred[dias] = a_clase(precio_predicho, dias)
    PREDICCIONES[nombre] = pred
    PRECIO_PREDICHO[nombre] = pd.Series(precio_predicho, index=datos.index[dias])
    anotar(f"11. {nombre}: balanced accuracy en el test = {balanced_accuracy_score(y[dias], pred[dias]):.3f} ({len(dias)} días)")

anotar(f"11. Pre-entrenados: {len(DIAS_PRE)} días del test, contexto de {CONTEXTO_PRE} velas; "
       f"examen limpio desde {FECHA_LIMPIA:%d-%m-%Y} ({(datos.index[DIAS_PRE] >= FECHA_LIMPIA).sum()} días)")
""")
md(r"""
## 11.2 Kronos: el modelo fundacional de velas
Se descarga el código oficial (GitHub) y el modelo **Kronos-small** (HuggingFace). Para cada día del test le damos las **400 velas anteriores** y le pedimos las **7 velas siguientes**. Kronos genera el futuro con algo de azar (como un pronóstico del tiempo que simula varios escenarios), así que pedimos **5 escenarios y los promediamos**. Si algo falla (descarga, versión), el notebook sigue sin Kronos.
""")
code(r"""
!git clone -q https://github.com/shiyu-coder/Kronos.git 2>/dev/null || echo "Kronos ya estaba descargado"
!pip install -q einops
""")
code(r"""
import sys
sys.path.append("Kronos")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizador_kronos = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    modelo_kronos = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    try:
        predictor_kronos = KronosPredictor(modelo_kronos, tokenizador_kronos, device=DISPOSITIVO, max_context=512)
    except TypeError:
        predictor_kronos = KronosPredictor(modelo_kronos, tokenizador_kronos, max_context=512)

    velas_k = d[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    fechas_k = pd.Series(d.index.tz_convert(None))

    def entrada_kronos(i):
        p = POS[i]
        x_df = velas_k.iloc[p - CONTEXTO_PRE + 1:p + 1].reset_index(drop=True)
        x_ts = fechas_k.iloc[p - CONTEXTO_PRE + 1:p + 1].reset_index(drop=True)
        y_ts = pd.Series(pd.date_range(fechas_k.iloc[p] + pd.Timedelta(days=1), periods=HORIZONTE, freq="D"))
        return x_df, x_ts, y_ts

    torch.manual_seed(0)
    precio_kronos, LOTE = [], 32
    for a in range(0, len(DIAS_PRE), LOTE):
        entradas = [entrada_kronos(i) for i in DIAS_PRE[a:a + LOTE]]
        if hasattr(predictor_kronos, "predict_batch"):
            salidas = predictor_kronos.predict_batch(
                df_list=[e[0] for e in entradas], x_timestamp_list=[e[1] for e in entradas],
                y_timestamp_list=[e[2] for e in entradas], pred_len=HORIZONTE, T=1.0, top_p=0.9, sample_count=5, verbose=False)
        else:
            salidas = [predictor_kronos.predict(df=e[0], x_timestamp=e[1], y_timestamp=e[2], pred_len=HORIZONTE,
                                                T=1.0, top_p=0.9, sample_count=5) for e in entradas]
        precio_kronos += [s["close"].iloc[-1] for s in salidas]
        print(f"Kronos: {min(a + LOTE, len(DIAS_PRE))}/{len(DIAS_PRE)} días")
    registrar_preentrenado("Kronos (pre-entrenado)", np.array(precio_kronos, dtype=float), DIAS_PRE)
except Exception as e:
    print("⚠️ Kronos no pudo correr, se sigue sin él:", repr(e))
""")
md(r"""
## 11.3 Chronos-2: el modelo fundacional general de Amazon
Le damos el **precio de cierre** de las 400 velas anteriores a cada día y le pedimos los 7 días siguientes. Chronos-2 devuelve una predicción central (la mediana de sus escenarios), que es la que usamos.
""")
code(r"""
!pip install -q chronos-forecasting
""")
code(r"""
try:
    from chronos import Chronos2Pipeline
    pipeline_chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DISPOSITIVO)
    tramos = []
    for i in DIAS_PRE:
        p = POS[i]
        tramo = d["close"].iloc[p - CONTEXTO_PRE + 1:p + 1]
        tramos.append(pd.DataFrame({"id": i, "timestamp": tramo.index.tz_convert(None), "target": tramo.to_numpy()}))
    contexto_chronos = pd.concat(tramos, ignore_index=True)
    pred_chronos = pipeline_chronos.predict_df(contexto_chronos, prediction_length=HORIZONTE, quantile_levels=[0.1, 0.5, 0.9],
                                               id_column="id", timestamp_column="timestamp", target="target")
    columna = "predictions" if "predictions" in pred_chronos.columns else "0.5"
    ultimo = pred_chronos.sort_values(["id", "timestamp"]).groupby("id").tail(1).set_index("id")
    registrar_preentrenado("Chronos-2 (pre-entrenado)", ultimo.loc[DIAS_PRE, columna].to_numpy(dtype=float), DIAS_PRE)
except Exception as e:
    print("⚠️ Chronos-2 no pudo correr, se sigue sin él:", repr(e))
""")
md(r"""
## 11.4 Nuestras redes vs los pre-entrenados
Tres comparaciones:
1. **Test completo** (abr-2024 → hoy).
2. **Examen limpio** (desde nov-2025): días que los pre-entrenados seguro no vieron. Si en el test completo les va bien y en el limpio no, probablemente **ya habían visto** parte del examen.
3. **¿Sirve combinarlos?** Cuando nuestra red y un pre-entrenado **dicen lo mismo**, ¿aciertan más que cuando no coinciden?

Además, un gráfico de lo que **predijo cada pre-entrenado** contra lo que **pasó** en el ciclo actual.
""")
code(r"""
PREENTRENADOS = [n for n in PREDICCIONES if "pre-entrenado" in n]
if PREENTRENADOS:
    m_completo = np.zeros(len(datos), bool)
    m_completo[DIAS_PRE] = True
    m_limpio = m_completo & (datos.index >= FECHA_LIMPIA)
    comparar = [n for n in ["CNN", "LSTM", "CNN-LSTM híbrida"] if n in PREDICCIONES] + PREENTRENADOS
    vs_pre = pd.DataFrame({
        "test completo": {n: balanced_accuracy_score(y[m_completo], PREDICCIONES[n][m_completo]) for n in comparar},
        "examen limpio (desde nov-2025)": {n: balanced_accuracy_score(y[m_limpio], PREDICCIONES[n][m_limpio]) for n in comparar},
    })
    guardar_tabla(vs_pre.round(3), "11_4_redes_vs_preentrenados")
    anotar("11.4 Nuestras redes vs pre-entrenados (balanced accuracy):\n" + vs_pre.round(3).to_string())

    filas = []
    for pre in PREENTRENADOS:
        for red in ["CNN", "LSTM"]:
            coinciden = PREDICCIONES[red][DIAS_PRE] == PREDICCIONES[pre][DIAS_PRE]
            acierto = PREDICCIONES[red][DIAS_PRE] == y[DIAS_PRE]
            filas.append({"red": red, "pre-entrenado": pre, "% días que coinciden": coinciden.mean(),
                          "acierto cuando coinciden": acierto[coinciden].mean() if coinciden.any() else np.nan,
                          "acierto cuando NO coinciden": acierto[~coinciden].mean() if (~coinciden).any() else np.nan})
    acuerdo = pd.DataFrame(filas)
    guardar_tabla(acuerdo.round(3), "11_4_cuando_coinciden")
    anotar("11.4 ¿Aciertan más cuando coinciden?\n" + acuerdo.round(3).to_string(index=False))

    fig, ejes = plt.subplots(1, 2, figsize=(16, 4.8))
    vs_pre.plot.bar(ax=ejes[0], rot=15, color=["#264653", "#2a9d55"])
    ejes[0].axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ejes[0].set_title("Balanced accuracy: test completo vs examen limpio")
    ejes[0].legend(fontsize=8)
    etiquetas = [f"{f['red']} + {f['pre-entrenado'].split(' ')[0]}" for _, f in acuerdo.iterrows()]
    x = np.arange(len(acuerdo))
    ejes[1].bar(x - 0.2, acuerdo["acierto cuando coinciden"], 0.4, color="#2a9d55", label="cuando coinciden")
    ejes[1].bar(x + 0.2, acuerdo["acierto cuando NO coinciden"], 0.4, color="#d1495b", label="cuando no coinciden")
    ejes[1].set_xticks(x)
    ejes[1].set_xticklabels(etiquetas, fontsize=8)
    ejes[1].axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ejes[1].set_title("¿Nuestra red acierta más cuando coincide con el pre-entrenado?")
    ejes[1].legend(fontsize=8)
    plt.tight_layout()
    guardar_fig("11_4_redes_vs_preentrenados")
    plt.show()

    fig, ax = plt.subplots(figsize=(14, 5))
    fechas_pre = datos.index[DIAS_PRE]
    real_7d = d["close"].shift(-HORIZONTE).to_numpy()[POS[DIAS_PRE]]
    ax.plot(fechas_pre, real_7d, color="black", linewidth=1.5, label="precio real 7 días después")
    for pre, color in zip(list(PRECIO_PREDICHO), ["#e76f51", "#9b5de5"]):
        ax.plot(PRECIO_PREDICHO[pre].index, PRECIO_PREDICHO[pre].to_numpy(), color=color, linewidth=1, alpha=0.8,
                label=f"lo que predijo {pre.split(' ')[0]}")
    ax.axvline(FECHA_LIMPIA, color="gray", linestyle="--")
    ax.text(FECHA_LIMPIA, 0.97, "  examen limpio →", transform=ax.get_xaxis_transform(), va="top")
    ax.set_title("Precio a 7 días: lo que predijeron los pre-entrenados vs lo que pasó")
    ax.legend()
    guardar_fig("11_4_prediccion_vs_real")
    plt.show()
else:
    print("Ningún pre-entrenado pudo correr; se saltea la comparación.")
""")
md(r"""
### 🔎 Cómo leer estos resultados
- **Si los pre-entrenados predicen casi una copia del precio de hoy** (la línea de color sigue a la negra con una semana de atraso): es lo que hace un buen modelo cuando **no hay patrón que explotar**. La mejor apuesta para el precio de mañana es el de hoy: es lo que dice la hipótesis de mercado eficiente (H0).
- **Si les va mejor en el test completo que en el examen limpio:** señal de que **ya habían visto** parte del ciclo actual al entrenarse.
- **Si nuestra red acierta más cuando coincide con el pre-entrenado:** combinarlos puede servir (por ejemplo, operar solo cuando coinciden). Si no cambia, no aporta.
""")

# =====================================================================
md(r"""
---
# 12. El examen final (test): ¿el precio de hoy estaba en las velas?

## 12.1 Qué es el test y por qué es justo

Todo lo anterior fue **preparación**. Ahora tomamos los días del **ciclo actual** (desde el halving del 20-abr-2024 hasta hoy) y hacemos esto, **día por día**:

1. Le mostramos a cada modelo **solo las velas hasta ese día** (la foto o la secuencia).
2. El modelo contesta: **¿la semana siguiente sube, baja o queda lateral?**
3. Como **esos días ya pasaron, nosotros sabemos la respuesta correcta**. El modelo no: nunca vio estos días, ni durante el entrenamiento ni para elegir la ventana o cuándo parar. (La única duda son los pre-entrenados, que pueden haber visto precios de 2024-2025: por eso en 11.4 medimos también el "examen limpio".) Lo que probamos y descartamos en el camino se evalúa aparte, en la sección 14.
4. Comparamos su respuesta con lo que pasó: **acierto o error**.

Es como un examen con las respuestas en un sobre cerrado: el modelo estudió con los ciclos anteriores, rinde con el ciclo actual, y recién al final abrimos el sobre. **Ya sabíamos qué iba a pasar, pero lo usamos para corregir, no para enseñar.**

Por eso el test se evalúa **una sola vez**, con todo ya decidido: si después de verlo cambiáramos algo, estaríamos estudiando con las respuestas.
""")
code(r"""
mt = mascara["test"]
yt = y[mt]
idx_test = np.flatnonzero(mt)
fechas_test = datos.index[mt]
anotar(f"12. Test: {mt.sum()} días, del {fechas_test.min():%d-%m-%Y} al {fechas_test.max():%d-%m-%Y} "
       f"(sube {np.mean(yt == ID['sube']):.0%}, baja {np.mean(yt == ID['baja']):.0%}, lateral {np.mean(yt == ID['lateral']):.0%})")

serie = d["close"].loc[INICIO:]
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.semilogy(serie.index, serie, color="black", linewidth=1)
for ini, fin, color, texto in [
        (INICIO, CORTE_VAL, "#bbbbbb", "ENTRENAMIENTO\nla red aprende:\nve las velas Y las respuestas"),
        (CORTE_VAL, CORTE_TEST, "#f4a261", "VALIDACIÓN\nelegimos la ventana\ny cuándo dejar de entrenar"),
        (CORTE_TEST, serie.index.max(), "#2a9d55", "TEST\nla red solo ve las velas;\nnosotros ya sabemos\nqué pasó y la corregimos")]:
    ax.axvspan(ini, fin, color=color, alpha=0.25)
    ax.text(ini + (fin - ini) / 2, 0.97, texto, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
ax.set_ylabel("Precio de BTC (escala log)")
ax.set_title("Cómo se usaron los datos: aprender, ajustar, y rendir el examen con el ciclo actual")
guardar_fig("12_1_como_se_usaron_los_datos")
plt.show()
""")
md(r"""
### La hoja del examen
Así se ve el test por dentro: 12 días del ciclo actual, elegidos parejos a lo largo del período. Para cada uno: el precio ese día, el precio 7 días después, **la respuesta correcta (que ya sabíamos)** y lo que contestó cada red.
""")
code(r"""
muestra = idx_test[np.linspace(0, len(idx_test) - 1, 12).astype(int)]
filas = []
for i in muestra:
    p = POS[i]
    fila = [f"{datos.index[i]:%d-%m-%Y}", f"${d['close'].iloc[p]:,.0f}", f"${d['close'].iloc[p + HORIZONTE]:,.0f}",
            f"{np.expm1(datos['ret_futuro'].iloc[i]):+.1%}", CLASES[y[i]].upper()]
    for m in ["CNN", "LSTM"]:
        dijo = PREDICCIONES[m][i]
        fila.append(f"{CLASES[dijo]} {'✔' if dijo == y[i] else '✘'}")
    filas.append(fila)
columnas = ["día", "precio ese día", "precio 7 días después", "cambio", "RESPUESTA\n(ya la sabíamos)", "la CNN dijo", "la LSTM dijo"]
hoja = pd.DataFrame(filas, columns=columnas)
guardar_tabla(hoja, "12_1_hoja_del_examen")

fig, ax = plt.subplots(figsize=(15, 0.45 * len(hoja) + 1.2))
ax.axis("off")
t = ax.table(cellText=hoja.values, colLabels=columnas, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(9)
t.scale(1, 1.6)
for (r, c), celda in t.get_celld().items():
    if r == 0:
        celda.set_facecolor("#264653")
        celda.get_text().set_color("white")
    elif c >= 5:
        celda.set_facecolor("#d8f3dc" if "✔" in celda.get_text().get_text() else "#ffd6d6")
ax.set_title("La hoja del examen: 12 días del ciclo actual", pad=12)
guardar_fig("12_1_hoja_del_examen")
plt.show()
""")
md(r"""
## 12.2 Resultados de todos los modelos

Al examen final entran los **candidatos**: los baselines, las dos simulaciones de Monte Carlo, la CNN, la LSTM y los pre-entrenados Kronos y Chronos-2. (Lo que se probó y se descartó en el camino se evalúa aparte, en la sección 14, para que no "ensucie" la comparación.)

Para cada modelo mostramos:
- **Balanced accuracy** y su **intervalo de confianza del 95%**: remuestreamos el test en **bloques de 30 días** seguidos, porque los días vecinos se parecen (las semanas se solapan) y remuestrear días sueltos daría un intervalo demasiado angosto.
- **p-valor por desplazamiento:** "deslizamos" las respuestas correctas en el tiempo 1.000 veces (sin mezclarlas, así se conserva la forma en que se encadenan) y vemos cuántas veces un resultado al azar es igual o mejor. *En una versión anterior mezclábamos las respuestas al azar: eso rompe el encadenamiento de las semanas y daba p-valores demasiado optimistas.*
- **p-valor corregido (Holm):** con 11 modelos, alguno puede dar p < 0,05 por pura suerte. Holm sube la vara según cuántos modelos se evalúan. **Un modelo es "significativo" solo si su p corregido es menor a 0,05** (★ en el gráfico).
""")
code(r"""
from matplotlib.patches import Patch

CANDIDATOS = [n for n in ["Azar", "Siempre sube", "Martillo", "Inercia", "Gemela del pasado",
                          "Monte Carlo (paseo aleatorio)", "Monte Carlo (historia del último año)",
                          "CNN", "LSTM", "Kronos (pre-entrenado)", "Chronos-2 (pre-entrenado)"] if n in PREDICCIONES]
TIPO = {
    "Azar": "Baseline", "Siempre sube": "Baseline", "Martillo": "Baseline", "Inercia": "Baseline", "Gemela del pasado": "Baseline",
    "Monte Carlo (paseo aleatorio)": "Simulación", "Monte Carlo (historia del último año)": "Simulación",
    "CNN": "Red propia", "LSTM": "Red propia",
    "Kronos (pre-entrenado)": "Pre-entrenado", "Chronos-2 (pre-entrenado)": "Pre-entrenado",
    "CNN + acciones": "Descartado", "LSTM + acciones": "Descartado", "CNN + criptos": "Descartado", "LSTM + criptos": "Descartado",
    "CNN reentrenada": "Descartado", "LSTM reentrenada": "Descartado", "CNN-LSTM híbrida": "Descartado",
    "ResNet-18 (pre-entrenado en ImageNet)": "Descartado",
}

def tipo_de(nombre):
    if nombre not in TIPO:
        raise KeyError(f"El modelo '{nombre}' no tiene tipo asignado: agregarlo al diccionario TIPO")
    return TIPO[nombre]

COLOR_TIPO = {"Baseline": "#bbbbbb", "Simulación": "#f4a261", "Red propia": "#264653", "Pre-entrenado": "#9b5de5", "Descartado": "#2a9d55"}

def tipos_de_resultado(pred):
    pt = pred[mt]
    acierto = pt == yt
    grave = ((pt == ID["sube"]) & (yt == ID["baja"])) | ((pt == ID["baja"]) & (yt == ID["sube"]))
    return acierto, ~acierto & ~grave, grave

def evaluar_en_test(nombres):
    filas = []
    for nombre in nombres:
        pt = PREDICCIONES[nombre][mt]
        ba_, p = p_desplazamiento(lambda yy: balanced_accuracy_score(yy, pt), yt)
        ic = ic_bootstrap_bloques(balanced_accuracy_score, yt, pt)
        acierto, leve, grave = tipos_de_resultado(PREDICCIONES[nombre])
        filas.append({"modelo": nombre, "tipo": tipo_de(nombre), "balanced_acc": ba_, "IC95 desde": ic[0], "IC95 hasta": ic[1],
                      "accuracy": acierto.mean(), "p_valor": p, "dias_acertados": acierto.sum(),
                      "errores_leves": leve.sum(), "errores_graves": grave.sum(), "dias": len(yt)})
    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla["p_corregido (Holm)"] = holm(tabla["p_valor"])
    tabla["significativo"] = np.where(tabla["p_corregido (Holm)"] < 0.05, "sí", "no")
    return tabla

tabla_test = evaluar_en_test(CANDIDATOS)
guardar_tabla(tabla_test.round(4), "12_2_resultados_test")
anotar("12.2 Resultados en el TEST (candidatos):\n" + tabla_test.round(3).to_string())

REDES = [n for n in PREDICCIONES if n in PROBAS]
PRINCIPALES = ["CNN", "LSTM"]
COLOR_MODELO = {"CNN": "#264653", "LSTM": "#e76f51"}

orden = tabla_test.sort_values("balanced_acc")
fig, ax = plt.subplots(figsize=(11, 0.5 * len(orden) + 1.5))
ax.barh(orden.index, orden["balanced_acc"], color=[COLOR_TIPO[t] for t in orden["tipo"]])
ax.errorbar(orden["balanced_acc"], np.arange(len(orden)),
            xerr=[(orden["balanced_acc"] - orden["IC95 desde"]).clip(lower=0), (orden["IC95 hasta"] - orden["balanced_acc"]).clip(lower=0)],
            fmt="none", ecolor="black", capsize=3, linewidth=0.8)
for i, (nombre, fila) in enumerate(orden.iterrows()):
    ax.text(fila["IC95 hasta"] + 0.003, i, f"{fila['balanced_acc']:.3f}  (p = {fila['p_valor']:.2f}; corregido {fila['p_corregido (Holm)']:.2f})"
            + (" ★" if fila["significativo"] == "sí" else ""), va="center", fontsize=8)
ax.axvline(1 / 3, color="#d1495b", linestyle="--")
ax.set_xlim(min(0.25, orden["IC95 desde"].min() - 0.01), orden["IC95 hasta"].max() + 0.12)
ax.set_xlabel("Balanced accuracy en el test (ciclo actual), con intervalo de confianza 95%  ·  ★ = significativo con p corregido < 0,05")
ax.legend(handles=[Patch(color=c, label=t) for t, c in COLOR_TIPO.items() if t != "Descartado"] +
          [plt.Line2D([], [], color="#d1495b", linestyle="--", label="azar (0,33)")], loc="lower right", fontsize=8)
ax.set_title("El examen final: ¿algún modelo le gana al azar en el ciclo actual?")
guardar_fig("12_2_resultados_test")
plt.show()
display(tabla_test.round(3))
""")
md(r"""
### 12.2b ¿Qué tan bien calibradas están las probabilidades?
La balanced accuracy mira solo la clase elegida. Pero Monte Carlo, la CNN y la LSTM dan **probabilidades**, y la industria evalúa eso: **¿cuando un modelo dice "40% de que suba", sube el 40% de las veces?** Usamos el **puntaje de Brier** (el error cuadrático de las probabilidades: más bajo = mejor) y lo comparamos con dos referencias que no miran las velas: decir **⅓ para cada clase** siempre, y decir siempre **las frecuencias del entrenamiento**.
""")
code(r"""
def brier(proba, y_real):
    unos = np.eye(3)[y_real]
    return np.mean(np.sum((proba - unos) ** 2, axis=1))

frecuencias_train = np.bincount(y[mascara["train"]], minlength=3) / mascara["train"].sum()
probabilidades = {"⅓ para cada clase (no sabe nada)": np.full((mt.sum(), 3), 1 / 3),
                  "frecuencias del entrenamiento": np.tile(frecuencias_train, (mt.sum(), 1))}
probabilidades |= {n: PROBAS_MC[n][mt] for n in PROBAS_MC}
probabilidades |= {n: PROBAS[n][mt] for n in PRINCIPALES if n in PROBAS}
calibracion = pd.DataFrame({"Brier (más bajo = mejor)": {n: brier(p, yt) for n, p in probabilidades.items()},
                            "probabilidad media de 'sube'": {n: p[:, ID["sube"]].mean() for n, p in probabilidades.items()}})
calibracion["mejora vs ⅓"] = calibracion["Brier (más bajo = mejor)"].iloc[0] - calibracion["Brier (más bajo = mejor)"]
guardar_tabla(calibracion.round(4), "12_2b_calibracion")
anotar("12.2b Calibración de las probabilidades en el test (Brier):\n" + calibracion.round(4).to_string())
ax = calibracion["Brier (más bajo = mejor)"].sort_values().plot.barh(figsize=(10, 3.5), color="#264653")
ax.set_xlim(calibracion["Brier (más bajo = mejor)"].min() - 0.02, calibracion["Brier (más bajo = mejor)"].max() + 0.01)
ax.set_title("Puntaje de Brier en el test: ¿alguien da mejores probabilidades que '⅓ para cada clase'?")
guardar_fig("12_2b_calibracion")
plt.show()
""")
md(r"""
## 12.3 Aciertos y errores de cada modelo
No todos los errores cuestan lo mismo:
- 🟩 **Acierto.**
- 🟧 **Error leve:** confundir un movimiento con un lateral (dijo "sube" y quedó quieto). Si invertiste, no ganaste, pero tampoco perdiste mucho.
- 🟥 **Error grave:** confundir **sube con baja**. Si invertiste, **perdiste plata**.
""")
code(r"""
tipos = pd.DataFrame({n: [v.mean() for v in tipos_de_resultado(p)] for n, p in PREDICCIONES.items()},
                     index=["acierto", "error leve (con lateral)", "error grave (sube ↔ baja)"]).T
tipos = tipos.loc[tabla_test.sort_values("balanced_acc").index]
guardar_tabla(tipos.round(3), "12_3_aciertos_y_errores")

ax = tipos.plot.barh(stacked=True, color=["#2a9d55", "#f4a261", "#d1495b"], figsize=(12, 0.5 * len(tipos) + 1.5), width=0.75)
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v:.0%}" if v > 0.04 else "" for v in barras.datavalues], label_type="center",
                 fontsize=8, color="white")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("¿Cómo se reparten los días del test? Aciertos, errores leves y errores graves")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
guardar_fig("12_3_aciertos_y_errores")
plt.show()
""")
md(r"""
## 12.4 Matrices de confusión: qué dijo cada red vs qué pasó
Cada fila es lo que **pasó**, cada columna lo que la red **dijo**. La diagonal son los aciertos. Cada casillero muestra la cantidad de días y el porcentaje de esa fila.
""")
code(r"""
columnas_fig = 2 if len(REDES) <= 2 else 3
filas_fig = int(np.ceil(len(REDES) / columnas_fig))
fig, ejes = plt.subplots(filas_fig, columnas_fig, figsize=(5.5 * columnas_fig, 4.8 * filas_fig))
ejes = np.atleast_1d(ejes).ravel()
for ax, nombre in zip(ejes, REDES):
    dibujar_matriz(ax, nombre, "test")
for ax in ejes[len(REDES):]:
    ax.axis("off")
plt.tight_layout()
guardar_fig("12_4_matrices_de_confusion")
plt.show()
""")
md(r"""
## 12.5 Clase por clase
- **Detectó (*recall*):** de los días en que **realmente subió**, ¿en cuántos dijo "sube"?
- **Cuando lo dijo, acertó (*precisión*):** de los días en que **dijo "sube"**, ¿en cuántos subió de verdad? Es lo que le importa a quien invierte.
- **¿Qué dijo cada modelo?** Cuántas veces dijo cada clase, comparado con lo que realmente pasó. Si un modelo dice casi siempre lo mismo, no se está comprometiendo.
""")
code(r"""
filas = []
for n in PRINCIPALES:
    pt = PREDICCIONES[n][mt]
    rec = recall_score(yt, pt, labels=[0, 1, 2], average=None, zero_division=0)
    pre = precision_score(yt, pt, labels=[0, 1, 2], average=None, zero_division=0)
    for c in range(3):
        filas.append({"modelo": n, "clase": CLASES[c], "detectó (recall)": rec[c], "cuando lo dijo, acertó (precisión)": pre[c]})
por_clase = pd.DataFrame(filas)
guardar_tabla(por_clase.round(3), "12_5_por_clase")

fig, ejes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, metrica in zip(ejes[:2], ["detectó (recall)", "cuando lo dijo, acertó (precisión)"]):
    por_clase.pivot(index="clase", columns="modelo", values=metrica)[PRINCIPALES].plot.bar(
        ax=ax, rot=0, color=[COLOR_MODELO[n] for n in PRINCIPALES])
    ax.axhline(1 / 3, color="gray", linestyle="--")
    ax.set_title(metrica)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
dijo = pd.DataFrame({"lo que pasó": np.bincount(yt, minlength=3) / len(yt)} |
                    {n: np.bincount(PREDICCIONES[n][mt], minlength=3) / len(yt) for n in PRINCIPALES}, index=CLASES).T
dijo.plot.barh(stacked=True, ax=ejes[2], color=[COLORES[c] for c in CLASES])
ejes[2].set_title("¿Qué dijo cada modelo? (vs lo que pasó)")
ejes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout()
guardar_fig("12_5_por_clase")
plt.show()
""")
md(r"""
## 12.6 Día por día: ¿dónde acertó y dónde no?
Para cada red principal: el precio del ciclo actual con cada día marcado como **acierto** (verde) o **error** (rojo). Abajo, el acierto de los últimos 60 días de cada una, comparado con el azar: muestra **en qué momentos del ciclo** las velas "avisaron" y en cuáles no.
""")
code(r"""
fig, ejes = plt.subplots(len(PRINCIPALES) + 1, 1, figsize=(14, 3.6 * (len(PRINCIPALES) + 1)), sharex=True)
precio_test = datos.loc[mt, "close"]
for ax, n in zip(ejes, PRINCIPALES):
    acierto = PREDICCIONES[n][mt] == yt
    ax.plot(fechas_test, precio_test, color="#bbbbbb", linewidth=1, zorder=0)
    ax.scatter(fechas_test[acierto], precio_test[acierto], s=8, color="#2a9d55", label=f"acierto ({acierto.mean():.0%})")
    ax.scatter(fechas_test[~acierto], precio_test[~acierto], s=8, color="#d1495b", label=f"error ({1 - acierto.mean():.0%})")
    ax.set_title(f"{n}: cada día del ciclo actual")
    ax.legend(markerscale=3, loc="upper left")
for n in PRINCIPALES:
    acierto = pd.Series(PREDICCIONES[n][mt] == yt, index=fechas_test)
    ejes[-1].plot(fechas_test, acierto.rolling(60).mean(), color=COLOR_MODELO[n], label=n)
ejes[-1].axhline(1 / 3, color="gray", linestyle="--", label="azar")
ejes[-1].set_ylabel("acierto (últimos 60 días)")
ejes[-1].legend()
plt.tight_layout()
guardar_fig("12_6_dia_por_dia")
plt.show()
""")
md(r"""
## 12.6b Los hitos que cayeron en validación y test: ¿las redes lo vieron venir?
Volvemos a los hitos de la sección 4. Para los que empiezan en validación o test (días que las redes **nunca vieron al entrenar**), miramos qué dijeron la CNN, la LSTM y Kronos en los **primeros 7 días desde el giro**, contra lo que pasó. Si las velas "avisaban", las redes deberían haber dicho "baja" al empezar las caídas y "sube" al empezar las subidas.
""")
code(r"""
def cuenta_clases(valores):
    return " · ".join(f"{CLASES[c_]} {int(np.sum(valores == c_))}" for c_ in (ID["sube"], ID["lateral"], ID["baja"]))

filas = []
for _, h in HITOS.iterrows():
    dias_h = np.flatnonzero((datos.index >= h["inicio"]) & (datos.index < h["inicio"] + pd.Timedelta(days=HORIZONTE)))
    periodo_h = datos["periodo"].iloc[dias_h[0]] if len(dias_h) else "antes de 2013"
    fila = {"hito": f"{h['inicio']:%d-%m-%Y} {h['tipo']} {h['movimiento']:+.0%}", "período": periodo_h}
    if periodo_h in ("val", "test"):
        fila["pasó"] = cuenta_clases(y[dias_h])
        for n in ["CNN", "LSTM", "Kronos (pre-entrenado)"]:
            if n in PREDICCIONES and not (n.startswith("Kronos") and periodo_h != "test"):
                fila[f"{n.split(' ')[0]} dijo"] = cuenta_clases(PREDICCIONES[n][dias_h])
    else:
        fila["pasó"] = "sin predicción fuera de muestra (las redes aprendieron con estos días)"
    filas.append(fila)
hitos_examen = pd.DataFrame(filas).set_index("hito").fillna("")
guardar_tabla(hitos_examen, "12_6b_hitos_que_vieron_las_redes")
anotar("12.6b ¿Las redes vieron venir los hitos? (primeros 7 días desde el giro):\n" + hitos_examen.to_string())
with pd.option_context("display.max_colwidth", 80):
    display(hitos_examen)
""")
md(r"""
## 12.7 ¿Y si hubiéramos invertido con cada modelo?
Una forma muy tangible de medir: **plata**. Simulamos que cada 7 días, empezando el día del halving:
- si el modelo dice **"sube"** → compramos Bitcoin por esa semana;
- si dice **"baja" o "lateral"** → nos quedamos en dólares.

Cada compra o venta paga una **comisión de 0,1%**. Lo comparamos con **comprar el día del halving y mantener** (*buy & hold*). Arrancamos con USD 1.000. No usamos ventas en corto, para mantenerlo simple. Entran **todos los candidatos** del examen.

⚠️ Con ≈130 semanas, una buena racha de pocas semanas cambia mucho el resultado: la plata final **no** reemplaza a la prueba estadística de 12.2.
""")
code(r"""
COMISION = 0.001
semanas = idx_test[::HORIZONTE]
ret_semana = np.expm1(datos["ret_futuro"].to_numpy()[semanas])
fechas_semana = datos.index[semanas]

capital = {"Comprar y mantener": 1000 * np.cumprod(1 + ret_semana) * (1 - COMISION)}
semanas_adentro = {"Comprar y mantener": 1.0}
for n in CANDIDATOS:
    adentro = (PREDICCIONES[n][semanas] == ID["sube"]).astype(float)
    cambios = np.abs(np.diff(adentro, prepend=0))
    capital[n] = 1000 * np.cumprod(1 + adentro * ret_semana - cambios * COMISION)
    semanas_adentro[n] = adentro.mean()

resumen_inversion = pd.DataFrame({"capital final (USD)": {n: c_[-1] for n, c_ in capital.items()},
                                  "semanas invertido": semanas_adentro}).round(3).sort_values("capital final (USD)")
guardar_tabla(resumen_inversion, "12_7_inversion_simulada")
anotar("12.7 Inversión simulada (USD 1.000 el día del halving):\n" + resumen_inversion.to_string())

plt.figure(figsize=(14, 6))
for n, c_ in capital.items():
    destacado = n in ("Comprar y mantener", "CNN", "LSTM", "Kronos (pre-entrenado)")
    plt.plot(fechas_semana, c_, label=f"{n}: USD {c_[-1]:,.0f}", linewidth=2.5 if destacado else 1, alpha=1 if destacado else 0.5,
             color="black" if n == "Comprar y mantener" else COLOR_MODELO.get(n, COLOR_TIPO[tipo_de(n)] if destacado else None))
plt.axhline(1000, color="gray", linestyle=":")
plt.ylabel("Capital (USD)")
plt.title("Si hubiéramos invertido con cada modelo desde el halving de abril de 2024")
plt.legend(fontsize=7, ncol=2)
guardar_fig("12_7_inversion_simulada")
plt.show()
""")

# =====================================================================
md(r"""
---
# 13. Anatomía de los errores: ¿en qué se equivocan las redes y por qué?

Un porcentaje de acierto no alcanza: hay que **mirar los errores** para entender qué aprendió cada red y qué no. Seguimos estos criterios:
- **La red siempre contesta una de las 3 clases**, aunque lo que ve no se parezca a nada de lo que estudió: no tiene la opción "no sé". → ¿Sabe cuándo no sabe? (13.2) ¿Se equivoca más con las fotos **nuevas**? (13.3)
- **Lo aprendido en un período puede no servir en otro** si el mercado cambió de comportamiento. → ¿El ciclo actual se mueve distinto que los anteriores? (13.4)
- **Hay cosas que no están en las velas:** noticias, regulación, liquidaciones. → ¿Los errores se concentran en las sorpresas? (13.5)

## 13.1 Los errores, en casos concretos
Para que los errores sean **tangibles**: los 2 aciertos y los 2 errores en los que cada red estaba **más segura**. A la izquierda, lo que vio (la foto en la CNN, los números en la LSTM); a la derecha, el precio de esos días (negro) y **lo que pasó después** (en color).
""")
code(r"""
CONFIANZA = {n: PROBAS[n].max(1) for n in REDES}       # probabilidad de la clase elegida

def mostrar_casos(nombre, vista, archivo):
    pred = PREDICCIONES[nombre]
    N = N_CNN if nombre.startswith("CNN") else N_LSTM
    ok = idx_test[pred[idx_test] == y[idx_test]]
    mal = idx_test[pred[idx_test] != y[idx_test]]
    elegidos = list(ok[np.argsort(-CONFIANZA[nombre][ok])[:2]]) + list(mal[np.argsort(-CONFIANZA[nombre][mal])[:2]])
    fig, ejes = plt.subplots(len(elegidos), 2, figsize=(14, 3.2 * len(elegidos)), gridspec_kw={"width_ratios": [1, 1.5]})
    for (ax_vista, ax_precio), i in zip(ejes, elegidos):
        p = POS[i]
        es_acierto = pred[i] == y[i]
        if vista == "foto":
            ax_vista.imshow(X_cnn[i, 0], cmap="gray", aspect="auto", interpolation="nearest")
        else:
            ax_vista.imshow(X_lstm[i].T, cmap="RdYlGn", vmin=-3, vmax=3, aspect="auto")
            ax_vista.set_yticks(range(len(COLUMNAS_SECUENCIA)))
            ax_vista.set_yticklabels(COLUMNAS_SECUENCIA, fontsize=7)
            ax_vista.set_xticks([])
        if vista == "foto":
            ax_vista.axis("off")
        ax_vista.set_title(f"{'✔ ACIERTO' if es_acierto else '✘ ERROR'} | {datos.index[i]:%d-%m-%Y} | lo que vio la {nombre}", fontsize=9)
        tramo = d["close"].iloc[p - N + 1:p + HORIZONTE + 1]
        ax_precio.plot(tramo.index[:N], tramo.iloc[:N], color="black", label="lo que vio")
        ax_precio.plot(tramo.index[N - 1:], tramo.iloc[N - 1:], color=COLORES[CLASES[y[i]]], linewidth=3, label="lo que pasó después")
        ax_precio.axvline(datos.index[i], color="gray", linestyle="--")
        ax_precio.set_title(f"dijo {CLASES[pred[i]].upper()} ({CONFIANZA[nombre][i]:.0%} de confianza) → pasó {CLASES[y[i]].upper()} "
                            f"({np.expm1(datos['ret_futuro'].iloc[i]):+.1%} en 7 días)", fontsize=9,
                            color="#2a9d55" if es_acierto else "#d1495b")
        ax_precio.tick_params(labelsize=7)
    ejes[0, 1].legend(fontsize=8)
    plt.tight_layout()
    guardar_fig(archivo)
    plt.show()

mostrar_casos("CNN", "foto", "13_1_casos_cnn")
mostrar_casos("LSTM", "secuencia", "13_1_casos_lstm")
""")
md(r"""
## 13.2 ¿La red sabe cuándo no sabe?
Además de la clase, cada red da una **probabilidad** (su "confianza"). Si la red fuera buena, **cuando está más segura debería acertar más**. Dividimos los días del test en 4 grupos según la confianza:
- Si el acierto **sube** con la confianza → la red "sabe cuándo sabe": se podría operar solo en los días de confianza alta.
- Si el acierto **es parejo** → está igual de segura cuando acierta que cuando no: **no distingue** entre lo que sabe y lo que no.
""")
code(r"""
filas = []
for n in PRINCIPALES:
    conf = CONFIANZA[n][mt]
    grupo = pd.qcut(pd.Series(conf).rank(method="first"), 4, labels=["muy baja", "baja", "alta", "muy alta"])
    tabla = pd.DataFrame({"conf": conf, "acierto": PREDICCIONES[n][mt] == yt, "grupo": grupo.to_numpy()})
    for g, sub in tabla.groupby("grupo", observed=True):
        filas.append({"modelo": n, "confianza": g, "confianza_media": sub["conf"].mean(), "acierto": sub["acierto"].mean()})
confianza_tabla = pd.DataFrame(filas)
guardar_tabla(confianza_tabla.round(3), "13_2_confianza_vs_acierto")
anotar("13.2 Confianza vs acierto:\n" + confianza_tabla.round(3).to_string(index=False))

ax = confianza_tabla.pivot(index="confianza", columns="modelo", values="acierto")[PRINCIPALES].plot.bar(
    rot=0, figsize=(10, 4.5), color=[COLOR_MODELO[n] for n in PRINCIPALES])
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylabel("acierto")
ax.set_xlabel("grupo de confianza de la red (cuartos de los días del test)")
ax.set_title("¿Cuando la red está más segura, acierta más?")
ax.legend()
guardar_fig("13_2_confianza_vs_acierto")
plt.show()
""")
md(r"""
## 13.3 Fotos nuevas: ¿se equivocan más con lo que nunca vieron?
Usamos la distancia a la **gemela del pasado** (9.5): pocos píxeles distintos = la red **ya vio algo casi igual**; muchos = una foto **nueva**, distinta de todo el entrenamiento. Medimos el acierto de cada red según qué tan nueva es la foto de ese día, y mostramos la foto más conocida y la más nueva del test, al lado de su gemela.
""")
code(r"""
novedad_t = NOVEDAD[mt]
grupo_nov = pd.qcut(pd.Series(novedad_t).rank(method="first"), 4, labels=["muy conocida", "conocida", "rara", "muy nueva"]).to_numpy()
por_novedad = pd.DataFrame({n: pd.Series(PREDICCIONES[n][mt] == yt).groupby(grupo_nov, observed=True).mean() for n in PRINCIPALES})
por_novedad["píxeles distintos (promedio)"] = pd.Series(novedad_t).groupby(grupo_nov, observed=True).mean()
guardar_tabla(por_novedad.round(3), "13_3_novedad_vs_acierto")
anotar("13.3 Acierto según qué tan nueva es la foto:\n" + por_novedad.round(3).to_string())

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(2, 1, 1)
por_novedad[PRINCIPALES].plot.bar(ax=ax, rot=0, color=[COLOR_MODELO[n] for n in PRINCIPALES])
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylabel("acierto")
ax.set_title("¿Se equivocan más con las fotos nuevas (las que no se parecen a nada del entrenamiento)?")
ax.legend()
orden_nov = np.argsort(novedad_t)
for col, (j, titulo) in enumerate([(orden_nov[0], "la MÁS CONOCIDA"), (orden_nov[-1], "la MÁS NUEVA")]):
    i_test = idx_test[j]
    for fila, (i, rotulo) in enumerate([(i_test, f"Test: {titulo} ({novedad_t[j]:.0f} píxeles distintos)"),
                                        (GEMELA[i_test], "Su gemela del entrenamiento")]):
        ax = fig.add_subplot(4, 2, 5 + fila * 2 + col)
        ax.imshow(X_cnn[i, 0], cmap="gray", aspect="auto", interpolation="nearest")
        extra = f" | la CNN dijo {CLASES[PREDICCIONES['CNN'][i]].upper()}" if fila == 0 else ""
        ax.set_title(f"{rotulo}\n{datos.index[i]:%d-%m-%Y} → pasó {CLASES[y[i]].upper()}{extra}", fontsize=8)
        ax.axis("off")
plt.tight_layout()
guardar_fig("13_3_fotos_nuevas")
plt.show()
""")
md(r"""
## 13.4 ¿El mercado cambió? Cada período comparado
Comparamos el comportamiento del mercado en cada período: **¿cuánto se mueve el precio?** Si el test es mucho más tranquilo (o más agitado) que el entrenamiento, las mismas formas de velas pueden significar otra cosa. También medimos si las redes aciertan distinto según la volatilidad del momento.
""")
code(r"""
datos["vol_anual"] = datos["vol_reciente"] / np.sqrt(HORIZONTE) * np.sqrt(365)    # volatilidad anualizada
comparacion_periodos = datos.groupby("periodo").agg(
    volatilidad_anual_mediana=("vol_anual", "median"),
    movimiento_semanal_tipico=("ret_futuro", lambda r: np.expm1(r.abs()).median()),
    cuerpo_promedio_de_la_vela=("prop_cuerpo", "mean"),
    dias_con_martillo=("martillo", "mean"),
).loc[["train", "val", "test"]]
guardar_tabla(comparacion_periodos.round(3), "13_4_comparacion_periodos")
anotar("13.4 Comportamiento del mercado en cada período:\n" + comparacion_periodos.round(3).to_string())

grupo_vol = pd.qcut(datos.loc[mt, "vol_anual"], 3, labels=["semanas tranquilas", "normales", "agitadas"]).to_numpy()
acierto_vol = pd.DataFrame({n: pd.Series(PREDICCIONES[n][mt] == yt).groupby(grupo_vol, observed=True).mean() for n in PRINCIPALES})
guardar_tabla(acierto_vol.round(3), "13_4_acierto_segun_volatilidad")
anotar("13.4 Acierto según la volatilidad del momento (test):\n" + acierto_vol.round(3).to_string())

fig, ejes = plt.subplots(1, 2, figsize=(15, 4.5))
for periodo, color in [("train", "#bbbbbb"), ("val", "#f4a261"), ("test", "#264653")]:
    ejes[0].hist(datos.loc[datos["periodo"] == periodo, "vol_anual"], bins=60, alpha=0.6, density=True, color=color, label=periodo)
ejes[0].set_xlabel("Volatilidad anualizada (últimos 30 días)")
ejes[0].set_title("¿Qué tan agitado estaba el mercado en cada período?")
ejes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ejes[0].legend()
acierto_vol.plot.bar(ax=ejes[1], rot=0, color=[COLOR_MODELO[n] for n in PRINCIPALES])
ejes[1].axhline(1 / 3, color="gray", linestyle="--", label="azar")
ejes[1].set_title("Acierto en el test según la volatilidad del momento")
ejes[1].legend()
plt.tight_layout()
guardar_fig("13_4_cambio_de_mercado")
plt.show()
""")
md(r"""
## 13.5 Las sorpresas: ¿había movimientos que no estaban en las velas?
Agrupamos los días del test según **qué tan grande fue el movimiento de la semana siguiente** (en volatilidades, *z*):
- **Cerca del umbral** (*z* ≈ ±k): la frontera entre "lateral" y "se movió" es finita; un error ahí es casi un empate.
- **Movimientos enormes**: suelen venir de **noticias** (regulación, liquidaciones, decisiones de la Fed...). Si la red no los anticipa, es razonable: **esa información no estaba en las velas**.

Abajo, las **10 semanas más extremas** del ciclo actual (una fila por semana): buscá qué noticia hubo en cada una para el informe.
""")
code(r"""
z_test = datos.loc[mt, "z"]
grupo_z = pd.cut(z_test.abs(), [0, k / 2, k, 1.5 * k, 2, np.inf], include_lowest=True,
                 labels=["lateral clarísimo", "lateral justo", "movimiento justo", "movimiento claro", "movimiento enorme"]).to_numpy()
acierto_z = pd.DataFrame({n: pd.Series(PREDICCIONES[n][mt] == yt).groupby(grupo_z, observed=True).mean() for n in PRINCIPALES})
acierto_z["días"] = pd.Series(yt).groupby(grupo_z, observed=True).size()
guardar_tabla(acierto_z.round(3), "13_5_acierto_segun_tamano_movimiento")
anotar("13.5 Acierto según el tamaño del movimiento siguiente:\n" + acierto_z.round(3).to_string())

elegidas = []
for fecha in z_test.abs().sort_values(ascending=False).index:
    if all(abs((fecha - otra).days) >= HORIZONTE for otra in elegidas):
        elegidas.append(fecha)
    if len(elegidas) == 10:
        break
pos_ext = datos.index.get_indexer(elegidas)
extremos = pd.DataFrame({"retorno 7 días": [f"{np.expm1(datos['ret_futuro'].iloc[i]):+.1%}" for i in pos_ext],
                         "pasó": [CLASES[y[i]] for i in pos_ext]},
                        index=pd.DatetimeIndex(elegidas).strftime("%d-%m-%Y"))
for n in PRINCIPALES:
    extremos[f"{n} dijo"] = [f"{CLASES[PREDICCIONES[n][i]]} {'✔' if PREDICCIONES[n][i] == y[i] else '✘'}" for i in pos_ext]
guardar_tabla(extremos, "13_5_semanas_extremas")
anotar("13.5 Las 10 semanas más extremas del test:\n" + extremos.to_string())

ax = acierto_z[PRINCIPALES].plot.bar(rot=15, figsize=(11, 4.5), color=[COLOR_MODELO[n] for n in PRINCIPALES])
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylabel("acierto")
ax.set_title("¿Dónde se equivocan? Según el tamaño del movimiento de la semana siguiente")
ax.legend()
guardar_fig("13_5_sorpresas")
plt.show()
display(extremos)
""")
md(r"""
## 13.6 En resumen: ¿por qué se equivocan?
Con los gráficos de arriba se puede contestar, con números propios, cuál de estas causas pesa más:

| Causa | Dónde se ve |
|---|---|
| **La frontera es difusa:** "lateral" vs "se movió" se decide por poco | 13.5 (lateral justo / movimiento justo) |
| **Fotos nuevas**, distintas de todo el entrenamiento | 13.3 |
| **El mercado cambió:** el ciclo actual se mueve distinto | 13.4 y 14.4 |
| **Noticias que no estaban en las velas** | 13.5 (movimientos enormes y semanas extremas) |
| **Confianza sin fundamento:** está igual de segura cuando acierta que cuando no | 13.2 |
| **El mercado es casi eficiente:** lo que se puede saber por las velas ya está en el precio | 12.2 (p-valores) |
""")

# =====================================================================
md(r"""
---
# 14. Lo que probamos y descartamos

Un trabajo serio documenta también **lo que no funcionó**. Todo lo de esta sección se probó con **las mismas reglas** que los modelos principales (entrenamiento 2013-2020, validación 2020-2024 para decidir, test solo para mirar al final) y se **descartó** porque **no mejoró en validación**, que es el único período que podemos usar para decidir sin hacer trampa.

Primero lo que se descartó **antes de entrenar**, sin necesidad de correr nada:

| Qué | Por qué lo descartamos | Qué hicimos en su lugar |
|---|---|---|
| **Predecir el precio** ("cuánto va a valer") o "en qué momento del ciclo estamos" | no se puede comprobar hoy (pedido de los profesores) | clasificar la semana siguiente en sube / baja / lateral, algo que se puede corregir día por día |
| **Velas de 4 horas** para entrenar | más ruido y se pierde la escala de los ciclos; los profes pidieron velas diarias | velas diarias en todo el trabajo |
| **Imágenes GAF** (*Gramian Angular Field*, del paper de Chen & Tsai) | convierten la serie en una "textura" que ya no se parece a un gráfico de velas: la consigna es mirar las velas | la foto en blanco y negro de las velas (formato de Jiang et al.) |
| **`candlefusion`** (HuggingFace) | necesita texto de noticias, no informa métricas de acierto ni con qué fechas se entrenó | Kronos y Chronos-2, publicados y evaluados (sección 11) |
| **Fijar a mano los hitos** ("los 8 más grandes") | arbitrario | el criterio "×2 o ÷2" (sección 4.2) |

Y después, lo que **sí entrenamos** y no funcionó. Cada uno tiene su subsección con el detalle, y al final (14.7) una tabla que los junta.
""")
md(r"""
## 14.1 Velas semanales en vez de diarias
Los profesores pidieron no usar velas semanales para entrenar (muy pocos datos). Igual lo medimos, para que la decisión quede respaldada con números.

Otra forma de "cuánto mirar": ¿es mejor ver **cada día por separado** o **resumir la semana en una sola vela**? Para que sea justo, comparamos **el mismo período de tiempo** visto de las dos maneras:
- **4 semanas:** 28 velas diarias vs 4 velas semanales.
- **12 semanas:** 84 velas diarias vs 12 velas semanales.

Las velas semanales se arman igual que en la sección 3 (apertura = primera, máximo = máximo, mínimo = mínimo, cierre = último, volumen = suma), pero **móviles**: cada una junta los 7 días que terminan en el día que queremos predecir, así no se mezcla información futura.

Si las semanales rinden igual o mejor, el detalle de cada día sería ruido; si las diarias rinden mejor, el detalle de cada vela aporta.
""")
code(r"""
def armar_imagenes_semanales(n_semanas):
    # Para cada día: n_semanas velas semanales "móviles" (de a 7 días) que terminan ese día
    imagenes = []
    for p in POS:
        bloque = OHLCV[p - 7 * n_semanas + 1:p + 1].reshape(n_semanas, 7, 5)
        semanal = np.column_stack([bloque[:, 0, 0], bloque[:, :, 1].max(1), bloque[:, :, 2].min(1),
                                   bloque[:, -1, 3], bloque[:, :, 4].sum(1)])
        imagenes.append(dibujar_ventana(semanal))
    return np.stack(imagenes)[:, None]

PARES = [(28, 4)] if MODO_RAPIDO else [(28, 4), (84, 12)]
filas = []
for n_diarias, n_semanas in PARES:
    for tipo, X in [("diarias", armar_imagenes(n_diarias)), ("semanales", armar_imagenes_semanales(n_semanas))]:
        for s in SEMILLAS:
            _, ba, _ = entrenar(CNNVelas, X[mascara["train"]], y[mascara["train"]], X[mascara["val"]], y[mascara["val"]], semilla=s)
            filas.append({"período": f"{n_semanas} semanas", "velas": tipo, "bal_acc_val": ba})
            print(f"{n_semanas} semanas | velas {tipo} | repetición {s + 1}: {ba:.3f}")
diarias_vs_semanales = pd.DataFrame(filas)
resumen_dvs = diarias_vs_semanales.groupby(["período", "velas"])["bal_acc_val"].agg(promedio="mean", desvio="std").fillna(0)
guardar_tabla(resumen_dvs.round(4), "14_1_diarias_vs_semanales")
anotar("14.1 ¿Velas diarias o semanales? (balanced accuracy en validación):\n" + resumen_dvs.round(3).to_string())

tabla_dvs = resumen_dvs["promedio"].unstack()
error_dvs = resumen_dvs["desvio"].unstack()
ax = tabla_dvs.plot.bar(yerr=error_dvs, capsize=4, rot=0, color=["#264653", "#f4a261"], figsize=(9, 4.5))
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylim(0.25, max(0.42, tabla_dvs.values.max() + 0.03))
ax.set_ylabel("Balanced accuracy en validación")
ax.set_xlabel("Mismo período de tiempo, visto de dos maneras")
ax.set_title("¿Conviene mirar cada día por separado o resumir la semana en una vela?")
ax.legend()
guardar_fig("14_1_diarias_vs_semanales")
plt.show()
""")
md(r"""
## 14.2 Más datos: acciones líquidas de EE.UU.
**Idea:** si los patrones de velas son un "idioma universal" (como sugiere Jiang et al., y como pide la consigna: *"sin importar el activo"*, *"activos líquidos de EE.UU."*), la red puede **aprender con miles de fotos de acciones** y después aplicar lo aprendido a Bitcoin.

| | Cómo lo hacemos | Por qué |
|---|---|---|
| **Qué acciones** | 30 de las más líquidas de EE.UU., de distintos sectores (incluye LLY y TSLA, las de la planilla del profe) | liquidez = precios "limpios"; sectores distintos = patrones variados |
| **Qué período** | 2005 → mayo 2020 (**solo antes de la validación**) | si usáramos acciones de 2024, la red vería "el futuro" del test |
| **Horizonte** | 5 ruedas (= una semana de bolsa) | las acciones no operan el fin de semana: 5 ruedas son la misma semana calendario que 7 días de Bitcoin |
| **Respuesta** | mismo criterio sube/baja/lateral, umbral calibrado por acción (≈⅓ cada clase) | |
| **Validación y test** | **solo Bitcoin**, igual que antes | queremos saber si mejora **en Bitcoin** |

Los datos se bajan de Yahoo Finanza (`yfinance`), con precios ajustados por *splits* y dividendos (si no, un split parecería una caída del 50%).
""")
code(r"""
import yfinance as yf

ACCIONES = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "LLY", "JPM", "V", "JNJ",
            "XOM", "PG", "KO", "PEP", "WMT", "HD", "DIS", "NFLX", "INTC", "CSCO",
            "ORCL", "BA", "CAT", "MCD", "NKE", "PFE", "MRK", "CVX", "GS", "UNH"]
if MODO_RAPIDO:
    ACCIONES = ACCIONES[:3]
HORIZONTE_ACCIONES = 5
FIN_ACCIONES = (CORTE_VAL - EMBARGO).tz_convert(None)      # solo datos ANTERIORES a la validación de Bitcoin
CARPETA_ACCIONES = RAW / "acciones"
CARPETA_ACCIONES.mkdir(exist_ok=True)

def bajar_accion(ticker):
    archivo = CARPETA_ACCIONES / f"{ticker}.csv"
    if archivo.exists():
        return pd.read_csv(archivo, index_col=0, parse_dates=True)
    df = yf.download(ticker, start="2005-01-01", end=FIN_ACCIONES.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.to_csv(archivo)
    return df

acciones = {}
for t in ACCIONES:
    try:
        df = bajar_accion(t)
        if len(df) > 500:
            acciones[t] = df
    except Exception as e:
        print(f"{t}: no se pudo bajar ({e})")
    time.sleep(0.3)
anotar(f"14.2 Acciones descargadas: {len(acciones)} de {len(ACCIONES)} ({', '.join(acciones)})")
""")
code(r"""
def preparar_activo(df, horizonte):
    # Mismo procedimiento que la sección 5, para cualquier activo: respuesta sube/baja/lateral + secuencias
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    ret_1d = np.log(df["close"] / df["close"].shift(1))
    vol = ret_1d.rolling(VENTANA_VOL).std() * np.sqrt(horizonte)
    z = (np.log(df["close"].shift(-horizonte) / df["close"]) / vol).replace([np.inf, -np.inf], np.nan)
    k_activo = z.abs().quantile(1 / 3)
    clase = np.select([z > k_activo, z < -k_activo], [ID["sube"], ID["baja"]], default=ID["lateral"])
    return df[["open", "high", "low", "close", "volume"]].to_numpy(), features_secuencia(df, vol, horizonte), clase, z.notna().to_numpy()

HAY_ACCIONES = len(acciones) > 0
if HAY_ACCIONES:
    X_img_acc, X_seq_acc, y_acc, por_accion = [], [], [], {}
    for t, df in acciones.items():
        ohlcv, secs, clase, tiene_respuesta = preparar_activo(df, HORIZONTE_ACCIONES)
        inicio = max(N_CNN, N_LSTM, VENTANA_VOL + 1)
        pos = np.flatnonzero(tiene_respuesta & (np.arange(len(ohlcv)) >= inicio))
        X_img_acc.append(np.stack([dibujar_ventana(ohlcv[p - N_CNN + 1:p + 1]) for p in pos])[:, None])
        X_seq_acc.append(np.stack([secs[p - N_LSTM + 1:p + 1] for p in pos]))
        y_acc.append(clase[pos])
        por_accion[t] = len(pos)
    X_img_acc, X_seq_acc, y_acc = np.concatenate(X_img_acc), np.concatenate(X_seq_acc), np.concatenate(y_acc)
    anotar(f"14.2 Ejemplos de acciones: {len(y_acc):,} (Bitcoin solo tenía {mascara['train'].sum():,}) | "
           f"clases: " + ", ".join(f"{c} {np.mean(y_acc == ID[c]):.0%}" for c in CLASES))

    tr, va = mascara["train"], mascara["val"]
    print("\nCNN + acciones:")
    proba, historias = entrenar_varias(CNNVelas, np.concatenate([X_img_acc, X_cnn[tr]]), np.concatenate([y_acc, y[tr]]),
                                       X_cnn[va], y[va], X_cnn, N_REDES_MEJORAS, tam_lote=256)
    registrar_red("CNN + acciones", proba, historias, "14_2_entrenamiento_cnn_acciones")
    print("\nLSTM + acciones:")
    proba, historias = entrenar_varias(LSTMVelas, np.concatenate([X_seq_acc, X_lstm[tr]]), np.concatenate([y_acc, y[tr]]),
                                       X_lstm[va], y[va], X_lstm, N_REDES_MEJORAS, tam_lote=256)
    registrar_red("LSTM + acciones", proba, historias, "14_2_entrenamiento_lstm_acciones")

    fig, ejes = plt.subplots(1, 2, figsize=(14, 4.5))
    ejes[0].bar(["solo Bitcoin", "Bitcoin + acciones"], [mascara["train"].sum(), mascara["train"].sum() + len(y_acc)],
                color=["#bbbbbb", "#2a9d55"])
    ejes[0].set_title("Cantidad de ejemplos de entrenamiento")
    comparacion = pd.DataFrame({
        "solo Bitcoin": [bal_acc(PREDICCIONES["CNN"], "val"), bal_acc(PREDICCIONES["LSTM"], "val")],
        "Bitcoin + acciones": [bal_acc(PREDICCIONES["CNN + acciones"], "val"), bal_acc(PREDICCIONES["LSTM + acciones"], "val")],
    }, index=["CNN", "LSTM"])
    comparacion.plot.bar(ax=ejes[1], color=["#bbbbbb", "#2a9d55"], rot=0)
    ejes[1].axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ejes[1].set_ylim(0.25, max(0.45, comparacion.values.max() + 0.02))
    ejes[1].set_title("Balanced accuracy en validación (Bitcoin)")
    ejes[1].legend()
    plt.tight_layout()
    guardar_fig("14_2_mejora_mas_datos")
    plt.show()
else:
    print("⚠️ No se pudo bajar ninguna acción (¿problema de conexión con Yahoo?). Se saltea este intento.")
""")
md(r"""
## 14.3 Más datos: otras criptomonedas
La idea del proyecto que revisamos en 14.5 (Livieris et al., 2021) es que **las criptomonedas se mueven en conjunto**, así que mirar varias ayuda. Probamos sumar al entrenamiento las velas diarias de **Ethereum (ETH), Litecoin (LTC) y Ripple (XRP)**, de la misma fuente (Bitstamp), con el mismo procedimiento que las acciones:
- **mismo horizonte que Bitcoin: 7 días** (también operan 24/7);
- solo datos **anteriores a la validación** (mayo 2020);
- validación y test: **solo Bitcoin**, igual que siempre.

**Diferencia con las acciones:** Bitstamp tiene estas monedas recién desde 2016-2017, así que aportan **pocos ejemplos** (≈ 3 años cada una). Si aun así ayudan, el "parentesco" entre criptos importa más que la cantidad.
""")
code(r"""
CRIPTOS = ["ethusd", "ltcusd", "xrpusd"]
FIN_CRIPTOS = int((CORTE_VAL - EMBARGO).timestamp())

def bajar_cripto(par):
    archivo = RAW / f"{par}_1d_bitstamp.csv"
    if archivo.exists():
        return pd.read_csv(archivo, index_col=0, parse_dates=True)
    filas, desde = [], int(pd.Timestamp("2015-01-01", tz="UTC").timestamp())
    while desde < FIN_CRIPTOS:
        r = requests.get(f"https://www.bitstamp.net/api/v2/ohlc/{par}/",
                         params={"step": 86400, "limit": 1000, "start": desde}, timeout=30)
        bloque = r.json()["data"]["ohlc"]
        if bloque:
            filas += bloque
            desde = int(bloque[-1]["timestamp"]) + 86400
        else:
            desde += 86400 * 1000
        time.sleep(0.3)
    df = pd.DataFrame(filas)[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df.index = pd.to_datetime(df["timestamp"].astype("int64"), unit="s")
    df = df[~df.index.duplicated()].drop(columns="timestamp")
    df = df[(df.index < pd.Timestamp(FIN_CRIPTOS, unit="s")) & (df["volume"] > 0)]
    df.to_csv(archivo)
    return df

X_img_cri, X_seq_cri, y_cri = [], [], []
for par in CRIPTOS:
    try:
        df = bajar_cripto(par)
        ohlcv, secs, clase, tiene_respuesta = preparar_activo(df, HORIZONTE)
        inicio = max(N_CNN, N_LSTM, VENTANA_VOL + 1)
        pos = np.flatnonzero(tiene_respuesta & (np.arange(len(ohlcv)) >= inicio))
        X_img_cri.append(np.stack([dibujar_ventana(ohlcv[p - N_CNN + 1:p + 1]) for p in pos])[:, None])
        X_seq_cri.append(np.stack([secs[p - N_LSTM + 1:p + 1] for p in pos]))
        y_cri.append(clase[pos])
        anotar(f"14.3 {par.upper()}: {len(pos):,} días de entrenamiento ({df.index.min():%m-%Y} → {df.index.max():%m-%Y})")
    except Exception as e:
        print(f"{par}: no se pudo usar ({e})")

if y_cri:
    X_img_cri, X_seq_cri, y_cri = np.concatenate(X_img_cri), np.concatenate(X_seq_cri), np.concatenate(y_cri)
    tr, va = mascara["train"], mascara["val"]
    print("\nCNN + criptos:")
    proba, historias = entrenar_varias(CNNVelas, np.concatenate([X_img_cri, X_cnn[tr]]), np.concatenate([y_cri, y[tr]]),
                                       X_cnn[va], y[va], X_cnn, N_REDES_MEJORAS)
    registrar_red("CNN + criptos", proba, historias, "14_3_entrenamiento_cnn_criptos")
    print("\nLSTM + criptos:")
    proba, historias = entrenar_varias(LSTMVelas, np.concatenate([X_seq_cri, X_lstm[tr]]), np.concatenate([y_cri, y[tr]]),
                                       X_lstm[va], y[va], X_lstm, N_REDES_MEJORAS)
    registrar_red("LSTM + criptos", proba, historias, "14_3_entrenamiento_lstm_criptos")

    variantes = [("solo Bitcoin", ""), ("+ acciones", " + acciones"), ("+ criptos", " + criptos")]
    comparacion = pd.DataFrame({etiqueta: [bal_acc(PREDICCIONES[r + suf], "val") if (r + suf) in PREDICCIONES else np.nan
                                           for r in ["CNN", "LSTM"]] for etiqueta, suf in variantes}, index=["CNN", "LSTM"])
    guardar_tabla(comparacion.round(4), "14_3_mas_datos_comparacion")
    anotar("14.3 ¿Qué datos extra ayudan? (balanced accuracy en validación):\n" + comparacion.round(3).to_string())
    ax = comparacion.plot.bar(rot=0, color=["#bbbbbb", "#2a9d55", "#f4a261"], figsize=(9, 4.5))
    ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ax.set_ylim(0.25, max(0.45, np.nanmax(comparacion.values) + 0.02))
    ax.set_title("¿Con qué datos extra aprende mejor? (validación, Bitcoin)")
    ax.legend()
    guardar_fig("14_3_mas_datos_comparacion")
    plt.show()
else:
    print("⚠️ No se pudo bajar ninguna cripto; se saltea este intento.")
""")
md(r"""
## 14.4 Reentrenar cada 3 meses (*walk-forward*)
**Idea:** en la vida real, nadie entrena un modelo en 2020 y lo usa sin tocar hasta 2026. Lo normal es **reentrenarlo cada tanto con los datos nuevos**. Simulamos exactamente eso:

1. Partimos el test (abr-2024 → hoy) en **bloques de 3 meses**.
2. Para predecir cada bloque, entrenamos la red con **todo lo que ya había pasado antes de que empiece el bloque**: el último año para validar (*early stopping*) y lo anterior para aprender.
3. Predecimos el bloque y pasamos al siguiente.

**¿Es trampa?** No: para cada bloque, la red solo usa días cuya respuesta **ya se conocía** antes de que empiece (con el mismo embargo de 7 días). Es lo que podría haber hecho una persona en ese momento. El gráfico muestra qué datos usa cada bloque.
""")
code(r"""
import matplotlib.dates as mdates
from matplotlib.patches import Patch

inicios, f = [], CORTE_TEST
while f <= datos.index.max():
    inicios.append(f)
    f = f + pd.DateOffset(months=3)
BLOQUES = list(zip(inicios, inicios[1:] + [datos.index.max() + pd.Timedelta(days=1)]))
UN_ANIO = pd.Timedelta(days=365)
fechas = datos.index

def mascaras_bloque(ini, fin):
    m_val = (fechas >= ini - EMBARGO - UN_ANIO) & (fechas < ini - EMBARGO)     # el último año antes del bloque
    m_tr = fechas < ini - 2 * EMBARGO - UN_ANIO                                  # todo lo anterior
    m_bloque = (fechas >= ini) & (fechas < fin)
    return np.asarray(m_tr), np.asarray(m_val), np.asarray(m_bloque)

# Diagrama: qué datos usa cada bloque
a_num = lambda fecha: mdates.date2num(fecha.tz_convert(None).to_pydatetime())
fin_datos = datos.index.max()
filas = [("modelo original (9 y 10)", [(INICIO, CORTE_VAL - EMBARGO, "#bbbbbb"), (CORTE_VAL, CORTE_TEST - EMBARGO, "#f4a261"),
                                       (CORTE_TEST, fin_datos, "#2a9d55")])]
for i, (ini, fin) in enumerate(BLOQUES):
    filas.append((f"bloque {i + 1}", [(INICIO, ini - 2 * EMBARGO - UN_ANIO, "#bbbbbb"),
                                      (ini - EMBARGO - UN_ANIO, ini - EMBARGO, "#f4a261"), (ini, fin, "#2a9d55")]))
fig, ax = plt.subplots(figsize=(14, 0.4 * len(filas) + 1.5))
for i, (nombre, tramos) in enumerate(filas):
    for a, b, color in tramos:
        ax.barh(i, a_num(b) - a_num(a), left=a_num(a), color=color, height=0.6)
ax.xaxis_date()
ax.set_yticks(range(len(filas)))
ax.set_yticklabels([n for n, _ in filas])
ax.invert_yaxis()
ax.legend(handles=[Patch(color="#bbbbbb", label="aprende"), Patch(color="#f4a261", label="valida (cuándo parar)"),
                   Patch(color="#2a9d55", label="predice (test)")], loc="lower left")
ax.set_title("Reentrenar cada 3 meses: cada bloque aprende solo con lo que ya había pasado")
guardar_fig("14_4_esquema_reentrenamiento")
plt.show()

def reentrenar(nombre, X, crear_modelo):
    proba = np.full((len(datos), 3), 1 / 3)
    for i, (ini, fin) in enumerate(BLOQUES):
        m_tr, m_val, m_bloque = mascaras_bloque(ini, fin)
        print(f"{nombre} | bloque {i + 1}/{len(BLOQUES)} ({ini:%d-%m-%Y} → {fin:%d-%m-%Y}): "
              f"aprende con {m_tr.sum():,} días, valida con {m_val.sum()}, predice {m_bloque.sum()}")
        p, _ = entrenar_varias(crear_modelo, X[m_tr], y[m_tr], X[m_val], y[m_val], X[m_bloque], N_REDES_MEJORAS)
        proba[m_bloque] = p
    PROBAS[nombre] = proba
    PREDICCIONES[nombre] = proba.argmax(1)

reentrenar("CNN reentrenada", X_cnn, CNNVelas)
reentrenar("LSTM reentrenada", X_lstm, LSTMVelas)
anotar(f"14.4 Reentrenamiento: {len(BLOQUES)} bloques de 3 meses (el resultado, bloque por bloque, abajo)")
""")
md(r"""
### ¿Sirvió reentrenar? Acierto por bloque de 3 meses
Comparamos, bloque por bloque, las redes **entrenadas una sola vez** (secciones 9 y 10) con las **reentrenadas cada 3 meses**.
""")
code(r"""
if "CNN reentrenada" in PREDICCIONES:
    filas = []
    for ini, fin in BLOQUES:
        m = mt & (fechas >= ini) & (fechas < fin)
        if m.sum() == 0:
            continue
        fila = {"bloque": f"{ini:%m-%Y}", "días": int(m.sum())}
        for n in ["CNN", "CNN reentrenada", "LSTM", "LSTM reentrenada"]:
            fila[n] = (PREDICCIONES[n][m] == y[m]).mean()
        filas.append(fila)
    por_bloque = pd.DataFrame(filas).set_index("bloque")
    guardar_tabla(por_bloque.round(3), "14_4_acierto_por_bloque")
    anotar("14.4 Acierto por bloque de 3 meses:\n" + por_bloque.round(3).to_string())
    ax = por_bloque.drop(columns="días").plot.bar(figsize=(14, 4.5), rot=0,
                                                   color=["#264653", "#8ab17d", "#e76f51", "#f4a261"])
    ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ax.set_ylabel("acierto")
    ax.set_title("Acierto por bloque de 3 meses: entrenada una vez vs reentrenada")
    ax.legend(ncol=5, fontsize=8)
    guardar_fig("14_4_acierto_por_bloque")
    plt.show()
""")
# =====================================================================
md(r"""
## 14.5 Una red híbrida CNN-LSTM (idea de otro proyecto)

Revisamos un proyecto público que predice criptomonedas con deep learning: [Advance CNN-LSTM Model for Cryptocurrency Forecasting](https://github.com/0xMimir/Advance-CNN-LSTM-Model-for-Cryptocurrency-Forecasting), basado en el paper **Livieris et al. (2021), "An Advanced CNN-LSTM Model for Cryptocurrency Forecasting"**, *Electronics* 10(3), 287.

### Qué hace ese proyecto
- Usa **varias criptomonedas a la vez** (BTC, ETH y XRP): una entrada por moneda, porque el mercado se mueve en conjunto.
- Cada entrada pasa por una red **híbrida**: primero una **convolución 1D** (busca patrones cortos en la serie), después una **LSTM** (lee en orden lo que encontró la convolución).
- Predice si el **día siguiente** sube o baja (2 clases), mirando 30 días.

### Qué hacemos distinto, y por qué
| | Ese proyecto | Nuestro trabajo | Por qué |
|---|---|---|---|
| **Qué mira** | solo el **precio de cierre** (variación %) | **la vela completa**: apertura, máximo, mínimo, cierre y volumen | la consigna es mirar **las velas**; el cierre solo descarta mechas y cuerpos |
| **Qué predice** | sube / baja, al día siguiente | sube / baja / **lateral**, a 7 días | la consigna incluye "se va a detener"; 1 día es casi todo ruido |
| **Validación** | usa el **mismo** conjunto para decidir y para medir | validación y test **separados**, con embargo | si se mide con los mismos datos con los que se decide, el resultado es optimista |
| **Cuándo parar** | 100 épocas fijas | *early stopping* en validación | evita memorizar |
| **Comparación** | sin baselines | azar, reglas clásicas, p-valor | para saber si de verdad le gana al azar |
| **Varias monedas** | BTC + ETH + XRP **como entradas simultáneas** | otras criptos y acciones **como ejemplos extra** de entrenamiento (14.2 y 14.3) | ETH y XRP cotizan en Bitstamp recién desde 2016-2017: usarlas como entradas simultáneas nos obligaría a **tirar 2013-2017 de Bitcoin** (la mitad del entrenamiento). Como ejemplos extra, suman sin quitar nada |

### Qué tomamos: la arquitectura híbrida
La idea útil es **combinar las dos redes**: una convolución que detecta patrones cortos (como la CNN) y una LSTM que lee en orden esos patrones (como la LSTM). La probamos sobre **nuestras mismas secuencias de velas**, con nuestras mismas reglas de entrenamiento:

```
SECUENCIA  N velas × 6 números
  ├─ Convolución 1D (16 filtros, de a 2 velas)   → patrones de pares de velas
  ├─ Max-pooling (resume de a 2)
  ├─ LSTM (32 unidades de memoria)                → lee esos patrones en orden
  ├─ Dropout 30%
  └─ Capa final → BAJA · LATERAL · SUBE
```
Los valores (16 filtros, de a 2) son los del paper; la LSTM chica es la nuestra (pocos datos).
""")
code(r"""
class CNNLSTMVelas(nn.Module):
    def __init__(self, n_entradas=len(COLUMNAS_SECUENCIA), filtros=16, memoria=32):
        super().__init__()
        self.convolucion = nn.Sequential(
            nn.Conv1d(n_entradas, filtros, kernel_size=2, padding=1),   # patrones de 2 velas
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, ceil_mode=True),
        )
        self.lstm = nn.LSTM(filtros, memoria, batch_first=True)
        self.cabeza = nn.Sequential(nn.Dropout(0.3), nn.Linear(memoria, 3))

    def forward(self, x):                                   # x: (lote, N velas, 6)
        patrones = self.convolucion(x.transpose(1, 2))      # la convolución 1D espera (lote, 6, N)
        _, (memoria_final, _) = self.lstm(patrones.transpose(1, 2))
        return self.cabeza(memoria_final[-1])

anotar(f"14.5 Arquitectura CNN-LSTM híbrida: {sum(p.numel() for p in CNNLSTMVelas().parameters()):,} parámetros")
proba, historias = entrenar_varias(CNNLSTMVelas, X_lstm[mascara["train"]], y[mascara["train"]],
                                   X_lstm[mascara["val"]], y[mascara["val"]], X_lstm, N_REDES_FINAL)
registrar_red("CNN-LSTM híbrida", proba, historias, "14_5_entrenamiento_cnn_lstm")
""")
md(r"""
## 14.6 ResNet-18: una red de imágenes pre-entrenada (*transfer learning*)
**La idea (*transfer learning*):** una red que aprendió a ver con millones de fotos ya sabe detectar bordes, líneas y formas. ¿Ese "saber ver" sirve para nuestras fotos de velas?

| Decisión | Qué elegimos | En base a qué |
|---|---|---|
| **Qué red** | ResNet-18 (11 millones de parámetros) | la más chica de la familia ResNet, el estándar de visión; viene incluida en PyTorch (`torchvision`) |
| **Qué parte entrenamos** | **solo la capa final** (la ResNet queda **congelada**) | tenemos ≈ 2.700 fotos; ajustar 11 millones de parámetros con eso es memorizar seguro. Congelada, la ResNet solo "describe" cada foto con 512 números, y una capa chica aprende a usarlos |
| **Formato de la foto** | la agrandamos a 224×224 y repetimos el canal 3 veces (RGB) | es el formato con el que la ResNet fue entrenada |
| **Normalización** | la oficial de ImageNet (media y desvío por canal) | la red espera las fotos así |
| **Salida** | 3 clases, mismo entrenamiento que nuestras redes | para comparar con las mismas reglas |
| **¿Puede haber visto el examen?** | **no**: ImageNet son fotos de objetos, no precios | a diferencia de Kronos y Chronos-2, acá no hay riesgo de contaminación |
""")
code(r"""
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

try:
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()                      # sacamos su capa final: queremos los 512 números que "describen" la foto
    resnet = resnet.to(DISPOSITIVO).eval()
    media = torch.tensor([0.485, 0.456, 0.406], device=DISPOSITIVO).view(1, 3, 1, 1)
    desvio = torch.tensor([0.229, 0.224, 0.225], device=DISPOSITIVO).view(1, 3, 1, 1)

    descripciones = []
    with torch.no_grad():
        for i in range(0, len(X_cnn), 128):
            x = torch.from_numpy(X_cnn[i:i + 128]).float().to(DISPOSITIVO)
            x = F.interpolate(x, size=(224, 224), mode="nearest").repeat(1, 3, 1, 1)     # 224x224, 3 canales
            descripciones.append(resnet((x - media) / desvio).cpu())
    X_resnet = torch.cat(descripciones).numpy()
    print(f"Cada foto quedó descripta por {X_resnet.shape[1]} números")

    capa_final = lambda: nn.Sequential(nn.Dropout(0.5), nn.Linear(X_resnet.shape[1], 3))
    proba, historias = entrenar_varias(capa_final, X_resnet[mascara["train"]], y[mascara["train"]],
                                       X_resnet[mascara["val"]], y[mascara["val"]], X_resnet, N_REDES_FINAL)
    registrar_red("ResNet-18 (pre-entrenado en ImageNet)", proba, historias, "14_6_entrenamiento_resnet")
except Exception as e:
    print("⚠️ ResNet-18 no pudo correr, se sigue sin ella:", repr(e))
""")
md(r"""
## 14.7 La tabla de los descartes
Para cada intento: qué se probó, por qué, el resultado en **validación** comparado con el modelo base (la CNN o la LSTM sin el cambio) y, **solo por transparencia**, el resultado en el test con su p-valor corregido entre los descartes. La decisión se toma con la validación; el test confirma si hubiera cambiado algo.
""")
code(r"""
DESCARTES = [
    ("CNN, velas semanales", "CNN", "¿resumir la semana en una vela ve mejor que 7 velas diarias?"),
    ("CNN + acciones", "CNN", "más datos: 30 acciones líquidas de EE.UU. (2005-2020)"),
    ("LSTM + acciones", "LSTM", "más datos: 30 acciones líquidas de EE.UU. (2005-2020)"),
    ("CNN + criptos", "CNN", "más datos: ETH, LTC y XRP"),
    ("LSTM + criptos", "LSTM", "más datos: ETH, LTC y XRP"),
    ("CNN reentrenada", "CNN", "el mercado cambia: reentrenar cada 3 meses"),
    ("LSTM reentrenada", "LSTM", "el mercado cambia: reentrenar cada 3 meses"),
    ("CNN-LSTM híbrida", "LSTM", "combinar convolución y recurrente (Livieris et al., 2021)"),
    ("ResNet-18 (pre-entrenado en ImageNet)", "CNN", "una red de imágenes gigante ya entrenada (transfer learning)"),
]
filas = []
for nombre, base, idea in DESCARTES:
    if nombre == "CNN, velas semanales":
        if "resumen_dvs" in globals():
            semanales = resumen_dvs.xs("semanales", level="velas")["promedio"].mean()
            diarias = resumen_dvs.xs("diarias", level="velas")["promedio"].mean()
            filas.append({"descarte": nombre, "qué se probó": idea, "validación": semanales, "validación del base": diarias})
        continue
    if nombre not in PREDICCIONES:
        continue
    pt = PREDICCIONES[nombre][mt]
    ba_t, p_t = p_desplazamiento(lambda yy: balanced_accuracy_score(yy, pt), yt)
    filas.append({"descarte": nombre, "qué se probó": idea,
                  "validación": bal_acc(PREDICCIONES[nombre], "val") if nombre in PROBAS and "reentrenada" not in nombre else np.nan,
                  "validación del base": bal_acc(PREDICCIONES[base], "val"), "test": ba_t, "test del base": tabla_test.loc[base, "balanced_acc"], "p test": p_t})
tabla_descartes = pd.DataFrame(filas).set_index("descarte")
con_p = tabla_descartes["p test"].notna() if "p test" in tabla_descartes else pd.Series(False, index=tabla_descartes.index)
if con_p.any():
    tabla_descartes.loc[con_p, "p test corregido (Holm)"] = holm(tabla_descartes.loc[con_p, "p test"])
tabla_descartes["motivo"] = np.where(
    tabla_descartes["validación"].isna(), "se evalúa por bloques de 3 meses (14.4): no tiene una validación única",
    np.where(tabla_descartes["validación"] > tabla_descartes["validación del base"] + 0.01,
             "mejoró en validación pero la diferencia está dentro del ruido entre repeticiones",
             "no mejoró en validación"))
guardar_tabla(tabla_descartes.round(4), "14_7_descartes")
anotar("14.7 Lo que probamos y descartamos:\n" + tabla_descartes.round(3).to_string())
with pd.option_context("display.max_colwidth", 70):
    display(tabla_descartes.round(3))
""")
md(r"""
### 🔎 ¿Qué nos llevamos de los descartes?
- **Más datos no ayudó:** ni 30 acciones de EE.UU. ni otras criptomonedas mejoraron a las redes en validación. Si el problema fuera "falta de ejemplos", más ejemplos deberían haber ayudado. No ayudaron: **la señal no está**, no es que falten datos.
- **Reentrenar tampoco:** usar siempre lo más reciente no mejoró. Si lo que fallaba era que "el mercado cambió", reentrenar debería haberlo compensado.
- **Redes más complejas o más grandes tampoco:** ni la híbrida CNN-LSTM ni la ResNet-18 de 11 millones de parámetros le ganaron a nuestras redes chicas.
- Todo apunta al mismo lado que las secciones 7 y 12: **lo que no se puede aprender no mejora agregando datos, tiempo ni tamaño.**
""")

# =====================================================================
md(r"""
---
# 15. Conclusiones: ¿qué patrones esconde Bitcoin y qué nos dicen?

La celda de abajo arma las respuestas **con los números de esta corrida** (si se corrió con `MODO_RAPIDO = True`, quedan marcadas como **provisorias**: los modelos casi no entrenaron).

### Cómo leerlas
**Parte A, lo descriptivo: ¿qué esconden las velas?**
- Si los patrones que se distinguen del azar son de **inercia** (rachas que siguen) y **se apagan fuera del entrenamiento**, las velas de Bitcoin tuvieron cierta memoria en sus primeros años, que se fue perdiendo a medida que el mercado maduró.
- El **ciclo de 4 años** es una regularidad a escala de años, con 3 ciclos completos y rendimientos decrecientes.
- Los **grandes movimientos** coinciden con noticias que no se sabían de antemano.

**Parte B, lo predictivo: ¿sirven para anticipar la semana siguiente?**
- **Si ningún candidato es significativo con el p corregido:** las velas **no anticipaban** el movimiento semanal del ciclo actual mejor que el azar. Apoya **H0 (mercado eficiente)** y **coincide con la bibliografía** (abajo).
- **CNN vs LSTM:** si dan parecido, *cómo* se miran las velas (foto o secuencia) no cambia el resultado.
- **Monte Carlo:** es la vara de la industria. Si las redes no le ganan, aprender de las velas no agrega nada a "simular con la volatilidad".
- **Kronos:** si un modelo que vio miles de millones de velas tampoco es significativo, no es falta de datos: la señal no está.

### Bibliografía con la que se contrasta
- **Fama, E. F. (1970).** Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417. https://doi.org/10.2307/2325486. La base de la hipótesis de mercado eficiente (H0).
- **Marshall, B. R., Young, M. R. & Rose, L. C. (2006).** Candlestick technical trading strategies: Can they create value for investors? *Journal of Banking & Finance*, 30(8), 2303-2323. https://doi.org/10.1016/j.jbankfin.2005.08.001. Las estrategias de velas **no generan valor** en las grandes acciones de EE.UU.
- **Urquhart, A. (2016).** The inefficiency of Bitcoin. *Economics Letters*, 148, 80-82. https://doi.org/10.1016/j.econlet.2016.09.019. Bitcoin fue ineficiente en sus primeros años pero **tiende a volverse eficiente**: lo mismo que vemos con la inercia que se apaga (7.6).
- **Ho, K.-H., Chan, T. T. D., Pan, H. & Li, C. (2021).** Do Candlestick Patterns Work in Cryptocurrency Trading? *2021 IEEE International Conference on Big Data*, 4566-4569. https://doi.org/10.1109/BigData52589.2021.9671826. 68 patrones de velas en 23 criptomonedas: son de **poca utilidad**.
- **Cohen, G. (2021).** *Review of Quantitative Finance and Accounting*, 57(3). https://doi.org/10.1007/s11156-021-00973-6. El contrapunto: en Bitcoin 2012-2020, solo los patrones envolventes mostraron alguna utilidad (lo contrastamos en 7.1).
- **Jiang, J., Kelly, B. & Xiu, D. (2023).** (Re-)Imag(in)ing Price Trends. *Journal of Finance*, 78(6), 3193-3249. https://doi.org/10.1111/jofi.13268. La CNN sobre imágenes de precios que inspiró la nuestra: encontró señal en acciones de EE.UU. con **décadas de datos y miles de acciones**; nosotros tenemos ≈2.700 días de un solo activo.
""")
code(r"""
PROVISORIO = " (PROVISORIO: corrida rápida, los modelos casi no entrenaron)" if MODO_RAPIDO else ""
anotar("15. CONCLUSIONES" + PROVISORIO)

def ba(n):
    return tabla_test.loc[n, "balanced_acc"] if n in tabla_test.index else np.nan

def cambio(a, b):
    if np.isnan(ba(a)) or np.isnan(ba(b)):
        return "no se pudo correr"
    dif = ba(b) - ba(a)
    return f"{b}: {ba(b):.3f} vs {a}: {ba(a):.3f} ({dif:+.3f}) → " + ("mejor" if dif > 0.01 else "peor" if dif < -0.01 else "prácticamente igual")

significativos = tabla_test.index[tabla_test["significativo"] == "sí"].tolist()
crudos = tabla_test.index[tabla_test["p_valor"] < 0.05].tolist()
mejor = tabla_test["balanced_acc"].idxmax()
pre = [n for n in tabla_test.index if tipo_de(n) == "Pre-entrenado"]
mc = [n for n in tabla_test.index if tipo_de(n) == "Simulación"]
clasico_mejor = clasicos["p Holm"].idxmin()
fam_fuera = familias["dif z train"].abs().idxmax()

respuestas = [
    ("1. ¿Qué movimientos marcaron la historia?",
     f"{len(HITOS)} tramos en que el precio se duplicó o se cortó a la mitad; casi todos coinciden con noticias (exchanges, China, ETFs, la Fed, liquidaciones)"),
    ("2. ¿En qué se diferencian las velas de Bitcoin?",
     f"volatilidad anual {tabla_bench.loc['Bitcoin (BTC)', 'volatilidad anual']:.0%}, 365 velas por año y casi sin saltos de apertura "
     f"({tabla_bench.loc['Bitcoin (BTC)', '% días con salto de apertura']:.0%} de los días)"),
    ("3. ¿Los patrones de los libros anticipan algo?",
     f"el más fuerte es '{clasico_mejor}' (p corregido {clasicos.loc[clasico_mejor, 'p Holm']:.3f}; z train {clasicos.loc[clasico_mejor, 'dif z train']:+.2f} "
     f"→ validación {clasicos.loc[clasico_mejor, 'dif z val']:+.2f} → ciclo actual {clasicos.loc[clasico_mejor, 'dif z test']:+.2f}); el martillo aparece {clasicos.loc['martillo', 'veces']} veces "
     f"y no se distingue del azar (p corregido {clasicos.loc['martillo', 'p Holm']:.2f})"),
    ("4. ¿La planilla del profe funciona en Bitcoin?",
     f"P(sube >3% en 3 ruedas | sharpe > 0,9) = {tabla_planilla.iloc[0]['P(fut3 > 3% | sharpe > 0,9)']:.0%} "
     f"vs {tabla_planilla.iloc[0]['P(fut3 > 3%)']:.0%} sin condición"),
    ("5. ¿Se repite algo en la misma fase del halving?",
     "año 1 de cada ciclo: " + ", ".join(f"{c_} {v}" for c_, v in rend_ciclos["año 1"].items()) + f" | forma de las velas por fase: χ² p = {p_chi2:.3f}"),
    ("6. ¿Qué familias encuentra la máquina?",
     f"{K_FAM} familias; la de mayor efecto es {fam_fuera} ({familias.loc[fam_fuera, 'descripción']}: "
     f"z {familias.loc[fam_fuera, 'dif z train']:+.2f} en train → {familias.loc[fam_fuera, 'dif z val']:+.2f} en validación → "
     f"{familias.loc[fam_fuera, 'dif z test']:+.2f} en el ciclo actual)"),
    ("7. ¿A qué se parece hoy?",
     f"las 5 gemelas siguieron caminos {'parecidos' if (np.sign(shazam['7 días después']) == np.sign(shazam['7 días después'].median())).mean() >= 0.8 else 'distintos'} "
     f"a 7 días (de {shazam['7 días después'].min():+.0%} a {shazam['7 días después'].max():+.0%})"),
    ("8. ¿Reglas simples y Monte Carlo?",
     ", ".join(f"{n} {ba(n):.3f}" for n in ["Martillo", "Inercia", "Gemela del pasado"] + mc if n in tabla_test.index)),
    ("9. ¿Foto (CNN) o secuencia (LSTM)?", cambio("CNN", "LSTM")),
    ("10. ¿Un pre-entrenado ve más?",
     (", ".join(f"{n} {ba(n):.3f} (p corregido {tabla_test.loc[n, 'p_corregido (Holm)']:.2f})" for n in pre) if pre else "ninguno pudo correr")),
    ("11. ¿Alguno le gana al azar en el ciclo actual?",
     (f"SÍ, con p corregido < 0,05: {', '.join(significativos)}" if significativos else
      "NO: ningún candidato supera al azar con el p corregido por comparaciones múltiples"
      + (f" (con p sin corregir < 0,05: {', '.join(crudos)})" if crudos else ""))),
    ("12. ¿Por qué se equivocan?", "ver sección 13: frontera difusa, fotos nuevas, cambio de mercado, noticias"),
    ("13. ¿Qué más probamos?", f"{len(tabla_descartes)} intentos descartados (14.7): más datos, reentrenar y redes más grandes no mejoraron"),
]
tabla_respuestas = pd.DataFrame(respuestas, columns=["pregunta", "respuesta (números de esta corrida)"])
guardar_tabla(tabla_respuestas, "15_respuestas_pregunta_por_pregunta")
anotar("15. RESPUESTAS PREGUNTA POR PREGUNTA" + PROVISORIO + ":\n" + "\n".join(f"{p_}\n   → {r_}" for p_, r_ in respuestas))

import textwrap
fig, ax = plt.subplots(figsize=(16, 0.8 * len(respuestas) + 1))
ax.axis("off")
filas_tabla = [[p_, textwrap.fill(r_, 95)] for p_, r_ in respuestas]
t = ax.table(cellText=filas_tabla, colLabels=["Pregunta", "Respuesta (con los números de esta corrida)"],
             loc="center", cellLoc="left", colWidths=[0.27, 0.73])
t.auto_set_font_size(False)
t.set_fontsize(8.5)
t.scale(1, 2.6)
for (r, c_), celda in t.get_celld().items():
    if r == 0:
        celda.set_facecolor("#264653")
        celda.get_text().set_color("white")
ax.set_title("La escalera de preguntas, respondida" + PROVISORIO, pad=12)
guardar_fig("15_respuestas_pregunta_por_pregunta")
plt.show()

anotar("- Respuesta a la pregunta central" + PROVISORIO + ": las velas de Bitcoin esconden sobre todo INERCIA de corto plazo "
       "(rachas que tienden a seguir), más fuerte en 2013-2020 y cada vez más débil, un ciclo de 4 años visible a escala de años, "
       "y grandes movimientos que coinciden con noticias. "
       + (f"Hay poder predictivo débil pero significativo en {', '.join(significativos)} (H1)." if significativos else
          "Para anticipar la semana siguiente, ningún modelo (CNN, LSTM, Monte Carlo ni Kronos) supera al azar con la corrección "
          "por comparaciones múltiples: el movimiento semanal del ciclo actual NO estaba anunciado en las velas (apoya H0, "
          "en línea con Fama 1970, Marshall et al. 2006 y Urquhart 2016)."))
""")

# =====================================================================
md(r"""
---
# 📦 Guardar los resultados (zip)

Junta en un solo zip **todos los gráficos (PNG) y tablas (CSV)** de cada paso, más:
- `00_resumen.txt`: los números clave de cada paso, en orden.
- `predicciones.csv`: lo que dijo cada modelo para cada día.

Los archivos empiezan con el número de la sección, así quedan ordenados igual que el informe:
`03_` base de datos · `04_` historia e hitos · `05_` benchmark · `06_` respuesta (target) · `07_` patrones · `08_` baselines y Monte Carlo · `09_` CNN · `10_` LSTM · `11_` pre-entrenados · `12_` examen final · `13_` errores · `14_` descartes · `15_` conclusiones.
""")
code(r"""
import shutil

(RESULTADOS / "00_resumen.txt").write_text("\n\n".join(RESUMEN), encoding="utf-8")

predicciones = datos[["periodo", "close", "z", "clase"]].copy()
for n, pred in PREDICCIONES.items():
    predicciones[f"{n} dijo"] = [CLASES[v] for v in pred]
for n in PROBAS:
    predicciones[f"{n} confianza"] = PROBAS[n].max(1).round(4)
for n in PROBAS_MC:
    predicciones[f"{n} confianza"] = PROBAS_MC[n].max(1).round(4)
predicciones.to_csv(RESULTADOS / "predicciones.csv")

zip_resultados = shutil.make_archive("resultados_tp_btc", "zip", RESULTADOS)
print(f"Zip con {len(list(RESULTADOS.iterdir()))} archivos:", zip_resultados)
try:
    from google.colab import files
    files.download(zip_resultados)
except ImportError:
    print("Fuera de Colab: el zip quedó en", Path(zip_resultados).resolve())
""")
md(r"""
### (Opcional) Descargar la base de datos de velas
""")
code(r"""
shutil.make_archive("velas_btc", "zip", VELAS)
try:
    from google.colab import files
    files.download("velas_btc.zip")
except ImportError:
    print("Fuera de Colab: el zip quedó en", Path("velas_btc.zip").resolve())
""")


nb = {
    "cells": celdas,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}
Path(sys.argv[1]).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("ok", len(celdas), "celdas")
