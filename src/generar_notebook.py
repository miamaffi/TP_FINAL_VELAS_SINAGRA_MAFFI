"""Genera TP_Velas_Bitcoin.ipynb (notebook único para Colab)."""
import json, sys
from pathlib import Path

celdas = []
def md(txt): celdas.append({"cell_type": "markdown", "metadata": {}, "source": txt.strip("\n")})
def code(txt): celdas.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": txt.strip("\n")})

# =====================================================================
md(r"""
# 🕯️ Velas japonesas y Deep Learning sobre Bitcoin
### ¿Se repiten los patrones? ¿Importa el ciclo del halving?

**Autora:** Mia Maffi

Este notebook contiene **todo el trabajo en un solo lugar**: desde la descarga de los datos crudos hasta los modelos. Cada sección explica **qué** hace el código y **por qué**.

| # | Sección | Estado |
|---|---|---|
| 0 | Contexto: qué es Bitcoin, oferta finita y halving | ✅ |
| 1 | Pregunta de investigación | 📝 borrador |
| 2 | Preparación del entorno | ✅ |
| 3 | Construcción de la base de datos de velas (descarga → limpieza → velas → validación) | ✅ |
| 4 | Exploración visual: los ciclos de halving | ✅ |
| 5 | Features: anatomía de cada vela + fase del halving | ⏳ |
| 6 | Target: ¿sube, baja o lateral? | ⏳ |
| 7 | Dataset para los modelos: ventanas, imágenes y secuencias | ⏳ |
| 8 | Baselines | ⏳ |
| 9 | CNN (imágenes) y LSTM (secuencias), con y sin halving | ⏳ |
| 10 | Ciclos superpuestos y "Shazam" de patrones | ⏳ |
| 11 | Resultados y conclusiones | ⏳ |

> **Cómo usarlo:** en Colab, `Entorno de ejecución → Ejecutar todas`. La descarga tarda ~2 minutos.
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

### Pregunta principal
> **¿El precio que hoy tiene Bitcoin ya estaba anunciado en sus velas japonesas?**
>
> Dicho de forma comprobable: si entrenamos una red **solo con los ciclos anteriores** (2013 → halving de abril de 2024) y le mostramos el ciclo actual **día por día, como si lo estuviera viviendo**, ¿habría anticipado si la **semana siguiente** el precio iba a **subir, bajar o quedar lateral**, mejor que el azar y que las reglas clásicas del análisis técnico?

**¿Por qué así y no "cuánto va a valer Bitcoin"?** Predecir un precio futuro no se puede comprobar hoy. En cambio, el ciclo actual (abr-2024 → hoy) **ya pasó**: sabemos qué ocurrió cada semana, así que cada predicción del modelo se puede marcar como **acierto o error**.

### Cómo lo respondemos
- **Datos:** velas **diarias** de Bitcoin (≈5.000 velas desde 2013). Las semanales se usan solo para visualizar: son muy pocas para entrenar.
- **Dos redes que miran las mismas velas diarias de dos maneras:**
  1. **CNN:** mira la **foto** de las velas, como lo haría un trader.
  2. **LSTM (red recurrente):** **lee** las velas una por una, en orden y con memoria.

### Sub-preguntas
1. ¿La CNN o la LSTM le ganan al azar (33%) y a las reglas clásicas (martillo, etc.) en el ciclo actual?
2. ¿Hasta qué horizonte llega esa capacidad (1, 3, 7 o 14 días)?
3. ¿Mejora si además el modelo sabe en qué fase del ciclo de halving estamos?

### Hipótesis
- **H0 (mercado eficiente):** el precio ya incorpora toda la información pasada, así que ningún modelo supera al azar de forma consistente: el precio de hoy **no** estaba en las velas.
- **H1:** las velas tienen un poder predictivo débil pero real.
- **H2:** ese poder cambia según la fase del ciclo de halving.

### 🪜 Las preguntas que vamos respondiendo, paso a paso
La pregunta principal se responde subiendo una escalera de preguntas más chicas. Cada una tiene **su experimento** y **su respuesta con números** (resumidas todas juntas en la sección 13):

| # | Pregunta | Cómo la respondemos | Sección |
|---|---|---|---|
| 1 | ¿Las reglas simples (martillo, "la tendencia sigue") ya anticipan algo? | baselines contra el azar | 6 |
| 2 | ¿**Cuántas velas** conviene mirar? | misma red con ventanas de 5, 10, 20, 30 y 60 velas | 7.4 y 8.4 |
| 3 | ¿Conviene mirar **velas diarias individuales o velas semanales**? | el mismo mes visto como 28 velas diarias o 4 semanales | 7.4b |
| 4 | ¿Es mejor **mirar la foto** (CNN) o **leer la secuencia** (LSTM)? | las dos redes, mismas reglas, mismos días | 7, 8 y 11 |
| 5 | ¿Y **combinando** las dos ideas (red híbrida)? | CNN-LSTM, tomada de otro proyecto | 9.3 |
| 6 | ¿Ayuda tener **más datos de otros mercados** (acciones de EE.UU.)? | entrenar con 30 acciones + Bitcoin | 9.1 |
| 7 | ¿Conviene agregar **otras criptomonedas**? | entrenar con ETH, LTC y XRP + Bitcoin | 9.1b |
| 8 | ¿Ayuda **reentrenar** a medida que pasa el tiempo? | reentrenar cada 3 meses | 9.2 |
| 9 | ¿Un **modelo pre-entrenado** gigante ve lo que nuestras redes no ven? | Kronos (velas), Chronos-2 (series), ResNet-18 (imágenes) | 10 |
| 10 | **¿Alguno le gana al azar en el ciclo actual?** (la pregunta principal) | examen final con p-valor | 11 |
| 11 | ¿**Por qué** se equivocan? | análisis de errores | 12 |
""")

# =====================================================================
md(r"""
---
# 2. Preparación del entorno

Instalamos y cargamos las librerías:
- **requests**: hace los pedidos a la API de Bitstamp (descarga de datos).
- **pandas**: maneja las tablas de velas (el "Excel" de Python).
- **matplotlib** y **mplfinance**: gráficos; mplfinance dibuja velas japonesas.

Colab trae todo menos `mplfinance`, así que lo instalamos.
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
| **4 horas** | ~33.000 | **Complemento para entrenar:** 6 veces más ejemplos para las redes. Además permite comparar si los patrones funcionan igual en distintas escalas. |
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
# 4. Exploración visual: los ciclos de halving

## 4.1 Panorama completo (velas semanales, escala logarítmica)

Usamos **escala logarítmica** porque el precio pasó de ~$10 a más de $100.000. En escala log, una misma distancia vertical representa el mismo **porcentaje** de cambio, así que podemos comparar 2013 con 2024.
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
guardar_fig("04_panorama_halvings")
plt.show()
""")
md(r"""
**Qué se ve:** después de cada halving vienen una **suba fuerte** (12 a 18 meses después), un **pico** (dic-2013, dic-2017, nov-2021, 2025), una **caída profunda** y una etapa de **acumulación** antes del siguiente halving. Es la "recurrencia" a gran escala que vamos a investigar.

## 4.2 Las velas diarias alrededor de cada halving

60 días antes y 120 días después de cada halving (línea punteada).
""")
code(r"""
estilo = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":")
d = btc["1d"]
for fecha, nombre in zip(HALVINGS, NOMBRES_HALVING):
    tramo = d.loc[fecha - pd.Timedelta(days=60): fecha + pd.Timedelta(days=120)].copy()
    tramo.index = tramo.index.tz_localize(None)                     # mplfinance no acepta zona horaria
    fig, _ = mpf.plot(tramo[["open", "high", "low", "close", "volume"]], type="candle", style=estilo, volume=True,
                      title=f"\nBTC/USD velas diarias – halving {nombre} ({fecha:%d-%m-%Y})",
                      vlines=dict(vlines=[fecha.tz_localize(None)], colors="#444", linestyle="--"),
                      figsize=(13, 6), returnfig=True)
    guardar_fig(f"04_halving_{fecha:%Y}", fig)
    plt.show()
""")

# =====================================================================
md(r"""
---
# 5. Preparación del experimento: "¿el precio de hoy ya estaba en las velas?"

### La idea
Nos paramos en el **día del halving de abril de 2024** y le **escondemos al modelo todo lo que pasó después**. Lo entrenamos solo con el pasado y después le mostramos el ciclo actual día por día. Como ese período **ya ocurrió**, cada predicción se puede corregir: ¿acertó o no?

```
 2013 ─────────── may-2020 │ may-2020 ─────── abr-2024 │ abr-2024 ─────── hoy
      ENTRENAMIENTO        │      VALIDACIÓN           │   TEST: ciclo actual
  (el modelo aprende acá)  │ (ajustamos el modelo)     │ (¿lo habría anticipado?)
     ciclos 2016 y 2020    │  ciclo del halving 2020   │  ciclo del halving 2024
```

Para **cada día** de la historia armamos una pregunta de examen:
- **Lo que el modelo ve:** las **últimas N velas diarias** hasta ese día (cuántas, lo decidimos con un experimento en 7.4 y 8.4).
- **Lo que tiene que adivinar:** qué pasó en los **7 días siguientes**: ¿**sube**, **baja** o **lateral**?

En esta sección armamos: (5.1) la anatomía de cada vela, (5.2) la respuesta correcta de cada día y (5.3) la división en entrenamiento / validación / test.
""")
md(r"""
## 5.1 Anatomía de cada vela (como la planilla del profesor)

Reproducimos las columnas de la planilla del profe (LLY) para cada vela diaria:

| Columna | Fórmula | Qué mide |
|---|---|---|
| `cuerpo` | \|close − open\| (mínimo 0,1% del precio) | la fuerza del día |
| `direccion` | +1 si cerró arriba (verde), −1 si cerró abajo (roja) | quién ganó: compradores o vendedores |
| `mecha_sup` | high − max(open, close) | cuánto subió y fue rechazado |
| `mecha_inf` | min(open, close) − low | cuánto bajó y fue rechazado |
| `prop_*` | cada parte / (high − low) | la forma de la vela, sin importar el precio |
| `martillo` | mecha inferior > 2 cuerpos y mecha superior < 0,2 cuerpos | el patrón clásico (lo usamos como baseline) |
| `ret_1d` | log(close hoy / close ayer) | el retorno del día |

**¿Por qué el cuerpo mínimo es 0,1% del precio y no 0,005 como en la planilla?** Porque en la planilla el mínimo está en dólares y LLY vale unos cientos; Bitcoin pasó de $13 a $126.000, así que el mínimo tiene que ser **proporcional al precio**.

Las proporciones son importantes para las redes: una vela de 2013 ($100) y una de 2025 ($100.000) pueden tener **la misma forma**, y eso es lo que queremos que el modelo compare.
""")
code(r"""
HORIZONTE = 7      # días hacia adelante que queremos anticipar
VENTANA_VOL = 30   # días que usamos para medir la volatilidad reciente

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
## 5.2 La respuesta correcta de cada día: ¿sube, baja o lateral?

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
## 5.3 División por ciclos: entrenamiento, validación y test

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
INICIO = pd.Timestamp("2013-01-01", tz="UTC")
CORTE_VAL, CORTE_TEST = HALVINGS[2], HALVINGS[3]     # halvings de may-2020 y abr-2024
EMBARGO = pd.Timedelta(days=HORIZONTE)

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

anotar(f"5. Umbral k = {k:.3f}  →  'lateral' si el movimiento de 7 días es menor a {k:.2f} volatilidades")
anotar(f"5. Días con respuesta: {len(datos):,}  (último: {datos.index.max():%Y-%m-%d})")
tabla = pd.crosstab(datos["periodo"], datos["clase"], margins=True, margins_name="total")
tabla = tabla.loc[["train", "val", "test", "total"], CLASES + ["total"]]
guardar_tabla(tabla, "05_clases_por_periodo")
anotar("5. Cantidad de días por período y clase:\n" + tabla.to_string())
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
anotar(f"5. Ejemplo {ej_fecha:%d-%m-%Y}: cierre ${ej['close']:,.0f} → 7 días después ${precio_despues:,.0f} "
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
guardar_fig("05_respuesta_de_cada_dia")
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
# 6. Baselines: la vara que hay que superar

Antes de entrenar redes, necesitamos saber **cuánto acierta alguien que no usa deep learning**. Si una red no le gana a esto, no sirve.

| Baseline | Regla | Qué representa |
|---|---|---|
| **Azar** | elige sube/baja/lateral al azar | tirar un dado de 3 caras |
| **Siempre "sube"** | la clase más común en entrenamiento | "Bitcoin siempre sube" |
| **Martillo** | si hoy hay martillo después de una semana bajista → "sube"; si no → "lateral" | el análisis técnico de libro (el detector de la planilla del profe) |
| **Inercia** | si la semana pasada subió fuerte → "sube"; si bajó fuerte → "baja"; si no → "lateral" | "la tendencia sigue" |
| **Gemela del pasado** *(sección 7.5)* | busca la foto de velas más parecida del entrenamiento y repite lo que pasó después | "este patrón se parece a..." (como el ejemplo de LLY del profe) |

### ¿Cómo medimos? *Balanced accuracy*
Es el **promedio del acierto dentro de cada clase**: (acierto en los días que subió + acierto en los que bajó + acierto en los laterales) / 3.
- El azar da ≈ **0,33**.
- "Siempre sube" acierta el 100% de los días de suba y 0% del resto: (1 + 0 + 0)/3 = **0,33**. La *accuracy* común lo premiaría si hubo muchas subas; la balanced no se deja engañar.

### ⚙️ Configuración de la corrida
`MODO_RAPIDO = True` corre **todo el notebook en pocos minutos**, con muy poco entrenamiento: sirve **solo para comprobar que no haya errores**, y sus resultados **no valen**. Para el informe: `MODO_RAPIDO = False` y **GPU activada** (Entorno de ejecución → Cambiar tipo → T4 GPU). La corrida completa puede tardar cerca de una hora.
""")
code(r"""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score

# ---------- Configuración de la corrida ----------
MODO_RAPIDO = False      # True = prueba de pocos minutos para ver que no haya errores (resultados NO válidos)

VENTANAS = [5, 10] if MODO_RAPIDO else [5, 10, 20, 30, 60]   # ventanas a probar (7.4 y 8.4)
SEMILLAS = [0] if MODO_RAPIDO else [0, 1, 2, 3, 4]           # repeticiones por ventana
N_REDES_FINAL = 1 if MODO_RAPIDO else 5                       # redes que se promedian en cada modelo final
N_REDES_MEJORAS = 1 if MODO_RAPIDO else 3                     # redes por modelo en las mejoras (sección 9)
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

anotar(f"6. Días con martillo después de una semana bajista: {(PREDICCIONES['Martillo'] == ID['sube']).sum()} de {len(datos)}")
tabla_baselines = pd.DataFrame({"balanced_acc_validacion": {n: bal_acc(p, "val") for n, p in PREDICCIONES.items()}}).round(3)
guardar_tabla(tabla_baselines, "06_baselines_validacion")
anotar("6. Baselines en VALIDACIÓN (el test se mira recién en la sección 11):\n" + tabla_baselines.to_string())
tabla_baselines
""")

# =====================================================================
md(r"""
---
# 7. CNN: una red que aprende mirando la FOTO de las velas

## 7.1 ¿Por qué una CNN?
- **La consigna es reconocer patrones mirando las velas**, y un trader no lee una tabla de números: **mira el gráfico**. Una CNN (red convolucional) es el tipo de red diseñado para **mirar imágenes**: es la que se usa para reconocer caras, objetos o radiografías.
- **Su herramienta, la convolución, busca formas locales en cualquier parte de la imagen.** Un martillo es un martillo esté al principio o al final de la ventana: la CNN usa **el mismo detector** en todas las posiciones.
- **Hay evidencia de que funciona en finanzas:** Jiang, Kelly & Xiu (*Journal of Finance*, 2023) entrenaron una CNN con imágenes de gráficos de acciones de EE.UU. y encontraron poder predictivo. Nosotros **replicamos esa idea sobre Bitcoin**.
- **Es el complemento de la red recurrente** (sección 8): la CNN mira **toda la foto de golpe**; la LSTM **lee las velas una por una**. Comparar las dos responde si importa *cómo* se miran las velas.

El camino de la CNN tiene 4 pasos: **la foto** (7.2) → **la arquitectura** (7.3) → **cuántas velas mirar** (7.4) → **el modelo final** (7.5).
""")
md(r"""
## 7.2 La foto: cómo convertimos las velas en una imagen

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

ALTO_PRECIO, SEPARACION, ALTO_VOL = 50, 2, 12
ALTO = ALTO_PRECIO + SEPARACION + ALTO_VOL       # 64 píxeles

OHLCV = d[["open", "high", "low", "close", "volume"]].to_numpy()
POS = d.index.get_indexer(datos.index)           # posición de cada día del dataset dentro de la tabla completa

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
guardar_fig("07_ejemplo_grafico_normal", fig)
plt.show()

plt.figure(figsize=(8, 3))
plt.imshow(dibujar_ventana(OHLCV[ej_pos - 29:ej_pos + 1]), cmap="gray", aspect="auto", interpolation="nearest")
plt.title("La misma ventana como la ve la CNN (64 × 90 píxeles)")
plt.axis("off")
guardar_fig("07_ejemplo_foto_cnn")
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
guardar_fig("07_ejemplos_fotos_por_clase")
plt.show()
del X_ejemplo
""")
md(r"""
## 7.3 La arquitectura de la CNN: qué decidimos y por qué

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
| **Grilla final de 4×4** (*adaptive pooling*) | en vez de aplanar todo | Permite usar **la misma red con cualquier ventana** (5 o 60 velas): así la prueba de ventana (7.4) compara solo la ventana, no arquitecturas distintas. Y conserva una idea de **dónde** está cada forma (las últimas velas pueden pesar más). |
| **Dropout 50%** | antes de la capa final | Obliga a la red a no depender de un solo detector: es la defensa principal contra **memorizar**. |
| **3 salidas + softmax** | una por clase | La red da una **probabilidad** para cada clase; elegimos la más alta. La probabilidad la usamos después para ver si "sabe cuándo no sabe" (12.2). |

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

describir_red(CNNVelas(), (1, 1, ALTO, 3 * 30), "07_arquitectura_cnn")
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
## 7.4 Experimento: ¿cuántas velas necesita ver la CNN?

No adivinamos el tamaño de la ventana: **lo medimos**. Entrenamos la misma CNN con ventanas de **5, 10, 20, 30 y 60 velas** y comparamos en **validación**.

Cuidados para que la comparación sea justa:
1. **Mismos días en todas las ventanas:** solo cambia cuánto pasado ve la red, no qué días se evalúan.
2. **5 repeticiones por ventana:** cada entrenamiento arranca con pesos al azar, así que da resultados algo distintos. Mostramos **cada repetición** (puntos grises) y el **promedio ± desvío**: diferencias más chicas que las barras **no son diferencias reales**.
3. **El test no se toca:** si eligiéramos la ventana mirando el test, ya no sería un examen limpio.

**Regla para elegir (regla de "un desvío"):** tomamos la ventana con mejor promedio; si una ventana **más chica** queda dentro de un desvío de esa, elegimos la más chica. Si dos ventanas empatan, la más simple es preferible: con pocos datos, menos información de entrada = menos riesgo de memorizar.
""")
code(r"""
N_CNN, resumen_ventana_cnn = prueba_de_ventana("CNN", armar_imagenes, CNNVelas, "07_prueba_ventana_cnn")
""")
md(r"""
### 🔎 Cómo leer este gráfico
- **Si todas las ventanas están cerca de 0,33**, ninguna ve nada: la red no encuentra patrones en ninguna escala. Es un resultado válido (apoya H0).
- **Si alguna sobresale por encima de su barra de error**, en esa escala hay algo de información.
- **Si mejora y después cae**, existe una "memoria" útil de las velas: más pasado que eso es ruido.

""")
md(r"""
## 7.4b ¿Velas diarias individuales o velas semanales?
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
guardar_tabla(resumen_dvs.round(4), "07_4b_diarias_vs_semanales")
anotar("7.4b ¿Velas diarias o semanales? (balanced accuracy en validación):\n" + resumen_dvs.round(3).to_string())

tabla_dvs = resumen_dvs["promedio"].unstack()
error_dvs = resumen_dvs["desvio"].unstack()
ax = tabla_dvs.plot.bar(yerr=error_dvs, capsize=4, rot=0, color=["#264653", "#f4a261"], figsize=(9, 4.5))
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylim(0.25, max(0.42, tabla_dvs.values.max() + 0.03))
ax.set_ylabel("Balanced accuracy en validación")
ax.set_xlabel("Mismo período de tiempo, visto de dos maneras")
ax.set_title("¿Conviene mirar cada día por separado o resumir la semana en una vela?")
ax.legend()
guardar_fig("07_4b_diarias_vs_semanales")
plt.show()
""")
md(r"""
## 7.5 La CNN final
Con la ventana elegida entrenamos **5 redes** y **promediamos sus probabilidades**. Después de entrenar mostramos: el error de validación de cada red época por época (dónde frenó el *early stopping*), su acierto, y la matriz de confusión en validación.
""")
code(r"""
X_cnn = armar_imagenes(N_CNN)
print(f"Fotos: {X_cnn.shape[0]:,} de {X_cnn.shape[2]}×{X_cnn.shape[3]} píxeles (ventana de {N_CNN} velas)")
proba, historias = entrenar_varias(CNNVelas, X_cnn[mascara["train"]], y[mascara["train"]],
                                   X_cnn[mascara["val"]], y[mascara["val"]], X_cnn, N_REDES_FINAL)
registrar_red("CNN", proba, historias, "07_entrenamiento_cnn")
""")
md(r"""
### El quinto baseline: la "gemela del pasado"
Para cada día de validación y test buscamos **la foto más parecida de todo el entrenamiento** (la que tiene menos píxeles distintos) y predecimos **lo mismo que pasó después de esa gemela**. Es la idea de "este patrón se parece a otro que ya vi" llevada al extremo, sin red. Guardamos también **cuántos píxeles difieren**: lo usamos en el análisis de errores (12.3) para saber qué fotos son **nuevas** para el modelo.
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
anotar(f"7.5 Gemela del pasado: balanced accuracy en validación = {bal_acc(pred_gemela, 'val'):.3f}")
""")

# =====================================================================
md(r"""
---
# 8. LSTM: una red recurrente que LEE las velas una por una

## 8.1 ¿Por qué una red recurrente (LSTM)?
- **Las velas son una secuencia en el tiempo**: el orden importa (no es lo mismo una vela grande y después una chica que al revés). Una red **recurrente** está diseñada para eso: lee los datos **en orden**, uno por uno, y va guardando una **memoria** de lo que vio.
- **¿Por qué LSTM y no una recurrente simple?** Una recurrente simple se "olvida" rápido de lo que vio al principio. La **LSTM** (*Long Short-Term Memory*) tiene **compuertas** que deciden, en cada vela, **qué recordar, qué olvidar y qué usar**. Es la recurrente estándar para series temporales (una GRU es parecida y más simple; con tan pocos datos rinden casi igual).
- **Mira lo mismo de otra manera:** la CNN ve una **foto** (redondeada a píxeles); la LSTM recibe los **números exactos** de cada vela. Si las dos fallan, no es por cómo miramos las velas; si una funciona y la otra no, *cómo* se miran importa.

## 8.2 La secuencia: qué números recibe por cada vela

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
guardar_fig("08_ejemplo_secuencia_lstm")
plt.show()
""")
md(r"""
## 8.3 La arquitectura de la LSTM: qué decidimos y por qué

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
| **Ventana** | se prueba igual que en la CNN (8.4) | Puede que a la LSTM le sirva una ventana distinta |
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
## 8.4 Experimento: ¿cuántas velas necesita leer la LSTM?
Exactamente la misma prueba que en 7.4: mismas ventanas, mismas repeticiones, misma regla para elegir.
""")
code(r"""
N_LSTM, resumen_ventana_lstm = prueba_de_ventana("LSTM", armar_secuencias, LSTMVelas, "08_prueba_ventana_lstm")
""")
md(r"""
## 8.5 La LSTM final
""")
code(r"""
X_lstm = armar_secuencias(N_LSTM)
print(f"Secuencias: {X_lstm.shape[0]:,} de {N_LSTM} velas × {X_lstm.shape[2]} números")
proba, historias = entrenar_varias(LSTMVelas, X_lstm[mascara["train"]], y[mascara["train"]],
                                   X_lstm[mascara["val"]], y[mascara["val"]], X_lstm, N_REDES_FINAL)
registrar_red("LSTM", proba, historias, "08_entrenamiento_lstm")

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
guardar_fig("08_prueba_ventana_cnn_vs_lstm")
plt.show()
""")

# =====================================================================
md(r"""
---
# 9. Mejoras: tres intentos para que las redes aprendan más

En la primera versión del trabajo vimos dos problemas:
1. **Pocos datos:** solo ≈ 2.700 días de entrenamiento. Una red de imágenes suele necesitar decenas de miles de ejemplos, y la red terminaba contestando casi siempre "lateral" (no se comprometía).
2. **El mercado cambió:** la red aprendía una sola vez con 2013-2020 y la usábamos hasta 2026, pero la volatilidad bajó de ≈65% a ≈42% anual. Lo aprendido se iba "venciendo".

Probamos una mejora para cada problema (9.1 y 9.2), y una tercera tomada de otro proyecto de criptomonedas (9.3). **En las dos, el test sigue siendo el mismo y limpio**: el modelo nunca ve el futuro.

## 9.1 Mejora 1: más datos, con acciones líquidas de EE.UU.
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
anotar(f"9.1 Acciones descargadas: {len(acciones)} de {len(ACCIONES)} ({', '.join(acciones)})")
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
    anotar(f"9.1 Ejemplos de acciones: {len(y_acc):,} (Bitcoin solo tenía {mascara['train'].sum():,}) | "
           f"clases: " + ", ".join(f"{c} {np.mean(y_acc == ID[c]):.0%}" for c in CLASES))

    tr, va = mascara["train"], mascara["val"]
    print("\nCNN + acciones:")
    proba, historias = entrenar_varias(CNNVelas, np.concatenate([X_img_acc, X_cnn[tr]]), np.concatenate([y_acc, y[tr]]),
                                       X_cnn[va], y[va], X_cnn, N_REDES_MEJORAS, tam_lote=256)
    registrar_red("CNN + acciones", proba, historias, "09_1_entrenamiento_cnn_acciones")
    print("\nLSTM + acciones:")
    proba, historias = entrenar_varias(LSTMVelas, np.concatenate([X_seq_acc, X_lstm[tr]]), np.concatenate([y_acc, y[tr]]),
                                       X_lstm[va], y[va], X_lstm, N_REDES_MEJORAS, tam_lote=256)
    registrar_red("LSTM + acciones", proba, historias, "09_1_entrenamiento_lstm_acciones")

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
    guardar_fig("09_1_mejora_mas_datos")
    plt.show()
else:
    print("⚠️ No se pudo bajar ninguna acción (¿problema de conexión con Yahoo?). Se saltea la mejora 1.")
""")
md(r"""
## 9.1b Mejora 1b: ¿conviene agregar otras criptomonedas?
La idea del proyecto que revisamos en 9.3 (Livieris et al., 2021) es que **las criptomonedas se mueven en conjunto**, así que mirar varias ayuda. Probamos sumar al entrenamiento las velas diarias de **Ethereum (ETH), Litecoin (LTC) y Ripple (XRP)**, de la misma fuente (Bitstamp), con el mismo procedimiento que las acciones:
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
        anotar(f"9.1b {par.upper()}: {len(pos):,} días de entrenamiento ({df.index.min():%m-%Y} → {df.index.max():%m-%Y})")
    except Exception as e:
        print(f"{par}: no se pudo usar ({e})")

if y_cri:
    X_img_cri, X_seq_cri, y_cri = np.concatenate(X_img_cri), np.concatenate(X_seq_cri), np.concatenate(y_cri)
    tr, va = mascara["train"], mascara["val"]
    print("\nCNN + criptos:")
    proba, historias = entrenar_varias(CNNVelas, np.concatenate([X_img_cri, X_cnn[tr]]), np.concatenate([y_cri, y[tr]]),
                                       X_cnn[va], y[va], X_cnn, N_REDES_MEJORAS)
    registrar_red("CNN + criptos", proba, historias, "09_1b_entrenamiento_cnn_criptos")
    print("\nLSTM + criptos:")
    proba, historias = entrenar_varias(LSTMVelas, np.concatenate([X_seq_cri, X_lstm[tr]]), np.concatenate([y_cri, y[tr]]),
                                       X_lstm[va], y[va], X_lstm, N_REDES_MEJORAS)
    registrar_red("LSTM + criptos", proba, historias, "09_1b_entrenamiento_lstm_criptos")

    variantes = [("solo Bitcoin", ""), ("+ acciones", " + acciones"), ("+ criptos", " + criptos")]
    comparacion = pd.DataFrame({etiqueta: [bal_acc(PREDICCIONES[r + suf], "val") if (r + suf) in PREDICCIONES else np.nan
                                           for r in ["CNN", "LSTM"]] for etiqueta, suf in variantes}, index=["CNN", "LSTM"])
    guardar_tabla(comparacion.round(4), "09_1b_mas_datos_comparacion")
    anotar("9.1b ¿Qué datos extra ayudan? (balanced accuracy en validación):\n" + comparacion.round(3).to_string())
    ax = comparacion.plot.bar(rot=0, color=["#bbbbbb", "#2a9d55", "#f4a261"], figsize=(9, 4.5))
    ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ax.set_ylim(0.25, max(0.45, np.nanmax(comparacion.values) + 0.02))
    ax.set_title("¿Con qué datos extra aprende mejor? (validación, Bitcoin)")
    ax.legend()
    guardar_fig("09_1b_mas_datos_comparacion")
    plt.show()
else:
    print("⚠️ No se pudo bajar ninguna cripto; se saltea la mejora 1b.")
""")
md(r"""
## 9.2 Mejora 2: reentrenar cada 3 meses (*walk-forward*)
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
filas = [("modelo original (7 y 8)", [(INICIO, CORTE_VAL - EMBARGO, "#bbbbbb"), (CORTE_VAL, CORTE_TEST - EMBARGO, "#f4a261"),
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
guardar_fig("09_2_esquema_reentrenamiento")
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
anotar(f"9.2 Reentrenamiento: {len(BLOQUES)} bloques de 3 meses (el resultado se ve en el test, sección 11)")
""")

# =====================================================================
md(r"""
## 9.3 Mejora 3: una red híbrida CNN-LSTM (idea de otro proyecto)

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
| **Varias monedas** | BTC + ETH + XRP **como entradas simultáneas** | otras criptos y acciones **como ejemplos extra** de entrenamiento (9.1 y 9.1b) | ETH y XRP cotizan en Bitstamp recién desde 2016-2017: usarlas como entradas simultáneas nos obligaría a **tirar 2013-2017 de Bitcoin** (la mitad del entrenamiento). Como ejemplos extra, suman sin quitar nada |

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

anotar(f"9.3 Arquitectura CNN-LSTM híbrida: {sum(p.numel() for p in CNNLSTMVelas().parameters()):,} parámetros")
proba, historias = entrenar_varias(CNNLSTMVelas, X_lstm[mascara["train"]], y[mascara["train"]],
                                   X_lstm[mascara["val"]], y[mascara["val"]], X_lstm, N_REDES_FINAL)
registrar_red("CNN-LSTM híbrida", proba, historias, "09_3_entrenamiento_cnn_lstm")
""")

md(r"""
---
# 10. Rivales pre-entrenados: ¿un modelo gigante ve lo que nuestras redes no ven?

## 10.1 Qué es un modelo pre-entrenado
Nuestras redes aprendieron **desde cero**, solo con Bitcoin (≈ 2.700 días). Un **modelo pre-entrenado** (o *fundacional*) es al revés: otra gente lo entrenó con **cantidades enormes de series de tiempo** y lo publicó para que cualquiera lo use. Lo usamos **tal cual, sin entrenarlo con nuestros datos** (*zero-shot*): le damos las velas hasta un día y nos dice qué espera para los próximos 7.

Es como comparar a un estudiante que estudió solo los ciclos de Bitcoin (nuestras redes) con un experto que vio millones de gráficos de todo tipo de mercados.

| | **Kronos** | **Chronos-2** | Nuestras redes |
|---|---|---|---|
| Quién | Shi et al., aceptado en **AAAI 2026** | **Amazon**, 2025 | nosotros |
| Con qué aprendió | **más de 12.000 millones de velas** (*K-lines*) de 45 mercados del mundo | millones de series de tiempo de todo tipo (energía, clima, ventas, finanzas...) | ≈ 2.700 días de Bitcoin |
| Qué mira | **velas completas** (apertura, máximo, mínimo, cierre, volumen) | una serie de números (usamos el cierre) | velas completas |
| Tamaño | Kronos-small: ≈ 25 millones de parámetros | ≈ 120 millones de parámetros | ≈ 5 a 30 mil parámetros |
| Contexto que usamos | las últimas 400 velas | las últimas 400 velas | 5 a 60 velas |

Y un tercer pre-entrenado, de otro tipo: **ResNet-18**, una red de **imágenes** entrenada con **ImageNet** (1,2 millones de fotos de perros, autos, objetos...). La usamos para mirar **nuestras fotos de velas** (10.4).

**Kronos es el más relevante para este trabajo: es un modelo de velas japonesas.** Si los patrones de velas tienen información, un modelo que vio miles de millones de velas debería encontrarla.

### Para qué nos sirven
1. **Como vara más alta:** si ni un modelo que vio miles de millones de velas anticipa la semana siguiente, la conclusión "no estaba en las velas" se vuelve mucho más fuerte.
2. **Para saber si el problema son los datos:** si el pre-entrenado le gana a nuestras redes, la diferencia es la cantidad de datos con la que aprendió (y eso conecta con la mejora 9.1).
3. **Para combinar:** ¿cuando nuestra red y el pre-entrenado **coinciden**, aciertan más? (10.5)

### Un modelo que evaluamos y descartamos: *candlefusion*
En HuggingFace hay un modelo llamado `tuankg1028/candlefusion` que mira imágenes de velas con un *Vision Transformer*. Lo revisamos y **no lo usamos** porque:
- además de la imagen, **necesita texto de noticias** como entrada (usa BERT), y nosotros no tenemos noticias;
- su ficha **no informa ninguna métrica de acierto** (solo un error de validación de 316.165, en precio);
- **no dice con qué fechas se entrenó** (con un dataset de BTC que podría incluir nuestro período de test);
- no tiene paper ni evaluación independiente.

Elegir rivales **documentados y evaluados** (Kronos: paper aceptado en AAAI 2026; Chronos-2: Amazon; ResNet: el estándar de visión) es parte de hacer una comparación seria.

### Cómo convertimos su predicción en sube / baja / lateral
Estos modelos no contestan "sube/baja/lateral": predicen **el precio de los próximos 7 días**. Con el precio predicho a 7 días calculamos el movimiento esperado en volatilidades (*z*) y aplicamos **exactamente la misma regla y el mismo umbral k** que usamos para armar la respuesta (sección 5). Así se los evalúa con las mismas reglas que a nuestras redes.

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
    anotar(f"10. {nombre}: balanced accuracy en el test = {balanced_accuracy_score(y[dias], pred[dias]):.3f} ({len(dias)} días)")

anotar(f"10. Pre-entrenados: {len(DIAS_PRE)} días del test, contexto de {CONTEXTO_PRE} velas; "
       f"examen limpio desde {FECHA_LIMPIA:%d-%m-%Y} ({(datos.index[DIAS_PRE] >= FECHA_LIMPIA).sum()} días)")
""")
md(r"""
## 10.2 Kronos: el modelo fundacional de velas
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
## 10.3 Chronos-2: el modelo fundacional general de Amazon
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
## 10.4 ResNet-18: una red de imágenes pre-entrenada (*transfer learning*)
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
    registrar_red("ResNet-18 (pre-entrenado en ImageNet)", proba, historias, "10_4_entrenamiento_resnet")
except Exception as e:
    print("⚠️ ResNet-18 no pudo correr, se sigue sin ella:", repr(e))
""")
md(r"""
## 10.5 Nuestras redes vs los pre-entrenados
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
    guardar_tabla(vs_pre.round(3), "10_5_redes_vs_preentrenados")
    anotar("10.5 Nuestras redes vs pre-entrenados (balanced accuracy):\n" + vs_pre.round(3).to_string())

    filas = []
    for pre in PREENTRENADOS:
        for red in ["CNN", "LSTM"]:
            coinciden = PREDICCIONES[red][DIAS_PRE] == PREDICCIONES[pre][DIAS_PRE]
            acierto = PREDICCIONES[red][DIAS_PRE] == y[DIAS_PRE]
            filas.append({"red": red, "pre-entrenado": pre, "% días que coinciden": coinciden.mean(),
                          "acierto cuando coinciden": acierto[coinciden].mean() if coinciden.any() else np.nan,
                          "acierto cuando NO coinciden": acierto[~coinciden].mean() if (~coinciden).any() else np.nan})
    acuerdo = pd.DataFrame(filas)
    guardar_tabla(acuerdo.round(3), "10_5_cuando_coinciden")
    anotar("10.5 ¿Aciertan más cuando coinciden?\n" + acuerdo.round(3).to_string(index=False))

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
    guardar_fig("10_5_redes_vs_preentrenados")
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
    guardar_fig("10_5_prediccion_vs_real")
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
# 11. El examen final (test): ¿el precio de hoy estaba en las velas?

## 11.1 Qué es el test y por qué es justo

Todo lo anterior fue **preparación**. Ahora tomamos los días del **ciclo actual** (desde el halving del 20-abr-2024 hasta hoy) y hacemos esto, **día por día**:

1. Le mostramos a cada modelo **solo las velas hasta ese día** (la foto o la secuencia).
2. El modelo contesta: **¿la semana siguiente sube, baja o queda lateral?**
3. Como **esos días ya pasaron, nosotros sabemos la respuesta correcta**. El modelo no: nunca vio estos días, ni durante el entrenamiento ni para elegir la ventana o cuándo parar. (La única duda son los pre-entrenados, que pueden haber visto precios de 2024-2025: por eso en 10.5 medimos también el "examen limpio".)
4. Comparamos su respuesta con lo que pasó: **acierto o error**.

Es como un examen con las respuestas en un sobre cerrado: el modelo estudió con los ciclos anteriores, rinde con el ciclo actual, y recién al final abrimos el sobre. **Ya sabíamos qué iba a pasar, pero lo usamos para corregir, no para enseñar.**

Por eso el test se evalúa **una sola vez**, con todo ya decidido: si después de verlo cambiáramos algo, estaríamos estudiando con las respuestas.
""")
code(r"""
mt = mascara["test"]
yt = y[mt]
idx_test = np.flatnonzero(mt)
fechas_test = datos.index[mt]
anotar(f"11. Test: {mt.sum()} días, del {fechas_test.min():%d-%m-%Y} al {fechas_test.max():%d-%m-%Y} "
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
guardar_fig("11_1_como_se_usaron_los_datos")
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
guardar_tabla(hoja, "11_1_hoja_del_examen")

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
guardar_fig("11_1_hoja_del_examen")
plt.show()
""")
md(r"""
## 11.2 Resultados de todos los modelos

Para cada modelo: **balanced accuracy**, **accuracy**, y un **p-valor** que responde *"¿esto le gana al azar o fue suerte?"*. Se calcula con una **prueba de permutación**: mezclamos las respuestas correctas 1.000 veces y vemos cuántas veces un resultado al azar es igual o mejor. Si el p-valor es **menor a 0,05**, la diferencia con el azar es **significativa** (★ en el gráfico).
""")
code(r"""
def p_valor_permutacion(y_real, pred, repeticiones=1000, semilla=0):
    rng_p = np.random.default_rng(semilla)
    real = balanced_accuracy_score(y_real, pred)
    azar = [balanced_accuracy_score(rng_p.permutation(y_real), pred) for _ in range(repeticiones)]
    return real, np.mean(np.array(azar) >= real)

def tipo_de(nombre):
    if nombre in ["Azar", "Siempre sube", "Martillo", "Inercia", "Gemela del pasado"]:
        return "Baseline"
    if "pre-entrenado" in nombre:
        return "Pre-entrenado"
    return "Mejora" if ("acciones" in nombre or "reentrenada" in nombre or "híbrida" in nombre) else "Red propia"

COLOR_TIPO = {"Baseline": "#bbbbbb", "Red propia": "#264653", "Mejora": "#2a9d55", "Pre-entrenado": "#9b5de5"}

def tipos_de_resultado(pred):
    pt = pred[mt]
    acierto = pt == yt
    grave = ((pt == ID["sube"]) & (yt == ID["baja"])) | ((pt == ID["baja"]) & (yt == ID["sube"]))
    return acierto, ~acierto & ~grave, grave

filas = []
for nombre, pred in PREDICCIONES.items():
    ba, p = p_valor_permutacion(yt, pred[mt])
    acierto, leve, grave = tipos_de_resultado(pred)
    filas.append({"modelo": nombre, "tipo": tipo_de(nombre), "balanced_acc": ba, "accuracy": acierto.mean(), "p_valor": p,
                  "dias_acertados": acierto.sum(), "errores_leves": leve.sum(), "errores_graves": grave.sum(), "dias": len(yt)})
tabla_test = pd.DataFrame(filas).set_index("modelo")
guardar_tabla(tabla_test.round(4), "11_2_resultados_test")
anotar("11.2 Resultados en el TEST:\n" + tabla_test.round(3).to_string())

REDES = [n for n in PREDICCIONES if n in PROBAS]
mejoras = tabla_test[tabla_test["tipo"] == "Mejora"]
PRINCIPALES = ["CNN", "LSTM"] + ([mejoras["balanced_acc"].idxmax()] if len(mejoras) else [])
COLOR_MODELO = dict(zip(PRINCIPALES, ["#264653", "#e76f51", "#2a9d55"]))

orden = tabla_test.sort_values("balanced_acc")
fig, ax = plt.subplots(figsize=(10, 0.45 * len(orden) + 1.5))
ax.barh(orden.index, orden["balanced_acc"], color=[COLOR_TIPO[t] for t in orden["tipo"]])
for i, (nombre, fila) in enumerate(orden.iterrows()):
    ax.text(fila["balanced_acc"] + 0.002, i, f"{fila['balanced_acc']:.3f}" + (" ★" if fila["p_valor"] < 0.05 else ""), va="center", fontsize=9)
ax.axvline(1 / 3, color="#d1495b", linestyle="--")
ax.set_xlim(min(0.28, orden["balanced_acc"].min() - 0.01), orden["balanced_acc"].max() + 0.03)
ax.set_xlabel("Balanced accuracy en el test (ciclo actual)  ·  ★ = le gana al azar (p < 0,05)")
ax.legend(handles=[Patch(color=c, label=t) for t, c in COLOR_TIPO.items()] +
          [plt.Line2D([], [], color="#d1495b", linestyle="--", label="azar (0,33)")], loc="lower right")
ax.set_title("El examen final: ¿algún modelo le gana al azar en el ciclo actual?")
guardar_fig("11_2_resultados_test")
plt.show()
display(tabla_test.round(3))
""")
md(r"""
## 11.3 Aciertos y errores de cada modelo
No todos los errores cuestan lo mismo:
- 🟩 **Acierto.**
- 🟧 **Error leve:** confundir un movimiento con un lateral (dijo "sube" y quedó quieto). Si invertiste, no ganaste, pero tampoco perdiste mucho.
- 🟥 **Error grave:** confundir **sube con baja**. Si invertiste, **perdiste plata**.
""")
code(r"""
tipos = pd.DataFrame({n: [v.mean() for v in tipos_de_resultado(p)] for n, p in PREDICCIONES.items()},
                     index=["acierto", "error leve (con lateral)", "error grave (sube ↔ baja)"]).T
tipos = tipos.loc[tabla_test.sort_values("balanced_acc").index]
guardar_tabla(tipos.round(3), "11_3_aciertos_y_errores")

ax = tipos.plot.barh(stacked=True, color=["#2a9d55", "#f4a261", "#d1495b"], figsize=(12, 0.5 * len(tipos) + 1.5), width=0.75)
for barras in ax.containers:
    ax.bar_label(barras, labels=[f"{v:.0%}" if v > 0.04 else "" for v in barras.datavalues], label_type="center",
                 fontsize=8, color="white")
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("¿Cómo se reparten los días del test? Aciertos, errores leves y errores graves")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
guardar_fig("11_3_aciertos_y_errores")
plt.show()
""")
md(r"""
## 11.4 Matrices de confusión: qué dijo cada red vs qué pasó
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
guardar_fig("11_4_matrices_de_confusion")
plt.show()
""")
md(r"""
## 11.5 Clase por clase
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
guardar_tabla(por_clase.round(3), "11_5_por_clase")

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
guardar_fig("11_5_por_clase")
plt.show()
""")
md(r"""
## 11.6 Día por día: ¿dónde acertó y dónde no?
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
guardar_fig("11_6_dia_por_dia")
plt.show()
""")
md(r"""
## 11.7 ¿Sirvió reentrenar? Acierto por bloque de 3 meses
Comparamos, bloque por bloque, las redes **entrenadas una sola vez** (secciones 7 y 8) con las **reentrenadas cada 3 meses** (9.2).
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
    guardar_tabla(por_bloque.round(3), "11_7_acierto_por_bloque")
    anotar("11.7 Acierto por bloque de 3 meses:\n" + por_bloque.round(3).to_string())
    ax = por_bloque.drop(columns="días").plot.bar(figsize=(14, 4.5), rot=0,
                                                   color=["#264653", "#8ab17d", "#e76f51", "#f4a261"])
    ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
    ax.set_ylabel("acierto")
    ax.set_title("Acierto por bloque de 3 meses: entrenada una vez vs reentrenada")
    ax.legend(ncol=5, fontsize=8)
    guardar_fig("11_7_acierto_por_bloque")
    plt.show()
""")
md(r"""
## 11.8 ¿Y si hubiéramos invertido con cada modelo?
Una forma muy tangible de medir: **plata**. Simulamos que cada 7 días, empezando el día del halving:
- si el modelo dice **"sube"** → compramos Bitcoin por esa semana;
- si dice **"baja" o "lateral"** → nos quedamos en dólares.

Cada compra o venta paga una **comisión de 0,1%**. Lo comparamos con **comprar el día del halving y mantener** (*buy & hold*). Arrancamos con USD 1.000. No usamos ventas en corto, para mantenerlo simple.
""")
code(r"""
COMISION = 0.001
semanas = idx_test[::HORIZONTE]
ret_semana = np.expm1(datos["ret_futuro"].to_numpy()[semanas])
fechas_semana = datos.index[semanas]

capital = {"Comprar y mantener": 1000 * np.cumprod(1 + ret_semana) * (1 - COMISION)}
semanas_adentro = {"Comprar y mantener": 1.0}
for n in PRINCIPALES + ["Inercia"]:
    adentro = (PREDICCIONES[n][semanas] == ID["sube"]).astype(float)
    cambios = np.abs(np.diff(adentro, prepend=0))
    capital[n] = 1000 * np.cumprod(1 + adentro * ret_semana - cambios * COMISION)
    semanas_adentro[n] = adentro.mean()

resumen_inversion = pd.DataFrame({"capital final (USD)": {n: c[-1] for n, c in capital.items()},
                                  "semanas invertido": semanas_adentro}).round(3)
guardar_tabla(resumen_inversion, "11_8_inversion_simulada")
anotar("11.8 Inversión simulada (USD 1.000 el día del halving):\n" + resumen_inversion.to_string())

plt.figure(figsize=(14, 5))
for n, c in capital.items():
    plt.plot(fechas_semana, c, label=f"{n}: USD {c[-1]:,.0f}", linewidth=2.5 if n == "Comprar y mantener" else 1.8,
             color="black" if n == "Comprar y mantener" else COLOR_MODELO.get(n, "#bbbbbb"))
plt.axhline(1000, color="gray", linestyle=":")
plt.ylabel("Capital (USD)")
plt.title("Si hubiéramos invertido con cada modelo desde el halving de abril de 2024")
plt.legend()
guardar_fig("11_8_inversion_simulada")
plt.show()
""")

# =====================================================================
md(r"""
---
# 12. Anatomía de los errores: ¿en qué se equivocan las redes y por qué?

Un porcentaje de acierto no alcanza: hay que **mirar los errores** para entender qué aprendió cada red y qué no. Seguimos estos criterios:
- **La red siempre contesta una de las 3 clases**, aunque lo que ve no se parezca a nada de lo que estudió: no tiene la opción "no sé". → ¿Sabe cuándo no sabe? (12.2) ¿Se equivoca más con las fotos **nuevas**? (12.3)
- **Lo aprendido en un período puede no servir en otro** si el mercado cambió de comportamiento. → ¿El ciclo actual se mueve distinto que los anteriores? (12.4)
- **Hay cosas que no están en las velas:** noticias, regulación, liquidaciones. → ¿Los errores se concentran en las sorpresas? (12.5)

## 12.1 Los errores, en casos concretos
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

mostrar_casos("CNN", "foto", "12_1_casos_cnn")
mostrar_casos("LSTM", "secuencia", "12_1_casos_lstm")
""")
md(r"""
## 12.2 ¿La red sabe cuándo no sabe?
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
guardar_tabla(confianza_tabla.round(3), "12_2_confianza_vs_acierto")
anotar("12.2 Confianza vs acierto:\n" + confianza_tabla.round(3).to_string(index=False))

ax = confianza_tabla.pivot(index="confianza", columns="modelo", values="acierto")[PRINCIPALES].plot.bar(
    rot=0, figsize=(10, 4.5), color=[COLOR_MODELO[n] for n in PRINCIPALES])
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylabel("acierto")
ax.set_xlabel("grupo de confianza de la red (cuartos de los días del test)")
ax.set_title("¿Cuando la red está más segura, acierta más?")
ax.legend()
guardar_fig("12_2_confianza_vs_acierto")
plt.show()
""")
md(r"""
## 12.3 Fotos nuevas: ¿se equivocan más con lo que nunca vieron?
Usamos la distancia a la **gemela del pasado** (7.5): pocos píxeles distintos = la red **ya vio algo casi igual**; muchos = una foto **nueva**, distinta de todo el entrenamiento. Medimos el acierto de cada red según qué tan nueva es la foto de ese día, y mostramos la foto más conocida y la más nueva del test, al lado de su gemela.
""")
code(r"""
novedad_t = NOVEDAD[mt]
grupo_nov = pd.qcut(pd.Series(novedad_t).rank(method="first"), 4, labels=["muy conocida", "conocida", "rara", "muy nueva"]).to_numpy()
por_novedad = pd.DataFrame({n: pd.Series(PREDICCIONES[n][mt] == yt).groupby(grupo_nov, observed=True).mean() for n in PRINCIPALES})
por_novedad["píxeles distintos (promedio)"] = pd.Series(novedad_t).groupby(grupo_nov, observed=True).mean()
guardar_tabla(por_novedad.round(3), "12_3_novedad_vs_acierto")
anotar("12.3 Acierto según qué tan nueva es la foto:\n" + por_novedad.round(3).to_string())

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
guardar_fig("12_3_fotos_nuevas")
plt.show()
""")
md(r"""
## 12.4 ¿El mercado cambió? Cada período comparado
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
guardar_tabla(comparacion_periodos.round(3), "12_4_comparacion_periodos")
anotar("12.4 Comportamiento del mercado en cada período:\n" + comparacion_periodos.round(3).to_string())

grupo_vol = pd.qcut(datos.loc[mt, "vol_anual"], 3, labels=["semanas tranquilas", "normales", "agitadas"]).to_numpy()
acierto_vol = pd.DataFrame({n: pd.Series(PREDICCIONES[n][mt] == yt).groupby(grupo_vol, observed=True).mean() for n in PRINCIPALES})
guardar_tabla(acierto_vol.round(3), "12_4_acierto_segun_volatilidad")
anotar("12.4 Acierto según la volatilidad del momento (test):\n" + acierto_vol.round(3).to_string())

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
guardar_fig("12_4_cambio_de_mercado")
plt.show()
""")
md(r"""
## 12.5 Las sorpresas: ¿había movimientos que no estaban en las velas?
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
guardar_tabla(acierto_z.round(3), "12_5_acierto_segun_tamano_movimiento")
anotar("12.5 Acierto según el tamaño del movimiento siguiente:\n" + acierto_z.round(3).to_string())

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
guardar_tabla(extremos, "12_5_semanas_extremas")
anotar("12.5 Las 10 semanas más extremas del test:\n" + extremos.to_string())

ax = acierto_z[PRINCIPALES].plot.bar(rot=15, figsize=(11, 4.5), color=[COLOR_MODELO[n] for n in PRINCIPALES])
ax.axhline(1 / 3, color="gray", linestyle="--", label="azar")
ax.set_ylabel("acierto")
ax.set_title("¿Dónde se equivocan? Según el tamaño del movimiento de la semana siguiente")
ax.legend()
guardar_fig("12_5_sorpresas")
plt.show()
display(extremos)
""")
md(r"""
## 12.6 En resumen: ¿por qué se equivocan?
Con los gráficos de arriba se puede contestar, con números propios, cuál de estas causas pesa más:

| Causa | Dónde se ve |
|---|---|
| **La frontera es difusa:** "lateral" vs "se movió" se decide por poco | 12.5 (lateral justo / movimiento justo) |
| **Fotos nuevas**, distintas de todo el entrenamiento | 12.3 |
| **El mercado cambió:** el ciclo actual se mueve distinto | 12.4 y 11.7 |
| **Noticias que no estaban en las velas** | 12.5 (movimientos enormes y semanas extremas) |
| **Confianza sin fundamento:** está igual de segura cuando acierta que cuando no | 12.2 |
| **El mercado es casi eficiente:** lo que se puede saber por las velas ya está en el precio | 11.2 (p-valores) |
""")

# =====================================================================
md(r"""
---
# 13. Conclusiones: ¿el precio de hoy estaba en las velas?

La celda de abajo arma las respuestas **con los números de esta corrida**. Cómo leerlas:
- **Si ninguna red tiene ★ (p < 0,05):** con estos métodos, las velas **no anticipaban** el movimiento semanal del ciclo actual mejor que el azar. Apoya **H0 (mercado eficiente)**: lo que dicen las velas ya está incorporado en el precio. Es una conclusión válida y defendible.
- **Si alguna red tiene ★:** hay un **poder predictivo débil pero real** (H1). Mirar en 12.4 y 11.7 si depende del momento del mercado (H2).
- **CNN vs LSTM:** si dan parecido, *cómo* se miran las velas (foto o secuencia) no cambia el resultado.
- **Mejoras:** si más datos o reentrenar ayudan, el problema era de cantidad de datos o de cambio del mercado; si no ayudan, la señal directamente no está.

- **Pre-entrenados:** si Kronos (que vio miles de millones de velas) tampoco le gana al azar, sobre todo en el examen limpio, la conclusión se refuerza: no es falta de datos, es que la señal no está.

**Trabajo futuro:** probar otros horizontes (1, 3, 14 días), agregar la fase del halving como dato extra, y ajustar (*fine-tuning*) Kronos con velas de Bitcoin.
""")
code(r"""
mejor_baseline = tabla_test[tabla_test["tipo"] == "Baseline"]["balanced_acc"].max()
anotar("13. CONCLUSIONES (números de esta corrida)")
for n in dict.fromkeys(REDES + [m for m in PREDICCIONES if "pre-entrenado" in m]):
    f = tabla_test.loc[n]
    veredicto = "LE GANA al azar (significativo)" if f["p_valor"] < 0.05 else "no se distingue del azar"
    anotar(f"- {n}: balanced accuracy {f['balanced_acc']:.3f} (azar 0,333; mejor baseline {mejor_baseline:.3f}) | "
           f"p-valor {f['p_valor']:.3f} → {veredicto} | errores graves {f['errores_graves'] / f['dias']:.0%}")
mejor = tabla_test["balanced_acc"].idxmax()
anotar(f"- Mejor modelo en el test: {mejor} ({tabla_test.loc[mejor, 'balanced_acc']:.3f})")
anotar(f"- Ventanas elegidas: CNN {N_CNN} velas, LSTM {N_LSTM} velas")
significativas = [n for n in REDES if tabla_test.loc[n, "p_valor"] < 0.05]

# --- La escalera de preguntas, respondida con los números de esta corrida ---
def ba(n):
    return tabla_test.loc[n, "balanced_acc"] if n in tabla_test.index else np.nan

def mejor_de(nombres):
    validos = [n for n in nombres if n in tabla_test.index]
    return max(validos, key=ba) if validos else None

def cambio(a, b):
    # Compara el modelo b contra a en el test
    if np.isnan(ba(a)) or np.isnan(ba(b)):
        return "no se pudo correr"
    dif = ba(b) - ba(a)
    return f"{b}: {ba(b):.3f} vs {a}: {ba(a):.3f} ({'+' if dif >= 0 else ''}{dif:.3f}) → " + (
        "mejora" if dif > 0.01 else "empeora" if dif < -0.01 else "prácticamente igual")

dvs = resumen_dvs["promedio"].unstack()
pre = [n for n in tabla_test.index if tipo_de(n) == "Pre-entrenado"]
respuestas = [
    ("1. ¿Las reglas simples anticipan algo?",
     f"mejor regla: {mejor_de(['Martillo', 'Inercia', 'Gemela del pasado'])} con {ba(mejor_de(['Martillo', 'Inercia', 'Gemela del pasado'])):.3f} (azar 0,333)"),
    ("2. ¿Cuántas velas conviene mirar?",
     f"CNN: {N_CNN} velas (todas entre {resumen_ventana_cnn['promedio'].min():.3f} y {resumen_ventana_cnn['promedio'].max():.3f}); "
     f"LSTM: {N_LSTM} velas (entre {resumen_ventana_lstm['promedio'].min():.3f} y {resumen_ventana_lstm['promedio'].max():.3f})"),
    ("3. ¿Velas diarias o semanales?",
     "; ".join(f"{per}: diarias {fila['diarias']:.3f} vs semanales {fila['semanales']:.3f}" for per, fila in dvs.iterrows()) + " (validación)"),
    ("4. ¿Foto (CNN) o secuencia (LSTM)?", cambio("CNN", "LSTM")),
    ("5. ¿Red híbrida CNN-LSTM?", cambio(mejor_de(["CNN", "LSTM"]), "CNN-LSTM híbrida")),
    ("6. ¿Más datos de acciones de EE.UU.?", cambio("CNN", "CNN + acciones") + " | " + cambio("LSTM", "LSTM + acciones")),
    ("7. ¿Agregar otras criptomonedas?", cambio("CNN", "CNN + criptos") + " | " + cambio("LSTM", "LSTM + criptos")),
    ("8. ¿Reentrenar cada 3 meses?", cambio("CNN", "CNN reentrenada") + " | " + cambio("LSTM", "LSTM reentrenada")),
    ("9. ¿Un pre-entrenado ve más?",
     (f"mejor pre-entrenado: {mejor_de(pre)} con {ba(mejor_de(pre)):.3f}" if pre else "ninguno pudo correr")),
    ("10. ¿Alguno le gana al azar en el ciclo actual?",
     (f"SÍ, de forma significativa: {', '.join(n for n in tabla_test.index if tabla_test.loc[n, 'p_valor'] < 0.05)}"
      if (tabla_test["p_valor"] < 0.05).any() else "NO: ningún modelo supera al azar de forma significativa (p < 0,05)")),
    ("11. ¿Por qué se equivocan?", "ver sección 12 (frontera difusa, fotos nuevas, cambio de mercado, noticias)"),
]
tabla_respuestas = pd.DataFrame(respuestas, columns=["pregunta", "respuesta (números de esta corrida)"])
guardar_tabla(tabla_respuestas, "13_respuestas_pregunta_por_pregunta")
anotar("13. RESPUESTAS PREGUNTA POR PREGUNTA:\n" + "\n".join(f"{p_}\n   → {r_}" for p_, r_ in respuestas))

import textwrap
fig, ax = plt.subplots(figsize=(16, 0.75 * len(respuestas) + 1))
ax.axis("off")
celdas = [[p_, textwrap.fill(r_, 95)] for p_, r_ in respuestas]
t = ax.table(cellText=celdas, colLabels=["Pregunta", "Respuesta (con los números de esta corrida)"],
             loc="center", cellLoc="left", colWidths=[0.27, 0.73])
t.auto_set_font_size(False)
t.set_fontsize(8.5)
t.scale(1, 2.6)
for (r, c), celda in t.get_celld().items():
    if r == 0:
        celda.set_facecolor("#264653")
        celda.get_text().set_color("white")
ax.set_title("La escalera de preguntas, respondida", pad=12)
guardar_fig("13_respuestas_pregunta_por_pregunta")
plt.show()
anotar("- Respuesta a la pregunta: " + (
    f"hay poder predictivo débil pero significativo en {', '.join(significativas)} (H1)." if significativas else
    "ninguna red supera al azar de forma significativa: el movimiento semanal del ciclo actual NO estaba "
    "anunciado en las velas, con estos métodos (apoya H0)."))
""")

# =====================================================================
md(r"""
---
# 📦 Guardar los resultados (zip)

Junta en un solo zip **todos los gráficos (PNG) y tablas (CSV)** de cada paso, más:
- `00_resumen.txt`: los números clave de cada paso, en orden.
- `predicciones.csv`: lo que dijo cada modelo para cada día.

Los archivos empiezan con el número de la sección, así quedan ordenados igual que el informe:
`03_` base de datos · `04_` ciclos · `05_` respuesta (target) · `06_` baselines · `07_` CNN · `08_` LSTM · `09_` mejoras · `10_` pre-entrenados · `11_` examen final · `12_` errores.
""")
code(r"""
import shutil

(RESULTADOS / "00_resumen.txt").write_text("\n\n".join(RESUMEN), encoding="utf-8")

predicciones = datos[["periodo", "close", "z", "clase"]].copy()
for n, pred in PREDICCIONES.items():
    predicciones[f"{n} dijo"] = [CLASES[v] for v in pred]
for n in PROBAS:
    predicciones[f"{n} confianza"] = PROBAS[n].max(1).round(4)
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
