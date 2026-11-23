"""
PASO 1 - Descargar velas de 1 hora de BTC/USD desde Bitstamp.

¿Por qué Bitstamp?  Es el único exchange con API pública gratis que tiene datos
desde 2011, así que cubre los 4 halvings (2012, 2016, 2020, 2024).

¿Por qué velas de 1 hora?  Es la "materia prima": con velas de 1h después
armamos nosotras las de 4h, 1 día y 1 semana (paso 3). Con 1 minuto serían
~8 millones de filas sin ganar nada para nuestro análisis.

La API devuelve como máximo 1000 velas por pedido, así que hacemos un loop:
pedimos 1000 horas, guardamos, avanzamos y repetimos hasta hoy.

Uso:  .venv/bin/python src/01_descargar.py
"""
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
PASO_SEG = 3600                # 3600 segundos = velas de 1 hora
LIMITE = 1000                  # máximo de velas por pedido que acepta Bitstamp
INICIO = int(pd.Timestamp("2011-08-01", tz="UTC").timestamp())  # antes del 1er dato
SALIDA = Path(__file__).resolve().parents[1] / "data" / "raw" / "btcusd_1h_bitstamp.csv"


def pedir_bloque(desde_unix: int) -> list[dict]:
    """Pide hasta 1000 velas de 1h empezando en `desde_unix`. Reintenta si falla."""
    params = {"step": PASO_SEG, "limit": LIMITE, "start": desde_unix}
    for intento in range(5):
        try:
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["data"]["ohlc"]
        except Exception as e:  # corte de internet, límite de pedidos, etc.
            espera = 2 ** intento
            print(f"  error ({e}); reintento en {espera}s")
            time.sleep(espera)
    raise RuntimeError("Bitstamp no respondió después de 5 intentos")


def main():
    ahora = int(time.time())
    desde = INICIO
    filas = []
    pedidos = 0

    while desde < ahora:
        bloque = pedir_bloque(desde)
        pedidos += 1
        if bloque:
            filas.extend(bloque)
            ultimo = int(bloque[-1]["timestamp"])
            desde = ultimo + PASO_SEG            # seguimos desde la hora siguiente
        else:
            desde += PASO_SEG * LIMITE           # tramo sin datos: saltamos 1000 horas
        if pedidos % 10 == 0:
            fecha = pd.to_datetime(desde, unit="s", utc=True)
            print(f"{pedidos} pedidos | {len(filas):,} velas | voy por {fecha:%Y-%m-%d}")
        time.sleep(0.25)                          # no saturar la API

    df = pd.DataFrame(filas)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"\nListo: {len(df):,} velas de 1h guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
