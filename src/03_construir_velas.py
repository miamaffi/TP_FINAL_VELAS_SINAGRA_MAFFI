"""
PASO 3 - Construir nuestras propias velas japonesas de 4h, 1 día y 1 semana
a partir de las velas de 1 hora.

La regla para "juntar" varias velas en una más grande (resample):
    open   = el PRIMER precio del período   (apertura de la primera hora)
    high   = el MÁXIMO de todos los máximos
    low    = el MÍNIMO de todos los mínimos
    close  = el ÚLTIMO precio del período   (cierre de la última hora)
    volume = la SUMA de los volúmenes

Detalle importante: las horas "sin operaciones" (velas planas que Bitstamp
rellena repitiendo el último precio) NO se usan para armar las velas, porque
ese precio no ocurrió en ese período. Si un período entero no tuvo
operaciones, queda una vela plana marcada con `sin_operaciones = True`.

Convenciones (las mismas que usa el mercado cripto):
  - Todo en UTC. La vela diaria va de 00:00 a 24:00 UTC (21:00 a 21:00 en Argentina).
  - Las velas de 4h arrancan a las 00, 04, 08, 12, 16 y 20 UTC.
  - La vela semanal va de lunes 00:00 a domingo 24:00 UTC.
  - La etiqueta de cada vela es su fecha/hora de APERTURA.
  - Se descarta la última vela de cada temporalidad si todavía no cerró.

Uso:  .venv/bin/python src/03_construir_velas.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "data" / "raw" / "btcusd_1h_limpio.csv"
CARPETA = BASE / "data" / "velas"

TEMPORALIDADES = {
    # nombre: (regla de pandas, duración de la vela)
    "4h": ("4h", pd.Timedelta(hours=4)),
    "1d": ("1D", pd.Timedelta(days=1)),
    "1w": ("W-MON", pd.Timedelta(weeks=1)),
}

REGLAS_OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def construir(h1: pd.DataFrame, regla: str, duracion: pd.Timedelta) -> pd.DataFrame:
    activas = h1[~h1["sin_operaciones"]]
    kwargs = {"label": "left", "closed": "left"}

    velas = activas.resample(regla, **kwargs).agg(REGLAS_OHLCV)
    velas["horas_con_operaciones"] = activas["close"].resample(regla, **kwargs).count()

    # Grilla completa de períodos (incluye los que no tuvieron ninguna operación)
    todas = h1["close"].resample(regla, **kwargs).last().index
    velas = velas.reindex(todas)
    velas["horas_con_operaciones"] = velas["horas_con_operaciones"].fillna(0).astype(int)
    velas["sin_operaciones"] = velas["horas_con_operaciones"] == 0

    # Períodos sin operaciones: vela plana en el último precio conocido
    ultimo_precio = h1["close"].resample(regla, **kwargs).last()
    for col in ["open", "high", "low", "close"]:
        velas[col] = velas[col].fillna(ultimo_precio)
    velas["volume"] = velas["volume"].fillna(0)

    # Sacar la última vela si todavía no cerró
    fin_datos = h1.index.max() + pd.Timedelta(hours=1)
    velas = velas[velas.index + duracion <= fin_datos]

    velas.index.name = "fecha"
    return velas


def main():
    h1 = pd.read_csv(ENTRADA, parse_dates=["fecha"]).set_index("fecha")
    CARPETA.mkdir(parents=True, exist_ok=True)

    # Guardamos también la de 1h (limpia) en la carpeta de velas, para tener todo junto
    h1_out = h1[["open", "high", "low", "close", "volume", "sin_operaciones"]]
    h1_out.to_csv(CARPETA / "btc_1h.csv")
    print(f"1h : {len(h1_out):>7,} velas")

    for nombre, (regla, duracion) in TEMPORALIDADES.items():
        velas = construir(h1, regla, duracion)
        velas.round(8).to_csv(CARPETA / f"btc_{nombre}.csv")
        planas = velas["sin_operaciones"].sum()
        print(f"{nombre:3}: {len(velas):>7,} velas | {velas.index.min():%Y-%m-%d} -> "
              f"{velas.index.max():%Y-%m-%d %H:%M} | sin operaciones: {planas}")

    # Control: la primera vela semanal tiene que empezar un lunes
    w = pd.read_csv(CARPETA / "btc_1w.csv", parse_dates=["fecha"])
    assert (w["fecha"].dt.day_of_week == 0).all(), "las semanas no arrancan en lunes"
    print("\nOK: todas las velas semanales empiezan un lunes.")


if __name__ == "__main__":
    main()
