"""
PASO 2 - Limpiar los datos crudos de 1 hora.

Qué revisamos (y por qué):
  1. Fechas: el timestamp viene en segundos Unix -> lo pasamos a fecha en UTC.
     Bitcoin opera 24/7, así que usamos UTC como referencia (no hay "horario de mercado").
  2. Duplicados y orden: el loop de descarga puede repetir alguna vela en los bordes.
  3. Velas imposibles: high < low, o open/close por fuera del rango [low, high].
  4. Huecos: horas que no existen en el archivo (el exchange caído o sin datos).
  5. Velas "planas": open = high = low = close con volumen 0 -> en esa hora
     NO hubo operaciones; Bitstamp repite el último precio. Muy común en 2011-2013,
     cuando casi todo el mercado operaba en Mt. Gox.
  6. La vela de la hora actual todavía no cerró -> la sacamos.

Las velas planas NO se borran: se marcan con la columna `sin_operaciones`,
así después decidimos cómo tratarlas (y el paso 3 las ignora al armar velas).

Uso:  .venv/bin/python src/02_limpiar.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ENTRADA = BASE / "data" / "raw" / "btcusd_1h_bitstamp.csv"
SALIDA = BASE / "data" / "raw" / "btcusd_1h_limpio.csv"
REPORTE = BASE / "data" / "validacion" / "reporte_limpieza.txt"

lineas = []
def log(txt=""):
    print(txt)
    lineas.append(str(txt))


def main():
    df = pd.read_csv(ENTRADA)
    log(f"Filas crudas: {len(df):,}")

    # 1. Fechas en UTC
    df["fecha"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    # 2. Duplicados y orden
    dup = df.duplicated("timestamp").sum()
    df = df.drop_duplicates("timestamp").sort_values("fecha").reset_index(drop=True)
    log(f"Duplicados eliminados: {dup}")

    # 6. Última vela incompleta (la hora en curso)
    ahora = pd.Timestamp.now(tz="UTC")
    incompletas = df["fecha"] + pd.Timedelta(hours=1) > ahora
    df = df[~incompletas]
    log(f"Velas sin cerrar eliminadas: {incompletas.sum()}")

    # 3. Velas imposibles
    malas = (
        (df["high"] < df["low"])
        | (df[["open", "close"]].max(axis=1) > df["high"] + 1e-9)
        | (df[["open", "close"]].min(axis=1) < df["low"] - 1e-9)
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    log(f"Velas imposibles (OHLC inconsistente): {malas.sum()}")
    df = df[~malas]

    # 5. Velas planas sin operaciones
    df["sin_operaciones"] = (df["volume"] == 0) & (df["high"] == df["low"])

    # 4. Huecos: comparamos contra una grilla completa de horas
    grilla = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="1h")
    faltantes = grilla.difference(pd.DatetimeIndex(df["fecha"]))

    log(f"\nPeríodo: {df['fecha'].min():%Y-%m-%d %H:%M} a {df['fecha'].max():%Y-%m-%d %H:%M} UTC")
    log(f"Horas esperadas: {len(grilla):,} | presentes: {len(df):,} | faltantes: {len(faltantes):,}")

    # Resumen por año: es lo que conviene mostrar en la presentación
    resumen = pd.DataFrame({
        "horas": df.groupby(df["fecha"].dt.year).size(),
        "sin_operaciones": df.groupby(df["fecha"].dt.year)["sin_operaciones"].sum(),
        "faltantes": pd.Series(faltantes.year).value_counts(),
    }).fillna(0).astype(int)
    resumen["%_con_operaciones"] = (
        100 * (resumen["horas"] - resumen["sin_operaciones"]) / (resumen["horas"] + resumen["faltantes"])
    ).round(1)
    resumen["volumen_BTC"] = df.groupby(df["fecha"].dt.year)["volume"].sum().round(0).astype(int)
    log("\nCalidad por año:")
    log(resumen.to_string())

    # Huecos largos (más de 6 horas seguidas sin ninguna vela)
    if len(faltantes):
        s = pd.Series(faltantes)
        grupo = (s.diff() != pd.Timedelta(hours=1)).cumsum()
        huecos = s.groupby(grupo).agg(["min", "max", "size"])
        largos = huecos[huecos["size"] > 6].sort_values("size", ascending=False)
        log(f"\nHuecos de más de 6 horas: {len(largos)}")
        for _, h in largos.head(10).iterrows():
            log(f"  {h['min']:%Y-%m-%d %H:%M} -> {h['max']:%Y-%m-%d %H:%M}  ({h['size']} horas)")

    df = df[["fecha", "timestamp", "open", "high", "low", "close", "volume", "sin_operaciones"]]
    df.to_csv(SALIDA, index=False)
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    REPORTE.write_text("\n".join(lineas), encoding="utf-8")
    log(f"\nGuardado: {SALIDA}\nReporte: {REPORTE}")


if __name__ == "__main__":
    main()
