"""
PASO 3b - Control de calidad + gráficos de los halvings.

1) VALIDACIÓN: ¿nuestras velas diarias (hechas a mano desde 1h) coinciden con
   las velas diarias "oficiales"?
     - Bitstamp diaria: misma fuente, debería dar casi idéntico.
     - Coinbase diaria (desde 2015, exchange de EE.UU.): otra fuente; tiene que
       dar MUY parecido (diferencias chicas por ser exchanges distintos).
2) GRÁFICOS:
     - Panorama semanal 2011-hoy en escala logarítmica con los 4 halvings.
     - Velas diarias alrededor de cada halving (60 días antes, 120 después).

Uso:  .venv/bin/python src/04_validar_y_graficar.py
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
VELAS = BASE / "data" / "velas"
VALID = BASE / "data" / "validacion"
GRAF = BASE / "graficos"

HALVINGS = {
    "1° halving (50 → 25 BTC)": "2012-11-28",
    "2° halving (25 → 12,5 BTC)": "2016-07-09",
    "3° halving (12,5 → 6,25 BTC)": "2020-05-11",
    "4° halving (6,25 → 3,125 BTC)": "2024-04-20",
}


# ---------- descarga de velas diarias de referencia ----------
def diaria_bitstamp() -> pd.DataFrame:
    filas, desde = [], int(pd.Timestamp("2011-08-01", tz="UTC").timestamp())
    while desde < time.time():
        r = requests.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/",
                         params={"step": 86400, "limit": 1000, "start": desde}, timeout=30)
        bloque = r.json()["data"]["ohlc"]
        if not bloque:
            break
        filas += bloque
        desde = int(bloque[-1]["timestamp"]) + 86400
        time.sleep(0.25)
    df = pd.DataFrame(filas).astype(float)
    df.index = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]].drop_duplicates()


def diaria_coinbase() -> pd.DataFrame:
    # Coinbase devuelve máx. 300 velas por pedido: [tiempo, low, high, open, close, volumen]
    filas, ini = [], pd.Timestamp("2015-07-20", tz="UTC")
    fin_total = pd.Timestamp.now(tz="UTC").normalize()
    while ini < fin_total:
        fin = min(ini + pd.Timedelta(days=299), fin_total)
        r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                         params={"granularity": 86400, "start": ini.isoformat(), "end": fin.isoformat()},
                         headers={"User-Agent": "tp-velas"}, timeout=30)
        filas += r.json()
        ini = fin + pd.Timedelta(days=1)
        time.sleep(0.4)
    df = pd.DataFrame(filas, columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]].sort_index().loc[lambda d: ~d.index.duplicated()]


def comparar(mia: pd.DataFrame, ref: pd.DataFrame, nombre: str) -> str:
    comun = mia.index.intersection(ref.index)
    m, r = mia.loc[comun], ref.loc[comun]
    txt = [f"\n=== Nuestras velas diarias vs {nombre} ({len(comun):,} días en común) ==="]
    for col in ["open", "high", "low", "close"]:
        dif = (m[col] / r[col] - 1).abs() * 100
        txt.append(f"  {col:5}: diferencia mediana {dif.median():.4f}% | "
                   f"días con dif > 1%: {(dif > 1).sum()} ({100*(dif > 1).mean():.2f}%)")
    return "\n".join(txt)


# ---------- gráficos ----------
def grafico_panorama(w: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.semilogy(w.index, w["close"], color="#f7931a", lw=1.4)
    for i, (nombre, fecha) in enumerate(HALVINGS.items()):
        f = pd.Timestamp(fecha, tz="UTC")
        ax.axvline(f, color="#444", ls="--", lw=1)
        ax.text(f, ax.get_ylim()[1] * 0.6 if i % 2 else ax.get_ylim()[1] * 0.25, " " + nombre,
                fontsize=8, rotation=90, va="top")
    ax.set_title("Bitcoin (BTC/USD, Bitstamp) – cierre semanal en escala logarítmica y halvings")
    ax.set_ylabel("USD (escala log)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(GRAF / "00_panorama_halvings.png", dpi=150)
    plt.close(fig)


def graficos_halvings(d: pd.DataFrame):
    estilo = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle=":")
    for i, (nombre, fecha) in enumerate(HALVINGS.items(), start=1):
        f = pd.Timestamp(fecha, tz="UTC")
        tramo = d.loc[f - pd.Timedelta(days=60): f + pd.Timedelta(days=120)].copy()
        tramo.index = tramo.index.tz_localize(None)
        mpf.plot(tramo[["open", "high", "low", "close", "volume"]], type="candle", style=estilo,
                 volume=True, title=f"\nBTC/USD velas diarias – {nombre} ({fecha})",
                 vlines=dict(vlines=[f.tz_localize(None)], colors="#444", linestyle="--"),
                 figsize=(14, 7), savefig=dict(fname=GRAF / f"0{i}_halving_{fecha[:4]}.png", dpi=150))


def main():
    VALID.mkdir(parents=True, exist_ok=True)
    GRAF.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(VELAS / "btc_1d.csv", parse_dates=["fecha"]).set_index("fecha")
    w = pd.read_csv(VELAS / "btc_1w.csv", parse_dates=["fecha"]).set_index("fecha")

    print("Descargando diaria de Bitstamp y de Coinbase para validar...")
    bs, cb = diaria_bitstamp(), diaria_coinbase()
    bs.to_csv(VALID / "bitstamp_1d_oficial.csv")
    cb.to_csv(VALID / "coinbase_1d_oficial.csv")

    activas = d[~d["sin_operaciones"]]
    reporte = comparar(activas, bs, "Bitstamp diaria oficial") + comparar(activas, cb, "Coinbase diaria (otro exchange)")
    print(reporte)
    (VALID / "reporte_validacion.txt").write_text(reporte, encoding="utf-8")

    grafico_panorama(w)
    graficos_halvings(d)
    print(f"\nGráficos guardados en {GRAF}")


if __name__ == "__main__":
    main()
