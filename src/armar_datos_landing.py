import json, re, ast, pandas as pd, numpy as np, sys
S, R = sys.argv[1], sys.argv[2]
out = S + "/landing/"
v = pd.read_csv(S + "/corrida/datos_tp_btc/velas/btc_1d.csv", parse_dates=["fecha"])
v = v[v["fecha"] >= "2011-08-18"]
velas = [[int(f.timestamp()), round(o, 2), round(h, 2), round(l, 2), round(c, 2)] for f, o, h, l, c in
         zip(v.fecha, v.open, v.high, v.low, v.close)]
json.dump(velas, open(out + "btc_1d.json", "w"), separators=(",", ":"))

src = open(S + "/frag/04_historia.py").read()
m = re.search(r"CONTEXTO_HITOS = (\{.*?\n\})", src, re.S)
ctx = ast.literal_eval(m.group(1))
h = pd.read_csv(R + "/04_2_hitos.csv")
hitos = []
for _, r in h.iterrows():
    ini = pd.to_datetime(r["inicio"], dayfirst=True)
    clave = min(ctx, key=lambda k: abs((pd.Timestamp(k) - ini).days))
    hitos.append(dict(inicio=ini.strftime("%Y-%m-%d"), fin=pd.to_datetime(r["fin"], dayfirst=True).strftime("%Y-%m-%d"),
                      dias=int(r["días"]), p0=float(r["precio inicio"]), p1=float(r["precio fin"]), mov=r["movimiento"],
                      tipo=r["tipo"], abierto=r["estado"] != "cerrado", contexto=ctx[clave][0], fuente=ctx[clave][1]))
ev = pd.read_csv(R + "/04_3_eventos.csv")
eventos = [dict(fecha=pd.to_datetime(r.fecha, dayfirst=True).strftime("%Y-%m-%d"), evento=r.evento, ret=r["retorno 7 días"],
                pct=int(r.percentil), fuente=r.fuente) for _, r in ev.iterrows()]
b = pd.read_csv(R + "/05_benchmark.csv", index_col=0)
bench = [dict(activo=i, rend=r["rendimiento anual"], vol=r["volatilidad anual"], caida=r["peor caída"], velas=r["velas por año"],
              cuerpo=r["cuerpo medio (% del precio)"], salto=r["% días con salto de apertura"], corr=r["correlación con BTC"]) for i, r in b.iterrows()]
def tabla(nombre, **kw):
    return json.loads(pd.read_csv(R + "/" + nombre, **kw).to_json(orient="split", force_ascii=False))
fam = pd.read_csv(R + "/07_4_familias.csv", index_col=0)
cent = json.load(open(S + "/corrida/centros_familias.json"))["centros"]
familias = [dict(id=i, desc=r["descripción"], veces=int(r.veces), tr=r["dif z train"], va=r["dif z val"], te=r["dif z test"],
                 holm=r["p Holm"], sost=r["¿se sostiene?"], centro=c) for (i, r), c in zip(fam.iterrows(), cent)]
res = pd.read_csv(R + "/07_6_resumen_patrones.csv", index_col=0)
clas = res[res.tipo == "clásico"]
clasicos = [dict(nombre=i, veces=int(r.veces), tr=r["dif z train"], va=r["dif z val"], te=r["dif z test"], holm=r["p Holm (todos juntos)"],
                 veredicto=r.veredicto) for i, r in clas.iterrows()]
t = pd.read_csv(R + "/12_2_resultados_test.csv", index_col=0)
test = [dict(modelo=i, tipo=r.tipo, ba=r.balanced_acc, lo=r["IC95 desde"], hi=r["IC95 hasta"], p=r.p_valor, holm=r["p_corregido (Holm)"])
        for i, r in t.iterrows()]
inv = pd.read_csv(R + "/12_7_inversion_simulada.csv", index_col=0)
datos = dict(hitos=hitos, eventos=eventos, bench=bench, familias=familias, clasicos=clasicos, test=test,
             ciclos=tabla("07_3_rendimiento_por_anio_del_ciclo.csv", index_col=0),
             tipo_fase=tabla("07_3_tipo_de_vela_por_anio_del_ciclo.csv", index_col=0),
             planilla=tabla("07_2_planilla_del_profe.csv", index_col=0),
             brier=tabla("12_2b_calibracion.csv", index_col=0),
             inversion=[dict(modelo=i, capital=float(r.iloc[0])) for i, r in inv.iterrows()],
             extremas=tabla("13_5_semanas_extremas.csv", index_col=0),
             hitos_redes=tabla("12_6b_hitos_que_vieron_las_redes.csv", index_col=0),
             periodos=tabla("13_4_comparacion_periodos.csv", index_col=0),
             descartes=tabla("14_7_descartes.csv", index_col=0))
json.dump(datos, open(out + "datos.json", "w"), ensure_ascii=False, indent=0, default=float)
print(len(velas), velas[-1]); print(json.dumps(datos["ciclos"], ensure_ascii=False)[:300]); print(datos["periodos"]); print(datos["hitos_redes"]["columns"], datos["hitos_redes"]["data"][-1]); print(datos["planilla"])

# ---- v2: porcentajes simples por período y velas de otros activos ----
dd = pd.read_csv(S + "/corrida/datos_familias.csv", index_col=0)
PER = ["train", "val", "test"]
normal = {c: [(dd.clase[dd.periodo == p] == c).mean() for p in PER] for c in ["sube", "baja"]}
def porc(m, c):
    return [(dd.clase[m & (dd.periodo == p)] == c).mean() if (m & (dd.periodo == p)).any() else None for p in PER]
for f in datos["familias"]:
    n = int(f["id"][1:]); m = dd.familia == n
    c = "sube" if f["tr"] >= 0 else "baja"
    f.update(dir=c, pct=porc(m, c), normal=normal[c])
for c_ in datos["clasicos"]:
    m = dd["pat_" + c_["nombre"]]
    c = "sube" if c_["tr"] >= 0 else "baja"
    c_.update(dir=c, pct=porc(m, c), normal=normal[c])
VENTANAS = [("Últimos 3 meses", "2026-08-24", "2026-11-23"), ("La caída de oct-2025", "2025-09-20", "2025-12-20"),
            ("El pánico del COVID", "2020-02-10", "2020-04-30"), ("El pico de 2021", "2021-03-20", "2021-06-30")]
ACT = [("Bitcoin", None), ("Ethereum", "ETH-USD"), ("XRP", "XRP-USD"), ("Litecoin", "LTC-USD"), ("S&P 500", "GSPC")]
B = S + "/corrida/datos_tp_btc/raw/benchmark/"
series = {}
for nom, f in ACT:
    if f is None:
        x = v.set_index(v.fecha.dt.tz_localize(None).dt.normalize())[["open", "high", "low", "close"]]
    else:
        x = pd.read_csv(B + f + ".csv", index_col=0, parse_dates=True)[["open", "high", "low", "close"]]
    series[nom] = x
comp = []
for titulo, a, z in VENTANAS:
    w = {"titulo": titulo, "desde": a, "hasta": z, "activos": []}
    for nom, _ in ACT:
        x = series[nom].loc[a:z].dropna()
        base = x.open.iloc[0]
        w["activos"].append({"nombre": nom, "velas": [[k.strftime("%Y-%m-%d")] + [round(float(q) / base * 100, 2) for q in r] for k, r in x.iterrows()]})
    comp.append(w)
datos["comparacion"] = comp
datos["normal"] = normal
json.dump(datos, open(out + "datos.json", "w"), ensure_ascii=False, separators=(",", ":"), default=float)
print("v2 ok", [(f["id"], f["dir"], [round(p, 2) for p in f["pct"]]) for f in datos["familias"]][:3], len(json.dumps(datos)))
