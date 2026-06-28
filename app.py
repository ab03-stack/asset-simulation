# =============================================================================
#  Asset Simulationstool fuer Versorgungsnetze  (angelehnt an KANEW 3S) – v2.1
#  -----------------------------------------------------------------------------
#  Einzeldatei-Streamlit-App. Starten mit:   streamlit run app.py
#
#  THEORETISCHE GRUNDLAGE (Vorlesung Asset Management, HSD, V8):
#   * Schadens-/Ausfallrate = HAZARDRATE  lambda(t) = f(t) / (1 - F(t))   (S.9/25)
#   * Kanonische Verteilung: WEIBULL  lambda(t) = (beta/eta)*(t/eta)^(beta-1),
#       F(t) = 1 - exp(-(t/eta)^beta),  R(t) = 1 - F(t)                    (S.26)
#   * Kumulierte Intensitaet  Lambda(t) = Integral_0^t lambda  vereinheitlicht:
#       - erwartete Brueche einer Leitung (Laenge L) in (a, a+H] = L*(Lambda(a+H)-Lambda(a))
#       - P(mind. 1 Bruch) = 1 - exp(-L*DeltaLambda)        (Poisson, reparierbar)
#       - FOpt: erneuern, wenn erwartete Reparaturkosten > Erneuerungskosten
#   * Kosten: CAPEX + OPEX = TOTEX, je mit Inflation UND Diskontsatz       (S.20/34)
#   * Strategien: Zielwert (Alter/Schadensrate), Risiko-Budget, FOpt        (S.31)
#
#  Benoetigte Pakete:  streamlit, pandas, numpy, plotly
# =============================================================================

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Asset Simulation Versorgungsnetz",
                   page_icon="🛠️", layout="wide")


def _finde_csv(dateiname):
    """Sucht eine CSV-Datei an den ueblichen Orten (lokal & Browser/stlite)."""
    kandidaten = [dateiname, os.path.join(os.getcwd(), dateiname)]
    try:
        kandidaten.insert(1, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dateiname))
    except NameError:
        pass
    for p in kandidaten:
        if os.path.exists(p):
            return p
    return dateiname

DATEI_LEIT   = "Leitungen_Alle.csv"
DATEI_SCHAED = "Schäden.csv"

BEOB_START, BEOB_ENDE = 1960, 2017
MATERIALIEN = ["GG", "PE"]
ALTER_MAX   = 130
BEZUGSALTER = 80     # Anker fuer Linear/Potenz/Exponential (sr80 = Rate bei Alter 80)

# Funktionstypen und ihre Parameter inkl. Slider-Bereichen (variable Anzahl)
PARAMSPEC = {
    "Weibull":     [("eta",  "Skala η (charakt. Lebensdauer)", 10.0, 200.0, 1.0),
                    ("beta", "Form β (>1 = Verschleiß)",       0.3,   6.0, 0.1)],
    "Sigmoid":     [("c",  "Sättigungsrate c (max λ)",          0.0,   3.0, 0.01),
                    ("t0", "Wendepunkt-Alter t₀ [a]",          20.0, 120.0, 1.0),
                    ("k",  "Steilheit k",                       0.02,  0.5, 0.01)],
    "Zonen":       [("t1",   "Ende Frühausfälle t₁ [a]",        1.0,  40.0, 1.0),
                    ("t2",   "Beginn Verschleiß t₂ [a]",       20.0, 120.0, 1.0),
                    ("lam1", "λ Frühausfälle [1/(km·a)]",       0.0,   3.0, 0.01),
                    ("lam2", "λ Zufallsausfälle [1/(km·a)]",    0.0,   2.0, 0.01),
                    ("lam3", "λ Verschleiß [1/(km·a)]",         0.0,   5.0, 0.01)],
    "Potenz":      [("sr80", "sr₈₀ (Rate bei Alter 80)",        0.0,   6.0, 0.01),
                    ("b",    "Exponent b (= β − 1)",            0.0,   7.0, 0.1)],
    "Exponential": [("sr80", "sr₈₀ (Rate bei Alter 80)",        0.0,   6.0, 0.01),
                    ("b",    "Wachstum b",                      0.0,   0.2, 0.001)],
    "Linear":      [("sr80", "sr₈₀ (Rate bei Alter 80)",        0.0,   6.0, 0.01),
                    ("b",    "Steigung b",                     -0.5,   0.5, 0.001)],
}
TYPEN = list(PARAMSPEC.keys())
# Sinnvolle Vorbelegung je Material (Vorgabe Dozent): GG=Sigmoid, PE=Zonen
DEFAULT_TYP = {"GG": "Sigmoid", "PE": "Zonen"}


def _round_to_step(typ, key, val):
    """Rundet einen Parameterwert auf die Slider-Schrittweite und in den Bereich."""
    for k, _label, lo, hi, s in PARAMSPEC[typ]:
        if k == key:
            return float(np.clip(round(round(val / s) * s, 6), lo, hi))
    return float(val)


def params_text(p):
    """Kompakte Parameterbeschriftung fuer beliebigen Funktionstyp."""
    teile = []
    for k, *_ in PARAMSPEC[p["typ"]]:
        v = p.get(k)
        if v is None:
            continue
        teile.append(f"{k}={v:.3g}")
    return ", ".join(teile)


# =============================================================================
#  1) DATEN-IMPORT & BEREINIGUNG
# =============================================================================
@st.cache_data(show_spinner=False)
def lade_daten(leit_bytes=None, sch_bytes=None):
    """Liest beide CSV-Dateien (Semikolon, deutsches Zahlenformat, UTF-8-BOM)."""
    leit_src = io.BytesIO(leit_bytes) if leit_bytes is not None else _finde_csv(DATEI_LEIT)
    sch_src  = io.BytesIO(sch_bytes)  if sch_bytes  is not None else _finde_csv(DATEI_SCHAED)

    leit = pd.read_csv(leit_src, sep=";", decimal=",", encoding="utf-8-sig")
    leit.columns = [c.strip() for c in leit.columns]
    leit = leit[[c for c in leit.columns if not c.startswith("Unnamed")]]
    leit = leit.rename(columns={
        "Risikobewertung_Städtische Lage*Bebauung": "Risiko",
        "Städtischen Lage": "Lage"})
    for c in ["MAT", "Lage", "Bebauung", "MAT_zusammengefasst"]:
        if c in leit.columns:
            leit[c] = leit[c].astype(str).str.strip()
    for c in ["DOI", "DN", "DOA", "LNG", "Druck", "Verlegetiefe", "Risiko"]:
        if c in leit.columns:
            leit[c] = pd.to_numeric(leit[c], errors="coerce")
    leit["LNG_km"] = leit["LNG"] / 1000.0

    sch = pd.read_csv(sch_src, sep=";", decimal=",", encoding="utf-8-sig")
    sch.columns = [c.strip() for c in sch.columns]
    sch = sch[[c for c in sch.columns if not c.startswith("Unnamed")]]
    sch = sch.rename(columns={"Alter bei Schadenseintritt DOE3": "Alter"})
    for c in ["Material", "Schadensart", "Schadensursache"]:
        if c in sch.columns:
            sch[c] = sch[c].astype(str).str.strip()
    for c in ["DOI", "DOA", "DOE3", "Alter"]:
        if c in sch.columns:
            sch[c] = pd.to_numeric(sch[c], errors="coerce")
    return leit, sch


# =============================================================================
#  EMPIRISCHE SCHADENSRATE  (Schaeden je Alter / Wissenslaenge)
# =============================================================================
@st.cache_data(show_spinner=False)
def empirische_rate(leit, sch):
    """Je Material/Alter: Wissenslaenge [km·a], Anzahl Schaeden, empirische
    Rate [1/(km·a)] = Schaeden / Wissenslaenge."""
    alter = np.arange(0, ALTER_MAX + 1)
    ergebnis = {}
    for mat in MATERIALIEN:
        teil = leit[leit["MAT"] == mat]
        exposition = np.zeros(len(alter))
        doi = teil["DOI"].to_numpy(); doa = teil["DOA"].to_numpy()
        L_km = teil["LNG_km"].to_numpy()
        serv_ende = np.where(np.isnan(doa), BEOB_ENDE, doa)
        for a in alter:
            kalender = doi + a
            aktiv = (kalender >= BEOB_START) & (kalender <= BEOB_ENDE) & \
                    (kalender >= doi) & (kalender <= serv_ende)
            exposition[a] = L_km[aktiv].sum()
        dmg = sch[(sch["Material"] == mat) & sch["Alter"].notna()]
        anzahl = np.array([(dmg["Alter"] == a).sum() for a in alter], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(exposition > 0, anzahl / exposition, np.nan)
        ergebnis[mat] = pd.DataFrame({"Alter": alter, "Wissenslaenge": exposition,
                                      "Schaeden": anzahl, "Rate": rate})
    return ergebnis


def binne_rate(df_rate, breite):
    """Fasst die alters-feine Rate in Altersklassen zusammen (ruhigere Punkte)."""
    df = df_rate.copy()
    df["Klasse"] = (df["Alter"] // breite) * breite + breite / 2.0
    g = df.groupby("Klasse").agg(Schaeden=("Schaeden", "sum"),
                                 Wissenslaenge=("Wissenslaenge", "sum")).reset_index()
    g = g[g["Wissenslaenge"] > 0]
    g["Rate"] = g["Schaeden"] / g["Wissenslaenge"]
    return g


# =============================================================================
#  SCHADENSFUNKTIONEN (Hazardraten) + Survival + kumulierte Intensitaet
# =============================================================================
def sr_eval(typ, p, alter):
    """Hazardrate lambda(alter) [1/(km·a)]."""
    t = np.asarray(alter, dtype=float)
    if typ == "Weibull":
        tt = np.maximum(t, 1e-6)
        y = (p["beta"] / p["eta"]) * (tt / p["eta"]) ** (p["beta"] - 1.0)
    elif typ == "Sigmoid":
        # logistische Hazardrate: niedrig -> steiler Anstieg um t0 -> Sättigung c
        y = p["c"] / (1.0 + np.exp(-p["k"] * (t - p["t0"])))
    elif typ == "Zonen":
        # Badewannenkurve: Frühausfälle | Zufallsausfälle | Verschleiß
        y = np.where(t < p["t1"], p["lam1"],
                     np.where(t < p["t2"], p["lam2"], p["lam3"]))
        y = np.asarray(y, dtype=float)
    elif typ == "Potenz":
        tt = np.maximum(t, 1e-6)
        y = p["sr80"] * (tt / BEZUGSALTER) ** p["b"]
    elif typ == "Exponential":
        y = p["sr80"] * np.exp(p["b"] * (t - BEZUGSALTER))
    else:  # Linear
        y = p["sr80"] + p["b"] * (t - BEZUGSALTER)
    return np.clip(y, 0, None)


def schadensrate(material, alter, fit):
    p = fit[material]
    return sr_eval(p["typ"], p, alter)


def kum_intensitaet(material, fit):
    """Kumulierte Intensitaet Lambda(t)=Integral_0^t lambda. Weibull analytisch
    ((t/eta)^beta), sonst numerische Trapez-Integration ueber ganze Jahre."""
    ages = np.arange(0, ALTER_MAX + 1).astype(float)
    p = fit[material]
    if p["typ"] == "Weibull":
        Lam = (ages / p["eta"]) ** p["beta"]
    else:
        lam = schadensrate(material, ages, fit)
        Lam = np.concatenate([[0.0], np.cumsum((lam[:-1] + lam[1:]) / 2.0)])
    return ages, Lam


def Lam_dict(fit):
    return {m: kum_intensitaet(m, fit) for m in MATERIALIEN}


def Lam_at(mat_arr, age_arr, Ld):
    """Interpoliert Lambda je Element anhand seines Materials (mit Clamping)."""
    out = np.zeros(len(age_arr))
    for m in MATERIALIEN:
        sel = mat_arr == m
        if sel.any():
            ages, Lam = Ld[m]
            out[sel] = np.interp(np.clip(age_arr[sel], 0, ages[-1]), ages, Lam)
    return out


# =============================================================================
#  AUTOMATISCHE KALIBRIERUNG  (gewichteter Ausgleich, Gewicht = Wissenslaenge)
# =============================================================================
def _wls(x, y, w):
    """Gewichtete lineare Regression y = A + B*x -> (A, B)."""
    W = w.sum()
    if W <= 0 or len(x) < 2:
        return (float(np.average(y, weights=w)) if W > 0 else 0.0), 0.0
    mx = (w * x).sum() / W; my = (w * y).sum() / W
    var = (w * (x - mx) ** 2).sum()
    if var <= 0:
        return my, 0.0
    B = (w * (x - mx) * (y - my)).sum() / var
    return my - B * mx, B


def _gew_r2(y, yhat, w):
    W = w.sum()
    if W <= 0:
        return 0.0
    my = (w * y).sum() / W
    ss_res = (w * (y - yhat) ** 2).sum()
    ss_tot = (w * (y - my) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _fit_sigmoid(t, y, w):
    """Gewichtete Anpassung der logistischen Hazardrate c/(1+exp(-k(t-t0)))
    per Grid-Search mit lokaler Verfeinerung (numpy-only, kein scipy noetig)."""
    ymax = max(float(y.max()), 0.1)

    def suche(cs, t0s, ks, best):
        for c in cs:
            for t0 in t0s:
                for k in ks:
                    yh = c / (1.0 + np.exp(-k * (t - t0)))
                    sse = (w * (y - yh) ** 2).sum()
                    if best is None or sse < best[0]:
                        best = (sse, c, t0, k)
        return best

    best = suche(np.linspace(0.5 * ymax, 2.0 * ymax, 14),
                 np.linspace(20, 120, 21), np.linspace(0.03, 0.45, 15), None)
    # lokale Verfeinerung um das Optimum
    _, c, t0, k = best
    best = suche(np.linspace(0.7 * c, 1.3 * c, 9),
                 np.linspace(t0 - 8, t0 + 8, 9),
                 np.linspace(max(0.02, k - 0.06), k + 0.06, 9), best)
    return {"typ": "Sigmoid", "c": best[1], "t0": best[2], "k": best[3]}


def kalibriere(df_rate, typ, p0=None):
    """Parameter je Typ per gewichtetem Ausgleich + R² auf Raten-Skala.
    p0: aktuelle Parameter (fuer 'Zonen' werden die Grenzen t1/t2 daraus uebernommen)."""
    df = df_rate[df_rate["Wissenslaenge"] > 0]
    t = df["Alter"].to_numpy(dtype=float)
    y = df["Rate"].to_numpy(dtype=float)
    w = df["Wissenslaenge"].to_numpy(dtype=float)

    if typ == "Sigmoid":
        p = _fit_sigmoid(t, y, w)
    elif typ == "Zonen":
        # Grenzen aus aktuellen Reglern (oder Defaults), Raten = zonenweises Mittel
        t1 = float(p0["t1"]) if p0 and "t1" in p0 else 5.0
        t2 = float(p0["t2"]) if p0 and "t2" in p0 else 60.0

        def zmean(mask):
            ww = w[mask]
            return float((ww * y[mask]).sum() / ww.sum()) if ww.sum() > 0 else 0.0
        p = {"typ": "Zonen", "t1": t1, "t2": t2,
             "lam1": zmean(t < t1), "lam2": zmean((t >= t1) & (t < t2)),
             "lam3": zmean(t >= t2)}
    elif typ == "Weibull":
        m = (y > 0) & (t > 0)
        c, s = _wls(np.log(t[m]), np.log(y[m]), w[m])   # log y = c + s*log t
        beta = float(np.clip(s + 1.0, 0.3, 6.0))
        # c = log(beta) - beta*log(eta)  ->  eta = exp((log(beta) - c)/beta)
        eta = float(np.clip(np.exp((np.log(beta) - c) / beta), 10.0, 200.0))
        p = {"typ": "Weibull", "eta": eta, "beta": beta}
    elif typ == "Potenz":
        m = (y > 0) & (t > 0)
        c, b = _wls(np.log(t[m] / BEZUGSALTER), np.log(y[m]), w[m])
        p = {"typ": "Potenz", "sr80": float(np.exp(c)), "b": float(np.clip(b, 0, 7))}
    elif typ == "Exponential":
        m = y > 0
        c, b = _wls(t[m] - BEZUGSALTER, np.log(y[m]), w[m])
        p = {"typ": "Exponential", "sr80": float(np.exp(c)), "b": float(np.clip(b, 0, 0.2))}
    else:  # Linear
        A, B = _wls(t - BEZUGSALTER, y, w)
        p = {"typ": "Linear", "sr80": max(A, 0.0), "b": B}

    p["R2"] = _gew_r2(y, sr_eval(typ, p, t), w)
    return p


def bestes_modell(df_rate):
    return max((kalibriere(df_rate, t) for t in TYPEN), key=lambda p: p["R2"])


def guete(df_rate, p):
    df = df_rate[df_rate["Wissenslaenge"] > 0]
    t = df["Alter"].to_numpy(dtype=float)
    y = df["Rate"].to_numpy(dtype=float)
    w = df["Wissenslaenge"].to_numpy(dtype=float)
    return _gew_r2(y, sr_eval(p["typ"], p, t), w)


# =============================================================================
#  BRUCH-PROGNOSE (B)  – je Leitung erwartete Brueche, P(>=1), naechstes Jahr
# =============================================================================
def bruch_prognose(start, fit, ref_jahr, horizont):
    Ld = Lam_dict(fit)
    a0 = start["alter0"].to_numpy(dtype=float)
    L = start["LNG_km"].to_numpy()
    mat = start["MAT"].to_numpy()
    aid = start["AID"].to_numpy() if "AID" in start.columns else np.arange(len(start))
    risiko = start["Risiko"].fillna(0).to_numpy()

    exp_h = np.zeros(len(start))
    next_year = np.full(len(start), np.nan)
    for m in MATERIALIEN:
        sel = mat == m
        if not sel.any():
            continue
        ages, Lam = Ld[m]
        L0 = np.interp(a0[sel], ages, Lam)
        LH = np.interp(np.minimum(a0[sel] + horizont, ages[-1]), ages, Lam)
        exp_h[sel] = L[sel] * (LH - L0)
        ziel = L0 + 1.0 / np.maximum(L[sel], 1e-9)        # 1 erwarteter Bruch
        na = np.interp(ziel, Lam, ages)                    # Inversion (monoton)
        ny = np.where(ziel <= Lam[-1], ref_jahr + (na - a0[sel]), np.nan)
        next_year[np.where(sel)[0]] = ny

    P = 1.0 - np.exp(-exp_h)
    df = pd.DataFrame({
        "AID": aid, "MAT": mat, "Alter": a0.astype(int),
        "Länge_km": np.round(L, 3), "Risiko": np.round(risiko, 0).astype(int),
        "Erw. Brüche (Horizont)": np.round(exp_h, 2),
        "P(≥1 Bruch)": np.round(P, 3),
        "Erw. nächstes Bruchjahr":
            np.where(np.isnan(next_year), np.nan, np.round(next_year, 0))})
    return df.sort_values("P(≥1 Bruch)", ascending=False).reset_index(drop=True)


# =============================================================================
#  ASSET-SIMULATION  (deterministisch, vektorisiert)
# =============================================================================
def simuliere(start, fit, ref_jahr, horizont,
              alters_ziel, rate_strategie, krit_rate,
              fopt, t_abschr,
              cap_aktiv, cap_km, risiko_prio,
              kosten_basis, kosten_schaden, erneuerungsmaterial,
              erneuerung_aktiv=True):
    """Bestandsentwicklung Jahr fuer Jahr. Kandidaten: Alters-Zielwert (Pflicht),
    optional Schadensraten-Schwelle und/oder FOpt (kostenoptimal). Kosten real
    (heutige Preise); Inflation/Diskont erst bei der Auswertung (barwert)."""
    mat = start["MAT"].to_numpy().copy()
    alter = start["alter0"].to_numpy().astype(float).copy()
    L_km = start["LNG_km"].to_numpy()
    L_m = start["LNG"].to_numpy()
    risiko = start["Risiko"].fillna(0).to_numpy()
    byear = start["DOI"].to_numpy().astype(float).copy()
    eur_pro_m = start["kostenfaktor"].to_numpy() * kosten_basis
    Ld = Lam_dict(fit)
    gesamt_km = L_km.sum() if L_km.sum() > 0 else 1.0

    def zeile(jahr, ern, sch, capex, opex, risk):
        return {"Jahr": jahr, "GG": L_km[mat == "GG"].sum(), "PE": L_km[mat == "PE"].sum(),
                "Erneuerung_km": ern, "Schaeden": sch, "CAPEX": capex, "OPEX": opex,
                "Risiko_gew": risk, "Netzalter": float((alter * L_km).sum() / gesamt_km),
                "GG_anteil": float(L_km[mat == "GG"].sum() / gesamt_km)}

    zeilen = [zeile(ref_jahr, 0.0, 0.0, 0.0, 0.0, 0.0)]

    for jahr in range(ref_jahr + 1, ref_jahr + horizont + 1):
        alter += 1.0
        sr = np.zeros(len(start))
        for m in MATERIALIEN:
            sel = mat == m
            if sel.any():
                sr[sel] = schadensrate(m, alter[sel], fit)
        erw = sr * L_km
        opex = erw.sum() * kosten_schaden
        risk = (erw * risiko).sum()

        idx = np.array([], dtype=int)
        if erneuerung_aktiv:
            kandidat = np.zeros(len(start), dtype=bool)
            for m in MATERIALIEN:
                kandidat |= (mat == m) & (alter >= alters_ziel[m])
            if rate_strategie:
                kandidat |= sr >= krit_rate
            if fopt:
                dL = Lam_at(mat, alter + t_abschr, Ld) - Lam_at(mat, alter, Ld)
                rep_kosten = L_km * dL * kosten_schaden     # erw. Reparaturkosten
                ern_kosten = L_m * eur_pro_m                # Erneuerungskosten
                kandidat |= rep_kosten > ern_kosten
            idx = np.where(kandidat)[0]

            if cap_aktiv and len(idx) > 0 and L_km[idx].sum() > cap_km:
                if risiko_prio:
                    ordnung = np.lexsort((-alter[idx], -risiko[idx]))
                else:
                    ordnung = np.argsort(-alter[idx])
                idx_sort = idx[ordnung]
                idx = idx_sort[np.cumsum(L_km[idx_sort]) <= cap_km]

        ern_laenge = L_km[idx].sum()
        capex = (L_m[idx] * eur_pro_m[idx]).sum()
        mat[idx] = erneuerungsmaterial; alter[idx] = 0.0; byear[idx] = jahr
        zeilen.append(zeile(jahr, ern_laenge, erw.sum(), capex, opex, risk))

    verlauf = pd.DataFrame(zeilen)
    end_bestand = pd.DataFrame({"Baujahr": byear, "MAT": mat, "LNG_km": L_km})
    return verlauf, end_bestand


def barwert(verlauf, ref_jahr, inflation, diskont):
    """Barwert/NPV der realen TOTEX, je Jahr aufgezinst (Inflation) und auf das
    Referenzjahr abgezinst (Diskont):  Summe TOTEX_real * ((1+i)/(1+r))^(t-ref)."""
    jahre = verlauf["Jahr"].to_numpy(dtype=float)
    totex = (verlauf["CAPEX"] + verlauf["OPEX"]).to_numpy()
    faktor = ((1.0 + inflation) / (1.0 + diskont)) ** (jahre - ref_jahr)
    return float((totex * faktor).sum())


# =============================================================================
#  APP-OBERFLAECHE
# =============================================================================
def main():
    st.title("🛠️ Asset Simulationstool für Versorgungsnetze")
    st.caption("Hazard-/Weibull-Schadensfunktionen · Bruch-Prognose · "
               "Zielwert-/FOpt-Strategien · TOTEX mit Inflation & Barwert — KANEW-3S-nah")

    sb = st.sidebar
    sb.header("⚙️ Steuerung")

    with sb.expander("🗂️ Datenquelle", expanded=False):
        up_leit = st.file_uploader("Leitungen-CSV (optional)", type=["csv"], key="up_leit")
        up_sch = st.file_uploader("Schäden-CSV (optional)", type=["csv"], key="up_sch")
    leit_bytes = up_leit.getvalue() if up_leit is not None else None
    sch_bytes = up_sch.getvalue() if up_sch is not None else None

    try:
        leit, sch = lade_daten(leit_bytes, sch_bytes)
    except FileNotFoundError:
        st.error("CSV-Dateien nicht gefunden. 'Leitungen_Alle.csv' und 'Schäden.csv' "
                 "ins App-Verzeichnis legen oder hochladen.")
        st.stop()

    raten = empirische_rate(leit, sch)

    with sb.expander("📅 Simulationsrahmen", expanded=True):
        ref_jahr = st.number_input("Referenzjahr (Start)", 2000, 2030, 2018, 1)
        horizont = st.slider("Zeithorizont [Jahre]", 10, 100, 50, 5)

    # --- Schadensfunktionen je Material (Auto-Kalibrierung + Regler) ---------
    def schreibe_params(m, typ, p):
        """Schreibt die kalibrierten Parameter schrittgenau in den Session-State."""
        for k, v in p.items():
            if k not in ("typ", "R2"):
                st.session_state[f"{m}_{typ}_{k}"] = _round_to_step(typ, k, float(v))

    def fit_block(material):
        m = material
        k_typ = f"typ_{m}"
        if k_typ not in st.session_state:
            st.session_state[k_typ] = DEFAULT_TYP.get(m, "Weibull")

        with sb.expander(f"📈 Schadensfunktion {m}", expanded=(m == "GG")):
            c1, c2 = st.columns(2)
            do_auto = c1.button("🎯 Auto-Fit", key=f"auto_{m}",
                                help="Kalibriert den gewählten Typ an die Daten")
            do_best = c2.button("🏆 Bestes Modell", key=f"best_{m}",
                                help="Wählt den Typ mit höchstem R²")
            if do_best:
                p = bestes_modell(raten[m])
                st.session_state[k_typ] = p["typ"]
                schreibe_params(m, p["typ"], p)
                st.rerun()

            typ = st.selectbox("Funktionstyp", TYPEN, key=k_typ)

            if do_auto:
                # aktuelle Parameter mitgeben (Zonen: Grenzen t1/t2 bleiben erhalten)
                cur = {k: st.session_state.get(f"{m}_{typ}_{k}")
                       for k, *_ in PARAMSPEC[typ]}
                cur = cur if all(v is not None for v in cur.values()) else None
                schreibe_params(m, typ, kalibriere(raten[m], typ, cur))
                st.rerun()

            # einmalig pro Typ vorbelegen (Sigmoid-Grid nur einmal rechnen)
            if any(f"{m}_{typ}_{k}" not in st.session_state for k, *_ in PARAMSPEC[typ]):
                schreibe_params(m, typ, kalibriere(raten[m], typ))

            vals = {}
            for key, label, lo, hi, step in PARAMSPEC[typ]:
                sk = f"{m}_{typ}_{key}"
                st.session_state[sk] = float(np.clip(st.session_state[sk], lo, hi))
                vals[key] = st.slider(f"{label} – {m}", lo, hi, step=step, key=sk)

            p = {"typ": typ, **vals}
            r2 = guete(raten[m], p)
            farbe = "🟢" if r2 >= 0.6 else ("🟡" if r2 >= 0.3 else "🔴")
            st.caption(f"{farbe} R² (aktuelle Kurve): **{r2:.2f}**")
            p["R2"] = r2
            return p

    fit = {"GG": fit_block("GG"), "PE": fit_block("PE")}

    with sb.expander("🔧 Erneuerungsstrategie", expanded=True):
        st.markdown("**Zielwertstrategie (Alter)** – Pflicht")
        alters_ziel = {
            "GG": st.slider("GG erneuern ab Alter [a]", 20, 130, 70, 5),
            "PE": st.slider("PE erneuern ab Alter [a]", 20, 130, 90, 5)}
        st.markdown("---")
        rate_strategie = st.checkbox("Zusätzlich: Strategie nach Schadensrate", False)
        krit_rate = st.slider("kritische Schadensrate [1/(km·a)]",
                              0.1, 5.0, 1.0, 0.1, disabled=not rate_strategie)
        st.markdown("---")
        fopt = st.checkbox("Zusätzlich: FOpt (kostenoptimal)", False,
                           help="Erneuern, wenn erwartete Reparaturkosten über den "
                                "Abschreibungszeitraum die Erneuerungskosten übersteigen")
        t_abschr = st.slider("Abschreibungszeitraum (FOpt) [a]",
                             10, 80, 50, 5, disabled=not fopt)
        st.markdown("---")
        cap_aktiv = st.checkbox("Jährliche Erneuerungslänge begrenzen (Budget)", False)
        cap_km = st.slider("max. Erneuerung [km/Jahr]", 0.1, 5.0, 1.0, 0.1,
                           disabled=not cap_aktiv)
        risiko_prio = st.checkbox("Bei Begrenzung: hohe Risiken priorisieren", True,
                                  disabled=not cap_aktiv)
        erneuerungsmaterial = st.selectbox("Erneuerungsmaterial", ["PE", "GG"], 0)

    with sb.expander("💶 Kostenmodell", expanded=False):
        kosten_basis = st.slider("Basiskosten Erneuerung [€/m]", 300, 1500, 700, 10)
        tiefen_zuschlag = st.slider("Zuschlag je Meter Mehrtiefe [Faktor/m]", 0.0, 0.6,
                                    0.15, 0.01, help="Aufschlag pro Meter Tiefe über 1,0 m")
        faktor_innenstadt = st.slider("Faktor Innenstadt-Lage", 1.0, 2.5, 1.4, 0.05)
        kosten_schaden = st.slider("Kosten je Schaden [€]", 5000, 50000, 20000, 1000)
        inflation = st.slider("Inflationsrate [%/a]", 0.0, 6.0, 2.0, 0.5) / 100.0
        diskont = st.slider("Diskontsatz [%/a]", 0.0, 8.0, 3.0, 0.5) / 100.0

    with sb.expander("⚠️ Risiko", expanded=False):
        risiko_schwelle = st.slider("Schwelle 'hohes Risiko' (Risikowert ≥)", 3, 30, 9, 1)

    sb.markdown("---")
    sb.caption(f"Datenbasis: {len(leit)} Leitungen · {len(sch)} Schäden · "
               f"{leit['LNG_km'].sum():.1f} km Gesamtlänge")

    # --- Kostenfaktor je Leitung (Tiefe & Lage) ------------------------------
    tiefe = leit["Verlegetiefe"].fillna(1.0)
    f_tiefe = 1.0 + tiefen_zuschlag * np.maximum(tiefe - 1.0, 0.0)
    f_lage = np.where(leit["Lage"].str.contains("Innenstadt", na=False),
                      faktor_innenstadt, 1.0)
    leit = leit.copy()
    leit["kostenfaktor"] = f_tiefe * f_lage

    # --- Startbestand --------------------------------------------------------
    in_betrieb = (leit["DOI"] <= ref_jahr) & \
                 (leit["DOA"].isna() | (leit["DOA"] > ref_jahr))
    start = leit[in_betrieb].copy()
    start["alter0"] = ref_jahr - start["DOI"]
    start = start[start["alter0"] >= 0]

    # --- Simulationen (Strategie + Baseline 'ohne Erneuerung') ---------------
    sim_args = dict(ref_jahr=ref_jahr, horizont=horizont, alters_ziel=alters_ziel,
                    rate_strategie=rate_strategie, krit_rate=krit_rate,
                    fopt=fopt, t_abschr=t_abschr, cap_aktiv=cap_aktiv, cap_km=cap_km,
                    risiko_prio=risiko_prio, kosten_basis=kosten_basis,
                    kosten_schaden=kosten_schaden, erneuerungsmaterial=erneuerungsmaterial)
    verlauf, end_bestand = simuliere(start, fit, **sim_args)
    verlauf_base, _ = simuliere(start, fit, **sim_args, erneuerung_aktiv=False)

    npv = barwert(verlauf, ref_jahr, inflation, diskont)
    totex = (verlauf["CAPEX"] + verlauf["OPEX"]).sum()
    verm_schaeden = verlauf_base["Schaeden"].sum() - verlauf["Schaeden"].sum()

    # =========================================================================
    #  TABS
    # =========================================================================
    tab_fit, tab_sim, tab_prog, tab_kos, tab_cmp, tab_dat = st.tabs(
        ["📈 Fitting", "📊 Simulation", "🔮 Bruch-Prognose",
         "💶 Kosten & Risiko", "⚖️ Strategievergleich", "🗃️ Daten"])

    # -------------------------------------------------------------------------
    #  TAB 1: FITTING
    # -------------------------------------------------------------------------
    with tab_fit:
        st.subheader("Historische Schäden & kalibrierte Hazard-/Schadensfunktion")
        st.markdown("Punkte = empirische Schadensrate (Größe ∝ Wissenslänge), "
                    "Linie = eingestellte Hazardrate λ(t). **Weibull** ist die "
                    "kanonische Form der Vorlesung; **🎯 Auto-Fit** / **🏆 Bestes "
                    "Modell** kalibrieren automatisch.")
        c1, c2 = st.columns([1, 3])
        klassen_breite = c1.selectbox("Altersklassen-Breite [a]", [1, 2, 5, 10], index=2)
        alter_max_plot = c2.slider("Altersachse bis [a]", 40, ALTER_MAX, 130, 10)
        x_fein = np.arange(0, alter_max_plot + 1)
        farben = {"GG": "#c81e3c", "PE": "#1f77b4"}

        for material in MATERIALIEN:
            punkte = binne_rate(raten[material], klassen_breite)
            punkte = punkte[punkte["Klasse"] <= alter_max_plot]
            w = punkte["Wissenslaenge"].to_numpy(dtype=float)
            wmax = w.max() if len(w) and w.max() > 0 else 1.0
            groesse = 8 + 22 * np.sqrt(w / wmax)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=raten[material]["Alter"], y=raten[material]["Wissenslaenge"],
                        name="Wissenslänge", marker_color="lightgray", opacity=0.5,
                        secondary_y=True)
            fig.add_scatter(x=punkte["Klasse"], y=punkte["Rate"], mode="markers",
                            name="empirische Rate",
                            marker=dict(size=groesse, color=farben[material],
                                        opacity=0.65, line=dict(width=1, color="white")),
                            secondary_y=False)
            fig.add_scatter(x=x_fein, y=schadensrate(material, x_fein, fit), mode="lines",
                            name="kalibrierte λ(t)",
                            line=dict(color=farben[material], width=3), secondary_y=False)
            p = fit[material]
            par = params_text(p)
            fig.update_layout(
                title=f"Materialgruppe {material} – {p['typ']} ({par}) · R² = {p['R2']:.2f}",
                height=380, margin=dict(t=50, b=10),
                legend=dict(orientation="h", y=1.12))
            fig.update_xaxes(title_text="Alter [Jahre]", range=[0, alter_max_plot])
            fig.update_yaxes(title_text="Schadensrate [1/(km·a)]", secondary_y=False,
                             rangemode="tozero")
            fig.update_yaxes(title_text="Wissenslänge [km·a]", secondary_y=True,
                             showgrid=False, rangemode="tozero")
            st.plotly_chart(fig, width="stretch")

    # -------------------------------------------------------------------------
    #  TAB 2: SIMULATION
    # -------------------------------------------------------------------------
    with tab_sim:
        st.subheader(f"Simulierte Bestandsentwicklung {ref_jahr}–{ref_jahr + horizont}")
        na0, na1 = verlauf["Netzalter"].iloc[0], verlauf["Netzalter"].iloc[-1]
        gg0, gg1 = verlauf["GG_anteil"].iloc[0] * 100, verlauf["GG_anteil"].iloc[-1] * 100
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Erneuerung gesamt", f"{verlauf['Erneuerung_km'].sum():.1f} km")
        k2.metric("Erwartete Schäden", f"{verlauf['Schaeden'].sum():.0f}")
        k3.metric("Netzalter (Ende)", f"{na1:.1f} a", f"{na1 - na0:+.1f} a")
        k4.metric("GG-Anteil (Ende)", f"{gg1:.0f} %", f"{gg1 - gg0:+.0f} %")
        k5.metric("TOTEX (real)", f"{totex/1e6:.2f} Mio €")

        fig_b = go.Figure()
        fig_b.add_bar(x=verlauf["Jahr"], y=verlauf["GG"], name="GG (Grauguss)",
                      marker_color="#c81e3c")
        fig_b.add_bar(x=verlauf["Jahr"], y=verlauf["PE"], name="PE (Kunststoff)",
                      marker_color="#1f77b4")
        fig_b.update_layout(barmode="stack", height=420,
                            title="Bestand [km] je Material über die Zeit",
                            xaxis_title="Jahr", yaxis_title="Bestand [km]",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_b, width="stretch")

        col_l, col_r = st.columns(2)
        fig_e = go.Figure(go.Bar(x=verlauf["Jahr"], y=verlauf["Erneuerung_km"],
                                 marker_color="#2ca02c"))
        fig_e.update_layout(title="Jährliche Erneuerungslänge [km]", height=320,
                            xaxis_title="Jahr", yaxis_title="km/Jahr")
        col_l.plotly_chart(fig_e, width="stretch")

        fig_s = go.Figure()
        fig_s.add_scatter(x=verlauf_base["Jahr"], y=verlauf_base["Schaeden"], mode="lines",
                          name="ohne Erneuerung", line=dict(color="#999999", dash="dash"))
        fig_s.add_scatter(x=verlauf["Jahr"], y=verlauf["Schaeden"], mode="lines+markers",
                          name="mit Strategie", line=dict(color="#ff7f0e"))
        fig_s.update_layout(title="Jährliche erwartete Schäden", height=320,
                            xaxis_title="Jahr", yaxis_title="Schäden/Jahr",
                            legend=dict(orientation="h", y=1.15))
        col_r.plotly_chart(fig_s, width="stretch")

        fig_na = go.Figure(go.Scatter(x=verlauf["Jahr"], y=verlauf["Netzalter"],
                                      mode="lines", line=dict(color="#8c564b")))
        fig_na.update_layout(title="Mittleres Netzalter [a]", height=300,
                             xaxis_title="Jahr", yaxis_title="Alter [a]")
        st.plotly_chart(fig_na, width="stretch")

        st.markdown("##### Bestandsverteilung nach Baujahr (Ende des Horizonts)")
        eb = end_bestand.copy(); eb["Baujahr_b"] = (eb["Baujahr"] // 5) * 5
        vert = eb.groupby(["Baujahr_b", "MAT"])["LNG_km"].sum().reset_index()
        fig_v = go.Figure()
        for m, c in [("GG", "#c81e3c"), ("PE", "#1f77b4")]:
            d = vert[vert["MAT"] == m]
            fig_v.add_bar(x=d["Baujahr_b"], y=d["LNG_km"], name=m, marker_color=c)
        fig_v.update_layout(barmode="stack", height=340,
                            xaxis_title="Baujahr (5-Jahres-Klassen)",
                            yaxis_title="Bestand [km]", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_v, width="stretch")

        st.markdown("##### 📝 Zusammenfassung")
        st.info(f"Über **{horizont} Jahre** werden **{verlauf['Erneuerung_km'].sum():.1f} km** "
                f"erneuert. Mittleres Netzalter **{na0:.0f}→{na1:.0f} a**, GG-Anteil "
                f"**{gg0:.0f}→{gg1:.0f} %**. Gegenüber *ohne Erneuerung* werden "
                f"**{verm_schaeden:.0f}** Schäden vermieden (TOTEX real "
                f"**{totex/1e6:.2f} Mio €**, Barwert **{npv/1e6:.2f} Mio €**).")

    # -------------------------------------------------------------------------
    #  TAB 3: BRUCH-PROGNOSE
    # -------------------------------------------------------------------------
    with tab_prog:
        st.subheader("Bruch-Prognose je Leitung (Startbestand)")
        st.markdown("Basis: kumulierte Intensität Λ(t)=∫λ. Erwartete Brüche einer "
                    "Leitung der Länge L in (a, a+H]: **L·(Λ(a+H)−Λ(a))**; "
                    "**P(≥1 Bruch) = 1−e^(−L·ΔΛ)** (Poisson). Das „erwartete nächste "
                    "Bruchjahr“ ist das Jahr, in dem kumuliert **ein** Bruch erwartet wird.")
        prog = bruch_prognose(start, fit, ref_jahr, horizont)

        m1, m2, m3 = st.columns(3)
        m1.metric("Leitungen im Bestand", f"{len(prog)}")
        m2.metric(f"Erw. Brüche gesamt ({horizont} a)",
                  f"{prog['Erw. Brüche (Horizont)'].sum():.0f}")
        bald = prog["Erw. nächstes Bruchjahr"].dropna()
        m3.metric("… davon mit Bruch < 10 a",
                  f"{int((bald <= ref_jahr + 10).sum())}")

        top_n = st.slider("Anzahl Leitungen in Rangliste", 10, 100, 30, 5)
        st.markdown(f"**Top {top_n} Leitungen nach Bruchwahrscheinlichkeit im Horizont**")
        st.dataframe(prog.head(top_n), width="stretch")

        # Netzweite erwartete Brüche je Kalenderjahr (= Schaeden-Verlauf)
        fig_p = go.Figure()
        fig_p.add_scatter(x=verlauf["Jahr"], y=verlauf["Schaeden"].cumsum(),
                          mode="lines", fill="tozeroy", name="mit Strategie",
                          line=dict(color="#d62728"))
        fig_p.add_scatter(x=verlauf_base["Jahr"], y=verlauf_base["Schaeden"].cumsum(),
                          mode="lines", name="ohne Erneuerung",
                          line=dict(color="#999999", dash="dash"))
        fig_p.update_layout(title="Kumulierte erwartete Brüche im Netz", height=340,
                            xaxis_title="Jahr", yaxis_title="Brüche (kumuliert)",
                            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_p, width="stretch")

        st.download_button("⬇️ Bruch-Prognose als CSV",
                           prog.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="bruchprognose.csv", mime="text/csv")

    # -------------------------------------------------------------------------
    #  TAB 4: KOSTEN & RISIKO
    # -------------------------------------------------------------------------
    with tab_kos:
        st.subheader("Kostenentwicklung (CAPEX / OPEX / TOTEX)")
        kk1, kk2, kk3, kk4 = st.columns(4)
        kk1.metric("CAPEX (real)", f"{verlauf['CAPEX'].sum()/1e6:.2f} Mio €")
        kk2.metric("OPEX (real)", f"{verlauf['OPEX'].sum()/1e6:.2f} Mio €")
        kk3.metric("TOTEX (real)", f"{totex/1e6:.2f} Mio €")
        kk4.metric("Barwert (NPV)", f"{npv/1e6:.2f} Mio €",
                   help=f"Inflation {inflation*100:.1f} %, Diskont {diskont*100:.1f} %, "
                        f"Bezug {ref_jahr}")

        fig_k = go.Figure()
        fig_k.add_bar(x=verlauf["Jahr"], y=verlauf["CAPEX"] / 1e6, name="CAPEX",
                      marker_color="#1f77b4")
        fig_k.add_bar(x=verlauf["Jahr"], y=verlauf["OPEX"] / 1e6, name="OPEX",
                      marker_color="#ff7f0e")
        fig_k.update_layout(barmode="stack", height=400, xaxis_title="Jahr",
                            yaxis_title="Kosten [Mio €]", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_k, width="stretch")

        kum = (verlauf["CAPEX"] + verlauf["OPEX"]).cumsum() / 1e6
        kum_base = (verlauf_base["CAPEX"] + verlauf_base["OPEX"]).cumsum() / 1e6
        fig_ck = go.Figure()
        fig_ck.add_scatter(x=verlauf_base["Jahr"], y=kum_base, mode="lines",
                           name="ohne Erneuerung", line=dict(color="#999999", dash="dash"))
        fig_ck.add_scatter(x=verlauf["Jahr"], y=kum, mode="lines", fill="tozeroy",
                           name="mit Strategie", line=dict(color="#9467bd"))
        fig_ck.update_layout(title="Kumulierte TOTEX (real) [Mio €]", height=300,
                             xaxis_title="Jahr", yaxis_title="Mio € (kumuliert)",
                             legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_ck, width="stretch")

        st.markdown("---")
        st.subheader("Risiko-Dashboard (Startbestand)")
        hoch = start[start["Risiko"] >= risiko_schwelle]
        r1, r2, r3 = st.columns(3)
        r1.metric("Risiko-gewichtete Länge",
                  f"{(start['Risiko'].fillna(0) * start['LNG_km']).sum():.0f}",
                  help="Σ Risikowert × Länge[km]")
        r2.metric(f"Hochrisiko-Leitungen (≥{risiko_schwelle})", f"{len(hoch)}",
                  help=f"Länge: {hoch['LNG_km'].sum():.1f} km")
        r3.metric("Ø Risikowert", f"{start['Risiko'].mean():.1f}")

        col_a, col_b = st.columns(2)
        rk = start.groupby("Risiko")["LNG_km"].sum().reset_index()
        fig_rk = go.Figure(go.Bar(x=rk["Risiko"].astype(str), y=rk["LNG_km"],
                                  marker_color="#d62728"))
        fig_rk.update_layout(title="Bestand [km] je Risikoklasse", height=320,
                             xaxis_title="Risikowert (Lage × Bebauung)", yaxis_title="km")
        col_a.plotly_chart(fig_rk, width="stretch")
        fig_rg = go.Figure(go.Scatter(x=verlauf["Jahr"], y=verlauf["Risiko_gew"],
                                      mode="lines+markers", line=dict(color="#d62728")))
        fig_rg.update_layout(title="Risikogewichtete erwartete Schäden über Zeit",
                             height=320, xaxis_title="Jahr",
                             yaxis_title="Σ Risiko × erw. Schäden")
        col_b.plotly_chart(fig_rg, width="stretch")

    # -------------------------------------------------------------------------
    #  TAB 5: STRATEGIEVERGLEICH
    # -------------------------------------------------------------------------
    with tab_cmp:
        st.subheader("Strategievergleich")
        alternative = st.selectbox(
            "Alternativstrategie",
            ["Keine Erneuerung", "GG 20 Jahre früher", "GG 20 Jahre später",
             "FOpt (kostenoptimal)", "Mit Budgetgrenze (1 km/Jahr)"])

        alt_args = dict(sim_args); alt_aktiv = True
        if alternative == "Keine Erneuerung":
            alt_aktiv = False
        elif alternative == "GG 20 Jahre früher":
            az = dict(alters_ziel); az["GG"] = max(20, az["GG"] - 20); alt_args["alters_ziel"] = az
        elif alternative == "GG 20 Jahre später":
            az = dict(alters_ziel); az["GG"] = min(130, az["GG"] + 20); alt_args["alters_ziel"] = az
        elif alternative == "FOpt (kostenoptimal)":
            alt_args["fopt"] = True
        else:
            alt_args["cap_aktiv"] = True; alt_args["cap_km"] = 1.0

        verlauf_alt, _ = simuliere(start, fit, **alt_args, erneuerung_aktiv=alt_aktiv)
        npv_alt = barwert(verlauf_alt, ref_jahr, inflation, diskont)

        tab = pd.DataFrame({
            "Kennzahl": ["Erneuerung [km]", "Erwartete Schäden", "CAPEX [Mio €]",
                         "OPEX [Mio €]", "Barwert/NPV [Mio €]", "Netzalter Ende [a]"],
            "Aktuelle Strategie": [
                round(verlauf["Erneuerung_km"].sum(), 1), round(verlauf["Schaeden"].sum(), 0),
                round(verlauf["CAPEX"].sum()/1e6, 2), round(verlauf["OPEX"].sum()/1e6, 2),
                round(npv/1e6, 2), round(verlauf["Netzalter"].iloc[-1], 1)],
            alternative: [
                round(verlauf_alt["Erneuerung_km"].sum(), 1), round(verlauf_alt["Schaeden"].sum(), 0),
                round(verlauf_alt["CAPEX"].sum()/1e6, 2), round(verlauf_alt["OPEX"].sum()/1e6, 2),
                round(npv_alt/1e6, 2), round(verlauf_alt["Netzalter"].iloc[-1], 1)]})
        st.table(tab)

        col_x, col_y = st.columns(2)
        kum_a = (verlauf["CAPEX"] + verlauf["OPEX"]).cumsum() / 1e6
        kum_b = (verlauf_alt["CAPEX"] + verlauf_alt["OPEX"]).cumsum() / 1e6
        fig_cx = go.Figure()
        fig_cx.add_scatter(x=verlauf["Jahr"], y=kum_a, mode="lines", name="Aktuell",
                           line=dict(color="#9467bd"))
        fig_cx.add_scatter(x=verlauf_alt["Jahr"], y=kum_b, mode="lines", name=alternative,
                           line=dict(color="#2ca02c", dash="dash"))
        fig_cx.update_layout(title="Kumulierte TOTEX [Mio €]", height=340,
                             xaxis_title="Jahr", yaxis_title="Mio €",
                             legend=dict(orientation="h", y=1.15))
        col_x.plotly_chart(fig_cx, width="stretch")
        fig_sy = go.Figure()
        fig_sy.add_scatter(x=verlauf["Jahr"], y=verlauf["Schaeden"], mode="lines",
                           name="Aktuell", line=dict(color="#ff7f0e"))
        fig_sy.add_scatter(x=verlauf_alt["Jahr"], y=verlauf_alt["Schaeden"], mode="lines",
                           name=alternative, line=dict(color="#1f77b4", dash="dash"))
        fig_sy.update_layout(title="Jährliche erwartete Schäden", height=340,
                             xaxis_title="Jahr", yaxis_title="Schäden/Jahr",
                             legend=dict(orientation="h", y=1.15))
        col_y.plotly_chart(fig_sy, width="stretch")

        d_npv = npv_alt - npv; d_sch = verlauf_alt["Schaeden"].sum() - verlauf["Schaeden"].sum()
        st.info(f"**Fazit:** *{alternative}* verändert den Barwert um "
                f"**{d_npv/1e6:+.2f} Mio €** und die erwarteten Schäden um **{d_sch:+.0f}**. "
                + ("Günstiger **und** schadensärmer." if d_npv > 0 and d_sch > 0 else
                   "Günstiger, aber mehr Schäden – Abwägung." if d_npv > 0 else
                   "Teurer, dafür weniger Schäden – höheres Schutzniveau." if d_sch < 0 else
                   "Teurer und mehr Schäden – aktuelle Strategie überlegen."))

    # -------------------------------------------------------------------------
    #  TAB 6: DATEN
    # -------------------------------------------------------------------------
    with tab_dat:
        st.subheader("Eingangsdaten (bereinigt)")
        st.markdown(f"**Leitungen** – {len(leit)} Zeilen, davon "
                    f"{int(in_betrieb.sum())} zum Referenzjahr {ref_jahr} in Betrieb")
        st.dataframe(leit.head(50), width="stretch")
        st.markdown(f"**Schäden** – {len(sch)} Ereignisse "
                    f"({int(sch['Alter'].notna().sum())} mit gültigem Schadensalter)")
        st.dataframe(sch.head(50), width="stretch")

        st.markdown("##### Empirische Eckwerte & kalibrierte Parameter je Material")
        zus = []
        for m in MATERIALIEN:
            df_m = raten[m]; p = fit[m]
            par = params_text(p)
            zus.append({"Material": m,
                        "Σ Wissenslänge [km·a]": round(df_m["Wissenslaenge"].sum(), 1),
                        "Σ Schäden": int(df_m["Schaeden"].sum()),
                        "Ø Rate [1/(km·a)]":
                            round(df_m["Schaeden"].sum() / df_m["Wissenslaenge"].sum(), 3),
                        "Modell": p["typ"], "Parameter": par, "R²": round(p["R2"], 2)})
        st.table(pd.DataFrame(zus))

        st.download_button("⬇️ Simulationsverlauf als CSV",
                           verlauf.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="simulationsverlauf.csv", mime="text/csv")


if __name__ == "__main__":
    main()
