# =============================================================================
#  Asset Simulationstool fuer Versorgungsnetze  (angelehnt an KANEW 3S)
#  -----------------------------------------------------------------------------
#  Einzeldatei-Streamlit-App. Starten mit:   streamlit run app.py
#
#  Funktionsumfang:
#   1) Automatischer Import & Bereinigung der beiden CSV-Dateien
#   2) Manuelles Fitting material-/altersabhaengiger Schadensfunktionen
#      (Echtzeit-Overlay ueber die historischen Schaeden)
#   3) Asset-Simulation mit Zielwertstrategie (Alter) + optional Schadensrate
#   4) Detailliertes Kostenmodell (Verlegetiefe & Lage) + Risiko-Dashboard
#
#  Benoetigte Pakete:  streamlit, pandas, numpy, plotly
#  Installation:       pip install streamlit pandas numpy plotly
# =============================================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
#  Grundkonfiguration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Asset Simulation Versorgungsnetz",
                   page_icon="🛠️", layout="wide")

# Dateinamen der Eingangsdaten (liegen im gleichen Verzeichnis wie app.py
# bzw. werden bei der Browser-Variante in das virtuelle Dateisystem gelegt)
def _finde_csv(dateiname):
    """Sucht eine CSV-Datei an den ueblichen Orten und gibt den ersten
    existierenden Pfad zurueck. Funktioniert lokal (streamlit run) UND im
    Browser (stlite legt die Dateien ins Arbeitsverzeichnis)."""
    kandidaten = [dateiname, os.path.join(os.getcwd(), dateiname)]
    try:  # __file__ existiert lokal, im Browser ggf. nicht
        kandidaten.insert(1, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), dateiname))
    except NameError:
        pass
    for p in kandidaten:
        if os.path.exists(p):
            return p
    return dateiname  # Fallback: pandas wirft sonst eine klare Fehlermeldung

DATEI_LEIT   = "Leitungen_Alle.csv"
DATEI_SCHAED = "Schäden.csv"

# Beobachtungszeitraum der historischen Schadensdaten (fuer die "Wissenslaenge")
BEOB_START, BEOB_ENDE = 1960, 2017
MATERIALIEN = ["GG", "PE"]          # GG = Grauguss, PE = Kunststoff
ALTER_MAX   = 130                   # maximales betrachtetes Leitungsalter [Jahre]


# =============================================================================
#  1) DATEN-IMPORT & BEREINIGUNG
# =============================================================================
@st.cache_data(show_spinner=False)
def lade_daten():
    """Liest beide CSV-Dateien ein und bereinigt sie.

    Besonderheiten der Rohdaten:
      - Semikolon als Trennzeichen
      - deutsches Zahlenformat (Komma statt Punkt)  -> decimal=','
      - UTF-8 mit BOM                               -> encoding='utf-8-sig'
      - leere Zusatzspalten am Zeilenende           -> 'Unnamed'-Spalten verwerfen
    """
    # --- Leitungsdaten ---------------------------------------------------------
    leit = pd.read_csv(_finde_csv(DATEI_LEIT), sep=";", decimal=",", encoding="utf-8-sig")
    leit.columns = [c.strip() for c in leit.columns]                 # Leerzeichen entfernen
    leit = leit[[c for c in leit.columns if not c.startswith("Unnamed")]]

    # Spalten mit Sonderzeichen auf kurze, sichere Namen umbenennen
    leit = leit.rename(columns={
        "Risikobewertung_Städtische Lage*Bebauung": "Risiko",
        "Städtischen Lage": "Lage",
    })

    # Textspalten von ueberfluessigen Leerzeichen befreien
    for c in ["MAT", "Lage", "Bebauung", "MAT_zusammengefasst"]:
        if c in leit.columns:
            leit[c] = leit[c].astype(str).str.strip()

    # Numerische Spalten sicher in Zahlen wandeln
    for c in ["DOI", "DN", "DOA", "LNG", "Druck", "Verlegetiefe", "Risiko"]:
        leit[c] = pd.to_numeric(leit[c], errors="coerce")

    # Laenge zusaetzlich in km bereitstellen
    leit["LNG_km"] = leit["LNG"] / 1000.0

    # --- Schadensdaten ---------------------------------------------------------
    sch = pd.read_csv(_finde_csv(DATEI_SCHAED), sep=";", decimal=",", encoding="utf-8-sig")
    sch.columns = [c.strip() for c in sch.columns]
    sch = sch[[c for c in sch.columns if not c.startswith("Unnamed")]]
    sch = sch.rename(columns={"Alter bei Schadenseintritt DOE3": "Alter"})

    for c in ["Material", "Schadensart", "Schadensursache"]:
        if c in sch.columns:
            sch[c] = sch[c].astype(str).str.strip()

    for c in ["DOI", "DOA", "DOE3", "Alter"]:
        sch[c] = pd.to_numeric(sch[c], errors="coerce")

    return leit, sch


# =============================================================================
#  EMPIRISCHE SCHADENSRATE  (Grundlage fuer den Scatterplot)
# =============================================================================
@st.cache_data(show_spinner=False)
def empirische_rate(leit, sch):
    """Berechnet je Material und Leitungsalter:
        - Wissenslaenge  (Exposition in km*Jahren)
        - Anzahl Schaeden
        - empirische Schadensrate = Schaeden / Wissenslaenge  [Ereignisse/(km*a)]

    Idee der Wissenslaenge: Eine Leitung (Baujahr DOI, Laenge L) "war im Alter a"
    im Kalenderjahr DOI+a. Faellt dieses Jahr in den Beobachtungszeitraum und war
    die Leitung da in Betrieb, traegt sie L (in km) zur Exposition im Alter a bei.
    """
    alter = np.arange(0, ALTER_MAX + 1)
    ergebnis = {}

    for mat in MATERIALIEN:
        teil = leit[leit["MAT"] == mat]
        exposition = np.zeros(len(alter))                # km*Jahre je Alter

        doi   = teil["DOI"].to_numpy()
        doa   = teil["DOA"].to_numpy()
        L_km  = teil["LNG_km"].to_numpy()
        # Betriebsende = Jahr der Ausserbetriebnahme, sonst Ende Beobachtung
        serv_ende = np.where(np.isnan(doa), BEOB_ENDE, doa)

        for a in alter:
            kalender = doi + a
            aktiv = (kalender >= BEOB_START) & (kalender <= BEOB_ENDE) & \
                    (kalender >= doi) & (kalender <= serv_ende)
            exposition[a] = L_km[aktiv].sum()

        # Nur echte Schaeden mit gueltigem Alter (Ausserbetriebnahmen sind
        # in den Rohdaten ohne Alter codiert und zaehlen hier nicht als Schaden)
        dmg = sch[(sch["Material"] == mat) & sch["Alter"].notna()]
        anzahl = np.array([(dmg["Alter"] == a).sum() for a in alter], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(exposition > 0, anzahl / exposition, np.nan)

        ergebnis[mat] = pd.DataFrame({
            "Alter": alter, "Wissenslaenge": exposition,
            "Schaeden": anzahl, "Rate": rate,
        })
    return ergebnis


def binne_rate(df_rate, breite):
    """Fasst die alters-feine empirische Rate in Altersklassen zusammen
    (ruhigere Punkte fuer den Scatterplot). Rate je Klasse = Summe Schaeden /
    Summe Wissenslaenge. Punktgroesse spaeter ~ Wissenslaenge (Konfidenz)."""
    df = df_rate.copy()
    df["Klasse"] = (df["Alter"] // breite) * breite + breite / 2.0   # Klassenmitte
    g = df.groupby("Klasse").agg(Schaeden=("Schaeden", "sum"),
                                 Wissenslaenge=("Wissenslaenge", "sum")).reset_index()
    g = g[g["Wissenslaenge"] > 0]
    g["Rate"] = g["Schaeden"] / g["Wissenslaenge"]
    return g


# =============================================================================
#  SCHADENSFUNKTIONEN  (theoretische Kurven fuer das manuelle Fitting)
# =============================================================================
def sr_exponential(alter, a, b):
    """Exponentielle Schadensrate:  SR(t) = a * exp(b * t)
    a = Rate bei Alter 0,  b = Wachstum.  -> KANEW-Typ 'Exponential'."""
    return a * np.exp(b * np.asarray(alter, dtype=float))


def sr_weibull(alter, skala, form, faktor):
    """Weibull-Hazardrate:  SR(t) = faktor * (form/skala) * (t/skala)^(form-1)
    form > 1  -> mit dem Alter steigende Rate (Verschleiss)."""
    t = np.maximum(np.asarray(alter, dtype=float), 1e-6)
    return faktor * (form / skala) * (t / skala) ** (form - 1.0)


def schadensrate(material, alter, fit):
    """Dispatcher: liefert die SR fuer ein Material anhand der aktuellen
    Slider-Parameter (Dictionary 'fit')."""
    p = fit[material]
    if p["typ"] == "Exponential":
        return np.clip(sr_exponential(alter, p["a"], p["b"]), 0, None)
    else:
        return np.clip(sr_weibull(alter, p["skala"], p["form"], p["faktor"]), 0, None)


# =============================================================================
#  3)+4) ASSET-SIMULATION  (deterministisch, vektorisiert)
# =============================================================================
def simuliere(start, fit, ref_jahr, horizont,
              alters_ziel, rate_strategie, krit_rate,
              cap_aktiv, cap_km, risiko_prio,
              kosten_basis, tiefen_zuschlag, faktor_innenstadt,
              kosten_schaden, erneuerungsmaterial):
    """Simuliert die Bestandsentwicklung Jahr fuer Jahr in die Zukunft.

    Pro Jahr:
      1. alle Leitungen altern um 1 Jahr
      2. erwartete Schaeden = SR(Alter) * Laenge_km   (Erwartungswert je Leitung)
      3. Erneuerungskandidaten bestimmen (Alters-Zielwert, optional Schadensrate)
      4. optional jaehrliche Laengenbegrenzung -> Priorisierung nach Risiko
      5. erneuerte Leitungen: Material -> Erneuerungsmaterial, Alter -> 0
      6. Kosten: CAPEX (Erneuerung, detailliert) und OPEX (Schaeden)

    'start' ist der zum Referenzjahr in Betrieb befindliche Bestand.
    """
    # Arbeits-Arrays (werden ueber die Jahre veraendert)
    mat   = start["MAT"].to_numpy().copy()              # aktuelles Material
    alter = start["alter0"].to_numpy().astype(float).copy()
    L_km  = start["LNG_km"].to_numpy()
    L_m   = start["LNG"].to_numpy()
    risiko = start["Risiko"].fillna(0).to_numpy()
    byear = start["DOI"].to_numpy().astype(float).copy()  # Baujahr (fuer Verteilung)
    # detaillierte Erneuerungskosten je Meter = Basis * Tiefenfaktor * Lagefaktor
    eur_pro_m = start["kostenfaktor"].to_numpy() * kosten_basis

    zeilen = []
    # Startzeile (Referenzjahr) als Ausgangsbestand
    zeilen.append({"Jahr": ref_jahr,
                   "GG": L_km[mat == "GG"].sum(), "PE": L_km[mat == "PE"].sum(),
                   "Erneuerung_km": 0.0, "Schaeden": 0.0,
                   "CAPEX": 0.0, "OPEX": 0.0, "Risiko_gew": 0.0})

    for jahr in range(ref_jahr + 1, ref_jahr + horizont + 1):
        alter += 1.0

        # --- Schadensrate je Leitung (nach aktuellem Material) -----------------
        sr = np.zeros(len(start))
        for m in MATERIALIEN:
            sel = mat == m
            if sel.any():
                sr[sel] = schadensrate(m, alter[sel], fit)

        erw_schaeden = sr * L_km                    # erwartete Schaeden je Leitung
        schaeden_jahr = erw_schaeden.sum()
        opex = schaeden_jahr * kosten_schaden
        risiko_gew = (erw_schaeden * risiko).sum()  # risikogewichtete Schaeden

        # --- Erneuerungskandidaten --------------------------------------------
        kandidat = np.zeros(len(start), dtype=bool)
        # Pflicht: Zielwertstrategie auf Basis des Alters (je Material)
        for m in MATERIALIEN:
            kandidat |= (mat == m) & (alter >= alters_ziel[m])
        # Optional: Erneuerung bei Ueberschreiten einer kritischen Schadensrate
        if rate_strategie:
            kandidat |= sr >= krit_rate

        idx = np.where(kandidat)[0]

        # --- Optional: jaehrliche Laengenbegrenzung mit Risiko-Prioritaet -----
        if cap_aktiv and len(idx) > 0 and L_km[idx].sum() > cap_km:
            if risiko_prio:
                # zuerst hohes Risiko, bei Gleichstand hoeheres Alter
                ordnung = np.lexsort((-alter[idx], -risiko[idx]))
            else:
                ordnung = np.argsort(-alter[idx])      # nur nach Alter
            idx_sort = idx[ordnung]
            kum = np.cumsum(L_km[idx_sort])
            idx = idx_sort[kum <= cap_km]

        ern_laenge = L_km[idx].sum()
        capex = (L_m[idx] * eur_pro_m[idx]).sum()

        # --- Erneuerung anwenden ----------------------------------------------
        mat[idx]   = erneuerungsmaterial
        alter[idx] = 0.0
        byear[idx] = jahr

        # --- Bestand je Material am Jahresende ---------------------------------
        zeilen.append({"Jahr": jahr,
                       "GG": L_km[mat == "GG"].sum(), "PE": L_km[mat == "PE"].sum(),
                       "Erneuerung_km": ern_laenge, "Schaeden": schaeden_jahr,
                       "CAPEX": capex, "OPEX": opex, "Risiko_gew": risiko_gew})

    verlauf = pd.DataFrame(zeilen)
    # Endbestand fuer die Baujahr-Verteilung (mirror der KANEW-Bestandsverteilung)
    end_bestand = pd.DataFrame({"Baujahr": byear, "MAT": mat, "LNG_km": L_km})
    return verlauf, end_bestand


# =============================================================================
#  APP-OBERFLAECHE
# =============================================================================
def main():
    st.title("🛠️ Asset Simulationstool für Versorgungsnetze")
    st.caption("Manuelles Fitting von Schadensfunktionen · Zielwertstrategie · "
               "detailliertes Kosten- & Risikomodell — angelehnt an KANEW 3S")

    # Daten laden -------------------------------------------------------------
    try:
        leit, sch = lade_daten()
    except FileNotFoundError:
        st.error("CSV-Dateien nicht gefunden. Bitte 'Leitungen_Alle.csv' und "
                 "'Schäden.csv' in dasselbe Verzeichnis wie app.py legen.")
        st.stop()

    raten = empirische_rate(leit, sch)

    # =========================================================================
    #  SEITENLEISTE  –  alle Steuerparameter
    # =========================================================================
    sb = st.sidebar
    sb.header("⚙️ Steuerung")

    # --- Simulationsrahmen ---------------------------------------------------
    with sb.expander("📅 Simulationsrahmen", expanded=True):
        ref_jahr = st.number_input("Referenzjahr (Start)", 2000, 2030, 2018, 1)
        horizont = st.slider("Zeithorizont [Jahre]", 10, 80, 50, 5)

    # --- Schadensfunktionen je Material (manuelles Fitting) ------------------
    def fit_block(material, default):
        """Erzeugt die Slider fuer eine Materialgruppe und gibt die Parameter
        als Dictionary zurueck."""
        with sb.expander(f"📈 Schadensfunktion {material}", expanded=(material == "GG")):
            typ = st.selectbox("Funktionstyp", ["Exponential", "Weibull"],
                               key=f"typ_{material}",
                               index=0 if default["typ"] == "Exponential" else 1)
            if typ == "Exponential":
                a = st.slider(f"a (Rate bei Alter 0) – {material}",
                              0.0, 2.0, default["a"], 0.005, key=f"a_{material}")
                b = st.slider(f"b (Wachstum) – {material}",
                              0.0, 0.12, default["b"], 0.001, key=f"b_{material}")
                return {"typ": typ, "a": a, "b": b}
            else:
                skala = st.slider(f"Skala η – {material}",
                                  10.0, 200.0, default.get("skala", 90.0), 1.0,
                                  key=f"sk_{material}")
                form = st.slider(f"Form β – {material}",
                                 0.5, 6.0, default.get("form", 2.5), 0.1,
                                 key=f"fo_{material}")
                faktor = st.slider(f"Faktor – {material}",
                                   0.1, 5.0, default.get("faktor", 1.0), 0.1,
                                   key=f"fa_{material}")
                return {"typ": typ, "skala": skala, "form": form, "faktor": faktor}

    # Sinnvolle Startwerte (grob an die empirischen Raten angelehnt)
    fit = {
        "GG": fit_block("GG", {"typ": "Exponential", "a": 0.02, "b": 0.045}),
        "PE": fit_block("PE", {"typ": "Exponential", "a": 0.15, "b": 0.010}),
    }

    # --- Erneuerungsstrategie ------------------------------------------------
    with sb.expander("🔧 Erneuerungsstrategie", expanded=True):
        st.markdown("**Zielwertstrategie (Alter)** – Pflicht")
        alters_ziel = {
            "GG": st.slider("GG erneuern ab Alter [a]", 20, 130, 70, 5),
            "PE": st.slider("PE erneuern ab Alter [a]", 20, 130, 90, 5),
        }
        st.markdown("---")
        rate_strategie = st.checkbox("Zusätzlich: Strategie nach Schadensrate", False)
        krit_rate = st.slider("kritische Schadensrate [1/(km·a)]",
                              0.1, 5.0, 1.0, 0.1, disabled=not rate_strategie)
        st.markdown("---")
        cap_aktiv = st.checkbox("Jährliche Erneuerungslänge begrenzen (Budget)", False)
        cap_km = st.slider("max. Erneuerung [km/Jahr]",
                           0.1, 5.0, 1.0, 0.1, disabled=not cap_aktiv)
        risiko_prio = st.checkbox("Bei Begrenzung: hohe Risiken priorisieren",
                                  True, disabled=not cap_aktiv)
        erneuerungsmaterial = st.selectbox("Erneuerungsmaterial", ["PE", "GG"], 0)

    # --- Detailliertes Kostenmodell ------------------------------------------
    with sb.expander("💶 Kostenmodell", expanded=False):
        kosten_basis = st.slider("Basiskosten Erneuerung [€/m]", 300, 1500, 700, 10)
        tiefen_zuschlag = st.slider("Zuschlag je Meter Mehrtiefe [Faktor/m]",
                                    0.0, 0.6, 0.15, 0.01,
                                    help="Aufschlag pro Meter Verlegetiefe über 1,0 m")
        faktor_innenstadt = st.slider("Faktor Innenstadt-Lage", 1.0, 2.5, 1.4, 0.05,
                                      help="Stadtrand = 1,0; Innenstadt teurer")
        kosten_schaden = st.slider("Kosten je Schaden [€]", 5000, 50000, 20000, 1000)

    # --- Risiko --------------------------------------------------------------
    with sb.expander("⚠️ Risiko", expanded=False):
        risiko_schwelle = st.slider("Schwelle 'hohes Risiko' (Risikowert ≥)",
                                    3, 30, 9, 1)

    sb.markdown("---")
    sb.caption(f"Datenbasis: {len(leit)} Leitungen · {len(sch)} Schäden · "
               f"{leit['LNG_km'].sum():.1f} km Gesamtlänge")

    # =========================================================================
    #  DETAILLIERTER KOSTENFAKTOR JE LEITUNG  (Tiefe & Lage)
    # =========================================================================
    # Faktor = (1 + Zuschlag*(Tiefe-1)) * (Innenstadt? Faktor : 1)
    tiefe = leit["Verlegetiefe"].fillna(1.0)
    f_tiefe = 1.0 + tiefen_zuschlag * np.maximum(tiefe - 1.0, 0.0)
    f_lage = np.where(leit["Lage"].str.contains("Innenstadt", na=False),
                      faktor_innenstadt, 1.0)
    leit = leit.copy()
    leit["kostenfaktor"] = f_tiefe * f_lage

    # =========================================================================
    #  STARTBESTAND  (zum Referenzjahr in Betrieb befindliche Leitungen)
    # =========================================================================
    # In Betrieb = gebaut bis Referenzjahr UND (keine Ausserbetriebnahme
    # oder Ausserbetriebnahme nach Referenzjahr)
    in_betrieb = (leit["DOI"] <= ref_jahr) & \
                 (leit["DOA"].isna() | (leit["DOA"] > ref_jahr))
    start = leit[in_betrieb].copy()
    start["alter0"] = ref_jahr - start["DOI"]
    start = start[start["alter0"] >= 0]

    # =========================================================================
    #  TABS
    # =========================================================================
    tab_fit, tab_sim, tab_kos, tab_dat = st.tabs(
        ["📈 Manuelles Fitting", "📊 Asset-Simulation",
         "💶 Kosten & Risiko", "🗃️ Daten"])

    # -------------------------------------------------------------------------
    #  TAB 1: MANUELLES FITTING
    # -------------------------------------------------------------------------
    with tab_fit:
        st.subheader("Historische Schäden & manuelles Fitting der Schadensfunktion")
        st.markdown("Die **Punkte** zeigen die empirische Schadensrate aus den "
                    "historischen Daten (Größe ∝ Wissenslänge/Konfidenz), die "
                    "**Linie** die per Slider eingestellte theoretische Funktion. "
                    "Bewege die Slider in der Seitenleiste – die Kurve aktualisiert sich live.")

        c1, c2 = st.columns([1, 3])
        klassen_breite = c1.selectbox("Altersklassen-Breite [a]", [1, 2, 5, 10], index=2)
        alter_max_plot = c2.slider("Altersachse bis [a]", 40, ALTER_MAX, 130, 10)

        x_fein = np.arange(0, alter_max_plot + 1)
        farben = {"GG": "#c81e3c", "PE": "#1f77b4"}

        for material in MATERIALIEN:
            punkte = binne_rate(raten[material], klassen_breite)
            punkte = punkte[punkte["Klasse"] <= alter_max_plot]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # Wissenslaenge als blasse Balken im Hintergrund (Sekundaerachse)
            fig.add_bar(x=raten[material]["Alter"], y=raten[material]["Wissenslaenge"],
                        name="Wissenslänge", marker_color="lightgray",
                        opacity=0.5, secondary_y=True)
            # empirische Raten-Punkte
            fig.add_scatter(x=punkte["Klasse"], y=punkte["Rate"], mode="markers",
                            name="empirische Rate",
                            marker=dict(size=np.sqrt(punkte["Wissenslaenge"]) * 6 + 5,
                                        color=farben[material], opacity=0.65,
                                        line=dict(width=1, color="white")),
                            secondary_y=False)
            # theoretische, gefittete Kurve
            y_fit = schadensrate(material, x_fein, fit)
            fig.add_scatter(x=x_fein, y=y_fit, mode="lines", name="gefittete Funktion",
                            line=dict(color=farben[material], width=3),
                            secondary_y=False)

            typ = fit[material]["typ"]
            fig.update_layout(title=f"Materialgruppe {material} – Funktionstyp: {typ}",
                              height=380, margin=dict(t=50, b=10),
                              legend=dict(orientation="h", y=1.12))
            fig.update_xaxes(title_text="Alter [Jahre]", range=[0, alter_max_plot])
            fig.update_yaxes(title_text="Schadensrate [1/(km·a)]", secondary_y=False,
                             rangemode="tozero")
            fig.update_yaxes(title_text="Wissenslänge [km·a]", secondary_y=True,
                             showgrid=False, rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    #  SIMULATION AUSFUEHREN (Ergebnis fuer Tab 2 und 3)
    # -------------------------------------------------------------------------
    verlauf, end_bestand = simuliere(
        start, fit, ref_jahr, horizont,
        alters_ziel, rate_strategie, krit_rate,
        cap_aktiv, cap_km, risiko_prio,
        kosten_basis, tiefen_zuschlag, faktor_innenstadt,
        kosten_schaden, erneuerungsmaterial)

    # -------------------------------------------------------------------------
    #  TAB 2: ASSET-SIMULATION
    # -------------------------------------------------------------------------
    with tab_sim:
        st.subheader(f"Simulierte Bestandsentwicklung {ref_jahr}–{ref_jahr + horizont}")

        # Kennzahlen
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Erneuerung gesamt", f"{verlauf['Erneuerung_km'].sum():.1f} km")
        k2.metric("Erwartete Schäden", f"{verlauf['Schaeden'].sum():.0f}")
        k3.metric("CAPEX gesamt", f"{verlauf['CAPEX'].sum()/1e6:.2f} Mio €")
        k4.metric("OPEX gesamt", f"{verlauf['OPEX'].sum()/1e6:.2f} Mio €")

        # Gestapeltes Saeulendiagramm: Bestand [km] je Material ueber die Zeit
        fig_b = go.Figure()
        fig_b.add_bar(x=verlauf["Jahr"], y=verlauf["GG"], name="GG (Grauguss)",
                      marker_color="#c81e3c")
        fig_b.add_bar(x=verlauf["Jahr"], y=verlauf["PE"], name="PE (Kunststoff)",
                      marker_color="#1f77b4")
        fig_b.update_layout(barmode="stack", height=420,
                            title="Bestand [km] je Material über die Zeit",
                            xaxis_title="Jahr", yaxis_title="Bestand [km]",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_b, use_container_width=True)

        col_l, col_r = st.columns(2)
        # Jaehrliche Erneuerungslaenge
        fig_e = go.Figure(go.Bar(x=verlauf["Jahr"], y=verlauf["Erneuerung_km"],
                                 marker_color="#2ca02c"))
        fig_e.update_layout(title="Jährliche Erneuerungslänge [km]", height=320,
                            xaxis_title="Jahr", yaxis_title="km/Jahr")
        col_l.plotly_chart(fig_e, use_container_width=True)

        # Jaehrliche (erwartete) Schaeden
        fig_s = go.Figure(go.Scatter(x=verlauf["Jahr"], y=verlauf["Schaeden"],
                                     mode="lines+markers", line=dict(color="#ff7f0e")))
        fig_s.update_layout(title="Jährliche erwartete Schäden", height=320,
                            xaxis_title="Jahr", yaxis_title="Schäden/Jahr")
        col_r.plotly_chart(fig_s, use_container_width=True)

        # Bestandsverteilung nach Baujahr am Ende (mirror KANEW)
        st.markdown("##### Bestandsverteilung nach Baujahr (Ende des Horizonts)")
        eb = end_bestand.copy()
        eb["Baujahr_b"] = (eb["Baujahr"] // 5) * 5
        vert = eb.groupby(["Baujahr_b", "MAT"])["LNG_km"].sum().reset_index()
        fig_v = go.Figure()
        for m, c in [("GG", "#c81e3c"), ("PE", "#1f77b4")]:
            d = vert[vert["MAT"] == m]
            fig_v.add_bar(x=d["Baujahr_b"], y=d["LNG_km"], name=m, marker_color=c)
        fig_v.update_layout(barmode="stack", height=340,
                            xaxis_title="Baujahr (5-Jahres-Klassen)",
                            yaxis_title="Bestand [km]",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_v, use_container_width=True)

    # -------------------------------------------------------------------------
    #  TAB 3: KOSTEN & RISIKO
    # -------------------------------------------------------------------------
    with tab_kos:
        st.subheader("Kostenentwicklung (CAPEX / OPEX)")

        fig_k = go.Figure()
        fig_k.add_bar(x=verlauf["Jahr"], y=verlauf["CAPEX"] / 1e6,
                      name="CAPEX (Erneuerung)", marker_color="#1f77b4")
        fig_k.add_bar(x=verlauf["Jahr"], y=verlauf["OPEX"] / 1e6,
                      name="OPEX (Schäden)", marker_color="#ff7f0e")
        fig_k.update_layout(barmode="stack", height=400,
                            xaxis_title="Jahr", yaxis_title="Kosten [Mio €]",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_k, use_container_width=True)

        # Kumulierte Kosten
        verlauf2 = verlauf.copy()
        verlauf2["Kosten_kum"] = (verlauf2["CAPEX"] + verlauf2["OPEX"]).cumsum() / 1e6
        fig_ck = go.Figure(go.Scatter(x=verlauf2["Jahr"], y=verlauf2["Kosten_kum"],
                                      mode="lines", fill="tozeroy",
                                      line=dict(color="#9467bd")))
        fig_ck.update_layout(title="Kumulierte Gesamtkosten [Mio €]", height=300,
                             xaxis_title="Jahr", yaxis_title="Mio € (kumuliert)")
        st.plotly_chart(fig_ck, use_container_width=True)

        st.markdown("---")
        st.subheader("Risiko-Dashboard (Startbestand)")

        # Risiko-Kennzahlen des aktuellen Startbestands
        hoch = start[start["Risiko"] >= risiko_schwelle]
        r1, r2, r3 = st.columns(3)
        r1.metric("Risiko-gewichtete Länge",
                  f"{(start['Risiko'].fillna(0) * start['LNG_km']).sum():.0f}",
                  help="Σ Risikowert × Länge[km]")
        r2.metric(f"Hochrisiko-Leitungen (≥{risiko_schwelle})",
                  f"{len(hoch)}", help=f"Länge: {hoch['LNG_km'].sum():.1f} km")
        r3.metric("Ø Risikowert", f"{start['Risiko'].mean():.1f}")

        col_a, col_b = st.columns(2)
        # Verteilung der Risikoklassen im Bestand
        rk = start.groupby("Risiko")["LNG_km"].sum().reset_index()
        fig_rk = go.Figure(go.Bar(x=rk["Risiko"].astype(str), y=rk["LNG_km"],
                                  marker_color="#d62728"))
        fig_rk.update_layout(title="Bestand [km] je Risikoklasse", height=320,
                             xaxis_title="Risikowert (Lage × Bebauung)",
                             yaxis_title="km")
        col_a.plotly_chart(fig_rk, use_container_width=True)

        # Entwicklung der risikogewichteten Schaeden ueber die Zeit
        fig_rg = go.Figure(go.Scatter(x=verlauf["Jahr"], y=verlauf["Risiko_gew"],
                                      mode="lines+markers", line=dict(color="#d62728")))
        fig_rg.update_layout(title="Risikogewichtete erwartete Schäden über Zeit",
                             height=320, xaxis_title="Jahr",
                             yaxis_title="Σ Risiko × erw. Schäden")
        col_b.plotly_chart(fig_rg, use_container_width=True)

        st.info("💡 Mit aktivierter Budget-Begrenzung **und** Risiko-Priorität werden "
                "hochriskante Leitungen zuerst erneuert – die risikogewichteten "
                "Schäden sinken dann schneller.")

    # -------------------------------------------------------------------------
    #  TAB 4: DATEN
    # -------------------------------------------------------------------------
    with tab_dat:
        st.subheader("Eingangsdaten (bereinigt)")
        st.markdown(f"**Leitungen** – {len(leit)} Zeilen, davon "
                    f"{int(in_betrieb.sum())} zum Referenzjahr {ref_jahr} in Betrieb")
        st.dataframe(leit.head(50), use_container_width=True)
        st.markdown(f"**Schäden** – {len(sch)} Ereignisse "
                    f"({int(sch['Alter'].notna().sum())} mit gültigem Schadensalter)")
        st.dataframe(sch.head(50), use_container_width=True)

        st.markdown("##### Empirische Eckwerte je Material")
        zus = []
        for m in MATERIALIEN:
            df_m = raten[m]
            zus.append({"Material": m,
                        "Σ Wissenslänge [km·a]": round(df_m["Wissenslaenge"].sum(), 1),
                        "Σ Schäden (gültig)": int(df_m["Schaeden"].sum()),
                        "Ø Rate [1/(km·a)]":
                            round(df_m["Schaeden"].sum() / df_m["Wissenslaenge"].sum(), 3)})
        st.table(pd.DataFrame(zus))

        # Download des Simulationsverlaufs
        st.download_button("⬇️ Simulationsverlauf als CSV",
                           verlauf.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="simulationsverlauf.csv", mime="text/csv")


if __name__ == "__main__":
    main()
