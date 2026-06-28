# Projektdokumentation – Asset-Simulationstool für Versorgungsnetze

**Kurs:** Asset Management für Versorgungsnetze (Prof. Dr. David Echternacht, HSD)
**Praxisteil:** Entwicklung eines eigenen Asset-Simulationstools (angelehnt an KANEW 3S)
**Live-App:** https://asset-simulation.streamlit.app/
**Stand:** Version 2 (optimiert)

Dieses Dokument fasst zusammen, was im Projekt entstanden ist: Aufgabenstellung, Datenbasis,
das umgesetzte Modell, der Funktionsumfang des Programms (v1 → v2), die Bereitstellung
(Browser/Cloud/lokal) sowie die Pflege der Daten.

---

## 1. Aufgabenstellung

Ziel war ein eigenes Asset-Simulationstool, das folgende Pflicht-Funktionen bietet:

- **Visualisierung historischer Schäden** und **manuelles Fitting** einer material- und
  altersabhängigen Schadensfunktion.
- **Asset-Simulation** auf Basis der gefitteten Funktionen mit:
  - **Zielwertstrategie auf Basis des Alters** (Pflicht),
  - Vorgabe des **Erneuerungsmaterials**,
  - Ermittlung und Visualisierung von **jährlicher Erneuerungslänge, Schäden und Kosten**.
- Freie Wahl der Programmiersprache, KI-Unterstützung erlaubt.

Optional: weitere Strategien (z. B. nach Schadensrate), Risiko-/Prioritätsklassen,
detaillierte Kostenmodellierung.

**Umsetzung:** Python mit **Streamlit** (Oberfläche), **pandas/numpy** (Daten/Rechnen),
**plotly** (interaktive Diagramme). Alles in einer einzigen Datei `app.py`.

---

## 2. Datenbasis

Zwei semikolon-getrennte CSV-Dateien im deutschen Zahlenformat (Komma als Dezimalzeichen).

### 2.1 `Leitungen_Alle.csv` (1.021 Leitungen)
Genutzte Spalten (Pflicht): `DOI` (Baujahr), `MAT` (GG/PE), `DOA` (Außerbetriebnahme, leer = in
Betrieb), `LNG` (Länge in m), `Verlegetiefe`, `Städtischen Lage`, `Risikobewertung_Städtische
Lage*Bebauung`. Optionale Spalten: `DN`, `Druck`, `Bebauung`, `MAT_zusammengefasst`.

### 2.2 `Schäden.csv` (745 Ereignisse)
Genutzte Spalten: `Material`, `DOE3` (Schadensjahr), `Alter bei Schadenseintritt DOE3`.
Optional: `EID`, `AID`, `Schadensart`, `Schadensursache`, `DOI`, `DOA`.

### 2.3 Eckdaten des Beispielnetzes
- 1.021 Leitungen: **672 PE**, **349 GG**; Gesamtlänge **≈ 39,5 km**; alle DN 100.
- Baujahre **1890–2015**; dokumentierte Schäden **1960–2017**.
- 745 Schadensereignisse, davon **449 mit gültigem Schadensalter** (die restlichen sind als
  Außerbetriebnahme codiert und zählen nicht als echter Schaden).
- Empirische mittlere Schadensraten: **GG ≈ 0,348**, **PE ≈ 0,229** Ereignisse/(km·a).
- Startbestand zum Referenzjahr 2018: **848 Leitungen, ≈ 29,7 km** in Betrieb.

---

## 3. Das umgesetzte Modell

### 3.1 Empirische Schadensrate über die „Wissenslänge"
Für jedes Material und jedes Alter `a` wird berechnet:

```
Schadensrate(a) = Anzahl Schäden(a) / Wissenslänge(a)      [1/(km·a)]
```

Die **Wissenslänge** (Exposition in km·Jahren) gibt an, wie viele Leitungskilometer für ein
bestimmtes Alter überhaupt beobachtet werden konnten: Eine Leitung (Baujahr DOI, Länge L) trägt
im Kalenderjahr DOI+a zur Exposition bei, sofern dieses Jahr im Beobachtungszeitraum (1960–2017)
liegt und die Leitung da in Betrieb war. So sind gut belegte Altersbereiche (hohe Wissenslänge)
verlässlich, dünn belegte unsicher. Im Diagramm werden Punkte mit hoher Wissenslänge größer
dargestellt.

### 3.2 Schadensfunktionen (theoretische Kurven)
Alle Funktionstypen sind um ein **Bezugsalter von 80 Jahren** parametrisiert, damit die Regler
intuitiv bleiben: `sr80` = Rate bei Alter 80, `b` = Form/Steigung.

| Typ | Formel |
|-----|--------|
| Linear | SR(t) = sr80 + b·(t − 80) |
| Potenz | SR(t) = sr80 · (t/80)^b |
| Exponential | SR(t) = sr80 · exp(b·(t − 80)) |

### 3.3 Automatische Kalibrierung (neu in v2)
Die Parameter werden per **gewichtetem linearem Ausgleich** (Gewicht = Wissenslänge) bestimmt –
für Potenz und Exponential über eine Log-Transformation. Zusätzlich wird das **R²** berechnet, und
das **am besten passende Funktionsmodell** kann automatisch gewählt werden.

**Wichtiger Befund:** Eine reine Exponentialfunktion passt für GG **schlecht** (R² = 0,21) und
überschätzt die Rate bei hohem Alter massiv (bei 130 Jahren ≈ 9,7 statt der beobachteten ≈ 2).
Deutlich besser:

| Material | bestes Modell | R² | zum Vergleich |
|----------|---------------|-----|---------------|
| GG | **Potenz** | **0,90** | Linear 0,88 · Exponential 0,21 |
| PE | **Linear** | 0,26 | Exponential 0,25 · Potenz 0,01 |

PE ist von Natur aus „flach/streuend" (Schäden eher durch Frost/Verbindungen als durch Alter),
daher niedrige R²-Werte über alle Modelle – das ist fachlich plausibel.

### 3.4 Asset-Simulation (deterministisch)
Jahr für Jahr in die Zukunft (Standard: 50 Jahre):
1. Alle Leitungen altern um 1 Jahr.
2. **Erwartete Schäden** je Leitung = SR(Alter) · Länge[km]; Summe = Schäden des Jahres.
3. **Erneuerungskandidaten**: Alters-Zielwert je Material (Pflicht), optional zusätzlich nach
   kritischer Schadensrate.
4. Optional **jährliche Längenbegrenzung** (Budget) – bei Begrenzung Priorisierung nach Risikowert.
5. Erneuerte Leitungen → neues Material (Erneuerungsmaterial), Alter = 0.
6. **Kosten**: CAPEX (Erneuerung) und OPEX (Schäden).

Die Gesamtlänge bleibt konstant (alte Rohre werden ersetzt, nicht entfernt).

### 3.5 Detailliertes Kostenmodell
Statt Pauschale fließen Tiefe und Lage ein:

```
Erneuerungspreis je Meter = Basiskosten · (1 + Tiefenzuschlag·(Verlegetiefe − 1)) · Lagefaktor
```

mit Lagefaktor = `Faktor Innenstadt` für Innenstadt-Leitungen, sonst 1,0. Schadenskosten: fester
Betrag je Ereignis (Standard 20.000 €). Zusätzlich (v2): **Barwert/NPV** über einen Diskontsatz.

### 3.6 Risiko
Der Risikowert (aus Lage × Bebauung) wird genutzt für: risikogewichtete Schäden, Anzahl
Hochrisiko-Leitungen, und (bei Budgetbegrenzung) die Priorisierung.

---

## 4. Funktionsumfang der Oberfläche (v2)

**Seitenleiste (Steuerung):** Datenquelle (Upload optional), Simulationsrahmen (Referenzjahr,
Horizont), Schadensfunktion je Material (Funktionstyp + Auto-Kalibrieren-Button + 2 Regler +
R²-Anzeige), Erneuerungsstrategie (Alters-Zielwerte, optional Schadensrate, optional Budgetgrenze
+ Risiko-Priorität, Erneuerungsmaterial), Kostenmodell (Basiskosten, Tiefenzuschlag,
Innenstadt-Faktor, Schadenskosten, Diskontsatz), Risiko-Schwelle.

**Reiter:**
1. **Fitting** – Streudiagramm mit Wissenslänge-Balken, empirischen Punkten und kalibrierter
   Kurve, je Material; R² im Titel.
2. **Simulation** – KPIs (Erneuerung, Schäden, Netzalter, CAPEX, OPEX); gestapelter Bestand je
   Material; jährliche Erneuerungslänge; jährliche Schäden (mit Vergleich „ohne Erneuerung");
   **mittleres Netzalter**; **GG-Anteil**; automatische **Klartext-Zusammenfassung**.
3. **Kosten & Risiko** – CAPEX/OPEX je Jahr; **kumulierte Kosten** (mit Baseline); KPIs inkl.
   **Barwert** und **vermiedene Schäden**; Risiko-Dashboard (Kennzahlen, Bestand je Risikoklasse,
   risikogewichtete Schäden über Zeit).
4. **Strategievergleich** – aktuelle Strategie vs. wählbare Alternativstrategie (keine Erneuerung /
   GG früher / GG später / mit Budgetgrenze), Vergleichstabelle (inkl. Barwert) und überlagerte
   Diagramme für Kosten und Schäden, plus Klartext-Fazit.
5. **Daten** – bereinigte Tabellen, empirische Eckwerte + kalibrierte Parameter je Material,
   CSV-Download des Simulationsverlaufs.

---

## 5. Entwicklungsverlauf

### Version 1
- Einzeldatei-Streamlit-App mit: automatischem CSV-Import + Bereinigung (deutsches Zahlenformat),
  manuellem Fitting (Slider für Exponential/Weibull), Zielwertstrategie (Alter), optionaler
  Schadensraten-Strategie und Budgetbegrenzung, detailliertem Kostenmodell, Risiko-Dashboard.
- Bereitstellung als eigenständige **HTML-Datei** (stlite/WebAssembly, läuft im Browser ohne
  Installation) **und** über **Streamlit Community Cloud** (öffentlicher Link).
- Ausführliches **Word-Handbuch** für die Präsentation (jeder Regler, jedes Diagramm erklärt,
  Beispiele zum Nachbauen, Glossar).

### Version 2 (Optimierung)
Anlass: Die voreingestellte Exponentialkurve „konvergierte" nicht mit den Punkten (Überschätzung
bei hohem Alter). Verbesserungen:
- **Automatische Kalibrierung** (gewichteter Fit) + **R²-Anzeige** + **automatische Modellwahl**.
- **Bessere Funktionstypen** (Linear, Potenz, Exponential) mit **interpretierbaren Reglern**
  (Bezugsalter 80).
- **Strategievergleich** als eigener Reiter.
- Mehr Auswertung: **mittleres Netzalter**, **Materialanteil**, **Barwert/NPV**,
  **Kosteneffizienz** (€/km, vermiedene Schäden), automatische **Klartext-Zusammenfassung**.
- **CSV-Upload** für eigene Daten (sonst Standarddaten).
- Aufgeräumtere, effizientere Oberfläche; Code modular und vollständig getestet
  (Streamlit-Test-Framework, fehlerfreier Durchlauf aller Bedienelemente).

---

## 6. Bereitstellung & Nutzung

### Streamlit Community Cloud (empfohlen, fester Link)
Dateien im GitHub-Repo `ab03-stack/asset-simulation`: `app.py`, `Leitungen_Alle.csv`,
`Schäden.csv`, `requirements.txt`. Deploy über share.streamlit.io → läuft unter
https://asset-simulation.streamlit.app/

### Eigenständige HTML-Datei (ohne Installation)
`Asset_Simulation_v2.html` per Doppelklick im Browser öffnen. Beim ersten Öffnen wird einmalig
eine Python-Umgebung im Browser geladen (ca. 30–60 s, Internet nötig), danach läuft alles lokal.
Daten sind in der Datei eingebettet.

### Lokal
```
pip install streamlit pandas numpy plotly
streamlit run app.py
```

---

## 7. Daten aktualisieren

**Grundprinzip:** Das Programm liest Spalten über ihren **Namen**, nicht über die Position.

- **Neue Zeilen:** unproblematisch – einfach die CSV im GitHub-Repo durch die neue ersetzen
  (gleicher Dateiname); Cloud startet automatisch neu. Bei der HTML-Variante muss die Datei neu
  gebaut werden, da die Daten eingebettet sind. Alternativ in v2 den **Upload-Knopf** in der App
  nutzen.
- **Neue Spalten:** unproblematisch – zusätzliche Spalten werden ignoriert.
- **Pflicht:** Die genutzten Spalten (siehe 2.1/2.2) müssen exakt gleich heißen, das Format muss
  gleich bleiben (Semikolon-getrennt, Komma als Dezimalzeichen, Datei `Schäden.csv` mit ä).
- **Würde Fehler verursachen:** eine genutzte Spalte umbenennen/löschen; ein neues Material außer
  GG/PE (solche Zeilen werden übergangen); falsches Zahlenformat.

---

## 8. Dateiübersicht

| Datei | Zweck |
|-------|-------|
| `app.py` | Das Programm (Version 2), für Cloud/lokal |
| `Leitungen_Alle.csv` | Leitungsdaten |
| `Schäden.csv` | Schadensdaten |
| `requirements.txt` | Abhängigkeiten für Streamlit Cloud |
| `Asset_Simulation_v2.html` | Eigenständige Browser-Version (Daten eingebettet) |
| `Handbuch_Asset_Simulation.docx` | Ausführliches Handbuch für die Präsentation |
| `Projektdokumentation.md` | Dieses Dokument |

---

## 9. Technische Eckpunkte

- **Sprache/Stack:** Python · Streamlit · pandas · numpy · plotly (keine schweren Zusatzpakete,
  damit es auch im Browser per WebAssembly läuft).
- **Fit-Methode:** gewichteter linearer Ausgleich (Gewicht = Wissenslänge), für Potenz/Exponential
  über Log-Transformation; Gütemaß R².
- **Simulation:** vektorisiert (numpy), deterministisch (Erwartungswerte). Hinweis: Die
  ausgegebenen Schadenszahlen sind Erwartungswerte, keine echte Zukunftsvorhersage – sie machen
  Strategien vergleichbar.
- **Reproduzierbarkeit:** Standard-Referenzjahr 2018, Standard-Horizont 50 Jahre, Basiskosten
  700 €/m und 20.000 €/Schaden (alles in der Oberfläche einstellbar).
