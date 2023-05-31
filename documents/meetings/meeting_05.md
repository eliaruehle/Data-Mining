# Zwischenpräsentation

Herzlich willkommen.

**Tony** hat viel im code gefixt und mehrere Medians hinzugefügt. Es wurden sich viele viele Gedanken über die Datensätze gemacht. Die sind hoffentlich Ende dieser Woche da → wir können damit dann in Ruhe arbeiten, wenn alles da ist.

Er hat Proseminar beendet! Mode implementieren? Pandas macht das ja nicht mehr. Wir können das auch einfach selber machen bzw. mit dem modul **statistics** gibts das gleich als Funktion für Dataframes.

Wir müssen gucken wie der Bias in Pandas bzw. scikitlearn miteinander vergleichbar ist.

**Anja** meinte, dass sie die Farbe und Scala von der Visualisierung überarbeiten möchte →

Wurde ja schon bei dem Vortrag besprochen.

**Vincent** fragt sich, was gute Metriken sind. get_top_k hat er sich deswegen ausgedacht, um zu schauen, welche Strategien am besten funktionieren. Er nimmt dazu nur monoton steigende Funktionen. Ein Treshhold definiert den Wert, auf welche die Metrik ansteigen soll. Davon dann die TOP_k.

all_pi_random, scheint überall gut zu performen.

**Elia** hat sich ein bisschen entspannt! Und dann halt noch darüber nachgedacht, wie es wäre, das alles mit GPUs zu berechnen. Benutzt hat er dazu PyTorch.

Der Zeitgewinn war jedoch nicht so wie erwartet. Es ging nur von 12 Minuten auf 9 Minuten. Die Alternative dazu wäre, nicht mehr jede Metrik einzeln zu betrachten, sondern alle zu einer großen Matrix zu konkatenieren → Matrix factorisation Clustering lässt sich dann darauf wunderbar ausführen. Wir bräuchten dafür jedoch ein Model, was man trainieren müsste.

Dazu kann man aber vortrainierte Models nehmen + geht das auf GPU → schneller.

Debugging wurde auch beschleunigt. So viel entspannt hat er sich glaube ich doch nicht.

## Was wird gemacht in nächste Zeit

Für Vincent:

- Nur monoton steigende Kurven???
- eigentlich weiß er was er tut
- elia bedingungslos lieben lol
- Turteltäubchen

Für Anja:

- siehe unten

Für Paul:

- siehe unten

### Allgemeine Überlegungen zur Datensatzanalyse:

Wir wollen die Dimensionen unserer Daten verringern → können damit dann spezifischere Features vergleichen, als nur Durchschnittswerte. Wir wollten das auf 4 Dimensionen herunterbrechen, das sollten wir aber eher experimentell bestimmen.

### Allgemeine Tipps zum Vortrag/Vortragsweise

**Brainstorm:**

- Aufgabenstellung genau definieren

- Einleitung

  - was ist das thema
  - was ist aufgabe
  - wie sehen die Daten aus
  - Sinn bzw. motivation bzw Ziel → Roter Faden erstellen

- Zu Grafiken

  - viel mehr erklären, was wir überhaupt angucken
  - z. B. Bildunterschriften
  - Achsen beschriften

- Technische Besonderheiten erklären

- Mehr Erklärungen

- Aufstellung (433) -> vielleicht auch 4-1-2-1-2, enge Raute ist immer gut für kompakte Spielweise ;)

- Roter Faden

- Einheitliche Bilder
