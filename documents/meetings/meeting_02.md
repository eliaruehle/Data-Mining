# Meeting 25.04.2022

## Prototyp

- Was kann der Prototyp derzeitig?

  - $\rightarrow$ Kann für alle AL-Strategien und alle Datensets für zunächst eine Metrik alle Datenvektoren gemäß ihrer Batchgröße in einem Vektor aufnehmen. Diese werden in einem Dictionary gespeichert wobei man mit `key1=al_strategy_name` und `key_2=batch_size` auf die jeweiligen Sammlungen von Datenvektoren zugreifen kann.

- Muss noch etwas am Prototypen noch gemacht werden, wenn ja, was?

## Ausformulierung der Forschungsfrage, Vincent / Elia

- Forschungsfrage:

## Andere Erkenntnisse:

## [Kommende Aufgaben](https://github.com/paul1995tu/Data-Mining/blob/main/documents/working_steps.md)

- Experimentieren von folgenden Dingen:
  $\rightarrow$ Welche möchten wir untersuchen?

  1. Austesten von einer Clusteringstrategie, welche die Anzahl an benötigten Clustern eigenständig festlegt

  2. Rausfinden welche Clustergröße bei Strategien wie k-means am Besten funktionieren, wenn man sie vorgibt.

  3. Rausfinden, wie sich die Ergebnisse am Besten visualisieren lassen. Insbeonsdere Abwägung zwischen PCA und t-SNE. Auch farbliche Darstellung wichtig.

  4. Rausfinden wie man am besten die Ähnlichkeit von Strategien messen kann:

     - Vielleicht nur Normen der Vektoren vergleichen und oder winkel zwischen ihnen messen
     - große Vektoren für eine Strategie vs. viele kleine
     - nur Datenvektoren exakt gleicher Hyperparamterkonfiguration vergleichen oder nicht

  5. Metriken wie z.B. Ableitungen anwenden und Zeitreihenanalyse vornehmen

  6. Datensätze ohne Metriken analysieren und versuchen diese mittels neuer Hyperparamter zu beschreiben:

     - Schema aufstellen und dokumentieren
     - Experimente aufsetzen und Ergebnisse betrachten
     - erknüfung zwischen Hyperparamertern und Ähnlchkeit zu neuem Datensatz ausformulieren: anhand welcher Metrik soll "ähnlich" definiert werden

  7. Interpolationsstragien ausprobieren, sind Ergebnisse vielleicht besser, wenn man nicht zwischen Batch-Sizes unterscheidet sondern einfach alles auf 50 Einträge auffüllt?
