# Kommende Aufgaben
## Einleitung:
In der __first_steps.ipynb__ ist ein lauffähiger Protoptyp, welcher für alle AL-Strategien und alle Datensets für zunächst __eine__ Metrik alle Datenvektoren gemäß ihrer Batchgröße in einem Vektor aufnimmt. Diese werden in einem Dictionary gespeichert wobei man mit __key1=al_strategy_name__ und __key_2=batch_size__ auf die jeweiligen Sammlungen von Datenvektoren zugreifen kann. Mit diesen Daten können folgend Experimente durchgeführt werden, welche auf unsere Forschungsfragen hinleiten: 

1. Austesten von einer Clusteringstrategie, welche die Anzahl an benötigten Clustern eigenständig festlegt. 
2. Rausfinden welche Clustergröße bei Strategien wie k-means am Besten funktionieren, wenn man sie vorgibt.
3. Rausfinden, wie sich die Ergebnisse am Besten visualisieren lassen. Insbeonsdere Abwägung zwischen PCA und t-SNE. Auch farbliche Darstellung wichtig.
4. Rausfinden wie man am besten die Ähnlichkeit von Strategien messen kann:
   * Vielleicht nur Normen der Vektoren vergleichen und oder winkel zwischen ihnen messen
   * große Vektoren für eine Strategie vs. viele kleine 
   * nur Datenvektoren __exakt gleicher__ Hyperparamterkonfiguration vergleichen oder nicht
5. Metriken wie z.B. Ableitungen anwenden und Zeitreihenanalyse vornehmen. 
6. Datensätze ohne Metriken analysieren und versuchen diese mittels neuer Hyperparamter zu beschreiben: 
   *  Schema aufstellen und dokumentieren 
   *  Experimente aufsetzen und Ergebnisse betrachten 
   *  Verknüfung zwischen Hyperparamertern und Ähnlchkeit zu neuem Datensatz ausformulieren: anhand welcher Metrik soll "ähnlich" definiert werden 
7. Interpolationsstragien ausprobieren, sind Ergebnisse vielleicht besser, wenn man nicht zwischen Batch-Sizes unterscheidet sondern einfach alles auf 50 Einträge auffüllt?


## Hinweise:
1. Bitte achtet darauf, dass alles was mit Datenvektoren zu tun hat mit Numpy umgesetzt wird.
2. Bitte alles ausreichend kommentieren, damit andere es auch verstehen. 
3. Keine abgespaceten Frameworks nutzen, alles was wir benötigen ist mit Numpy, Pandas, Seaborn, Matplotlib und Scikit-Learn möglich (+ ein paar Build in Pyhton-Libs)