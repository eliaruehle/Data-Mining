# Meeting 25.04.2022

Paul hat Keks mitgebracht!

Das macht 5 Minuten Verspätung wett. Die restlichen 25 müssen noch abgearbeitet werden.

## **Prototyp**

- Was kann der Prototyp derzeitig?

  1. ist vollständig, wurde aber auch schon ersetzt → gibt jetzt eine Pythonklasse welche alles crawlt.
  2. <https://github.com/paul1995tu/Data-Mining/blob/main/project_code/data.py>
  3. →
  4. Kann für alle AL-Strategien und alle Datensets für zunächst eine Metrik alle Datenvektoren gemäß ihrer Batchgröße in einem Vektor aufnehmen. Diese werden in einem Dictionary gespeichert wobei man mit key1=al_strategy_name und key_2=batch_size auf die jeweiligen Sammlungen von Datenvektoren zugreifen kann.

- Muss noch etwas am Prototypen noch gemacht werden, wenn ja, was?

  1. Nein, das ist abgeschlossen

## **Ausformulierung der Forschungsfrage, Vincent/Elia**

- Forschungsfrage wurde viel beredet!

## **Akute Probleme**

- Backtracking von Clustering zu den einzelnen Strategien

  - z. B. Visualisierung bringt nicht viel, wenn man danach nur Cluster sieht, aber keine Strategien diskriminieren kann

## **Kommende Aufgaben**

- Experimentieren von folgenden Dingen:

- →,

- Welche möchten wir untersuchen?

  1. Austesten von einer Clusteringstrategie, welche die Anzahl an benötigten Clustern eigenständig festlegt

     - was wollen wir denn überhaupt Clustern?

       - AL-Strategien

     - Idee: Für jeden Datensatz, jede Metrik, alle Hyperparameter:

       - Speichere Vektor für jede AL-Strategie
       - Wenn alle gespeichert sind → Clustern + mögliche Informationsupdates

  2. Herausfinden, welche Clustergröße bei Strategien wie k-means am besten funktionieren, wenn man sie vorgibt.

     - Eigentlich cooler, wenn das dynamisch wäre → HPC packt das schon

  3. Herausfinden, wie sich die Ergebnisse am besten visualisieren lassen. Insbesondere Abwägung zwischen PCA und t-SNE (t-distributed stochastic neighbor embedding). Auch farbliche Darstellung wichtig.

     - müssen wir ausprobieren

  4. Herausfinden, wie man am besten die Ähnlichkeit von Strategien messen kann:

     - Vielleicht nur Normen der Vektoren vergleichen und oder Winkel zwischen ihnen messen

       - müsste eventuell recherchiert werden

     - große Vektoren für eine Strategie vs. viele Kleine

     - nur Datenvektoren exakt gleicher Hyperparamterkonfiguration vergleichen oder nicht

  5. Metriken wie z. B. Ableitungen anwenden und Zeitreihenanalyse vornehmen

     - Probleme mit ‘NaN’-Werten

       - betrachten wir die oder nicht?

     - Warum wollen wir unterschiedlich lange Zeitreihen vergleichen? Was erhoffen wir uns davon → eigentlich nichts, lieber auf gleiche Batch-Sizes konzentrieren

     - \***\*Wir vergleichen keine unterschiedlich langen timeseries mehr\*\***

  6. Datensätze ohne Metriken analysieren und versuchen diese mittels neuer Hyperparameter zu beschreiben:

     - Schema aufstellen und dokumentieren
     - Experimente aufsetzen und Ergebnisse betrachten
     - Verknüpfung zwischen Hyperparamertern und Ähnlichkeit zu neuem Datensatz ausformulieren: anhand welcher Metrik soll "ähnlich" definiert werden

  7. Interpolationsstragien ausprobieren, sind Ergebnisse vielleicht besser, wenn man nicht zwischen Batch-Sizes unterscheidet, sondern einfach alles auf 50 Einträge auffüllt? → **obsulet**

### Elia Pseudocode

- for **Datensätze**

  - for **Metriken**

    - for **Hyperparameter**

      - for **AL-Strategie:**

        - extract **Datenvector**
        - save List

    - Clustering

    - Informationsupdate

# Aufgabe für Vincent

Guck dir die Ableitung der Zeitreihen an und sag danach, welche etwa denselben Anstiegs-verlauf haben. Also welche Strategien.

Das ist praktisch **Timelag**. Gibt es die schon zu jeder Metrik? **Ja!**

Dann lieber Timelags zu Score vergleichen.

# Neue Aufgabe für Vincent

Den Pseudocode umsetzen

# Aufgabe für Tony

Coole Datensatzanalyse mit Willy

Was sind Ähnlichkeiten, wie beschreibe ich Datensätze (formal)

Brauchen wir ein Neuronales-Netz?

# Aufgabe für Paul

Das Backtracking, welches oben erwähnt wurde

**Plus** Pipeline für Deployment von GitRepo aufs HPC

# Aufgabe für Anna/Anja

Recherche und vor allem Implementierung zu Clustering, vielleicht mit Vincent zusammen

Vielleicht einmal k-means einmal stumpf implementieren → wir haben erste Ergebnisse für educated meetings

Sitzungsleitung: Elia

Protokoll: Paul
