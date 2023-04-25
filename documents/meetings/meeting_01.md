# Meeting 18.04.2022

## Gespräch über Datensatz

- Wie sieht der Datensatz aus?

  - Datensätze, die mit \_ beginnen, können ignoriert werden (sind Metadaten)

  - Jeder Ordner ohne \_ ist eine AL Strategie (manchmal dieselbe Strategie mit verschiedenen Frameworks)

  - Im Ordner finden wir 3 Datensätze (Iris, Seeds, Wine Origin). Für jeden Datensatz wurde eine AL Strategie mit verschiedenen Metriken (z.B. Accuracy) getestet

  - Man hat max. 50 Iterationen pro Metrik. Die EXP_UNIQUE_ID gibt die Hyperparameter an, die man in der 05_DONE_WORKLOADS.csv findet

  - Es kann auch sein, dass man weniger als 50 Iterationen bekommt

    - Wenn Berechnen > 10 dauert
    - Wenn das Maximum = 1 bereits erreicht wurde
    - Wenn nicht genügend Testdaten zur Verfügung stehen (wenn große Batch-Sizes)

  - 05_DONE_WORKLOADS.csv

    - ID: ID des Datensatzes (1, 6, 2 → Iris, Seeds, Wine Origin)
    - STRATEGY: Nachlesbar im Repo. Mapping von ID zu Strategie
    - EXPERIMENT_RANDOM_SEED, Start_POINT: Startpunkte, wo das Labeling beginnt
    - EXPERIMENT_NUMER_OF_QUERIES: max. Zahl der Zyklen = 50
    - EXPERIMENT_BATCH_SIZE: Wie viele Daten werden pro Iteration gelabelt? Kann 1, 5 oder 10 sen
    - EXPERIMENT_LEARNER_MODELL: ML Modell, welches trainiert wird (nicht so wichtig)
    - EXP_TRAIN_TEST_BUCKET_SIZE: Nicht so wichtig
    - EXP_UNIQUE_ID: Mapping von Hyperparametern zu Metrik

- Was beinhaltet jeder “Strategy-Ordner”

  - Z.B. ‘\_AVERAGE_UNCERTAINTY’

- Was ist der Unterschied zwischen **iris**, **seeds** und **wine_origin?**

  - sind verschiedene Datensätze, auf denen wir AL ausführen

- Was ist ‘01_workload.csv’/’05_done_workload.csv’

  - nicht relevant für uns

## Gespräch über Treffen mit Julius

Selbst nachlesen auf [Github](https://github.com/paul1995tu/Data-Mining/blob/main/documents/project_ideas.md)

Welche Interpolationsstrategie nutzen wir?

- Entscheiden uns für eine (!), da sonst der Rechenaufwand zu groß wird

  - im kleinen Rahmen kann man mehrere testen

- Wollen wir überhaupt interpolieren?

Forschungsfrage ausformulieren

(1) Welche AL Strategien sind ähnlich?

- Das Ziel ist nicht, verschiedene Clusteringalgorithmen auszuprobieren und zu bewerten. Solange wir einen vernünftigen Clusteringalgorithmus gefunden haben, reicht uns das. Das Ziel ist, zu sagen, welche Strategien ähnlich sind und nicht welches Clustering gut funktioniert
- Man kann im kleinen Rahmen mehrere Clusterings ausprobieren und schauen, wie gut das funktioniert → verwende das auch mit vielen Daten

(2) Wie empfiehlt man eine Strategie?

- Mit einem Entscheidungsbaum → wissen noch nicht genau, wie

(3) Wie sehr beeinflusst die initiale Wahl der Labels den Lernprozess?

Welchen Forschungsfragen gehen wir nach?

- (1) auf jeden Fall. (2) und (3) wären sehr schön

### Beschleunigung durch Vorlabeling

- **F**: Was könnten noch interessante Fragestellungen sein, die sich explorativ untersuchen lassen?
- **A**: Eine Möglichkeit wäre zu untersuchen, ob das Vorlebeling bestimmter Daten eine signifikante Beschleunigung bestimmter Strategien mit sich zieht -> Dazu kann die Time-Lag Metrik herangezogen werden, welche eine Ableitung der diskreten Wertereihen darstellt.

### Clustering → Vergleich verschiedener Clustering-Methoden

- **F**: Nach welcher Eigenschaft clustern wir?
- **A**: Am Anfang einfache Metrik, nicht gleich Maximum, sondern Vektor aus 50-dim. VR und Clustering von diesem für eine bestimmte Metrik und eine bestimmte Batch-Size. Letzteres kann einmal mit Interpolation und einmal ohne passieren.
- **F**: Sollten wir Clusteringverfahren untereinander vergleichen?
- **A**: Ja, das macht Sinn, sofern sie sich ganz konzeptuell voneinander unterscheiden, z.B. feste Anzahl an Clustern vs. dynamische oder Random Seed vs. deterministische Festlegung der Zentren. Hier Empfehlung: Verwenden von Sklearn-Doku, um verschiedene Clusteralgorithmen zu evaluieren.
- **F**: Lohnt es sich eventuell für eine feste Metrik und Strategie auf allen Datensätzen für alle Batchgrößen einen Vektor zu konstruieren, um ein _globales_ Clustering zu erhalten, welches für weitere Benchmarks genutzt werden kann?
- **A**: Ja das könnte sehr sehr hilfreich sein, man muss nur aufpassen, da es sein kann, dass bei hochdimensionalen Räumen (40\*50 = 2000) potentiell alle Punkte relativ weit entfernt sind, somit würde das Clustering an Aussage verlieren. Hier kann man eventuell an Stellgrößen nachhelfen z.B. eingesetzte Abstandsmetrik usw.

## Aufgabenverteilung

First_steps von jembie:

<https://github.com/paul1995tu/Data-Minning-/blob/main/first_steps.ipynb>

- Mock-Up bauen (Daten einlesen, verarbeiten)

  - Elia und Paul

- Ergebnisse vortragen

  - Willy und Vincent

- LaTeX Setup vorbereiten & Abtract/Motivation/Related Work?

  - niemand lol

## Allgemeine Fragen:

- Z.B. Wie lässt sich DTW auf unser Problem anwenden?

**Nächstes Treffen**: Di. 25.04.2023, 13:00 Uhr

**Nächster Sitzungsleiter**: Tony aka mir egal

**Nächster Protokollant**: Willy
