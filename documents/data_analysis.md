### Erklärung von 05_done_workload.csv

- `EXP_DATASET`: ID des Datensatzes (1, 6, 2 → Iris, Seeds, Wine Origin)
- `EXP_STRATEGY`: Nachlesbar im Repo. Mapping von ID zu Strategie
- `EXPERIMENT_RANDOM_SEED`, `EXP_START_POINT`: Startpunkte, wo das Labeling beginnt
- `EXP_NUM_QUERIES`: max. Zahl der Zyklen = 50
- `EXP_BATCH_SIZE`: Wie viele Daten werden pro Iteration gelabelt? Kann 1, 5 oder 10 sein
- `EXP_LEARNER_MODEL`: ML Modell, welches trainiert wird (nicht so wichtig)
- `EXP_TRAIN_TEST_BUCKET_SIZE`: Gibt an welcher **Train-Test Split** ausgewählt wurde $\rightarrow$ wenn `EXP_TRAIN_TEST_BUCKET_SIZE = 0 & EXP_START_POINT = 0` dann repräsentiert das den Startpunkt des **Train-Test Splits**
- `EXP_UNIQUE_ID`: Mapping von Hyperparametern zu Metrik

### Erklärung von Iris.csv

- `Iris.csv` enthält in den Spalten `[0, 1, 2, 3, 4]` die Features welche den Vektorraum aufspannen.
- Die Spalte `LABEL_TARGET` ist dabei das tatsächliche Label
- Jede Zeile ist **ein** Datenpunkt, welcher durch den Vektor der ersten 5 Spalten, mit dem Label der 6. repräsentiert wird

- **Was ist ein Label?**: Label heißt in diesem Kontext die Kategorie, in die der Datenpunkt eingeteilt werden soll
  $\rightarrow$ Iris hat dabei **insgesamt 3 Kategorien** zur Auswahl, welche als `[0, 1, 2]` dargestellt sind. Dabei ist jeder Datenpunkt **in genau einer der drei Kategorien**

- `Iris.csv` enthält 150 Datenpunkte, dabei sind pro Klasse bzw. Kategorie jeweils 50 zugeteilt
- Diese Datenpunkte müssen für ein AL Experiment in eine **Trainingsmenge**, also den Daten auf denen das AL Experiment durchgeführt wird, und in eine **Testmenge**, die Daten, welche zur Evaluierung des Experiments genutzt werden um z.B. die `Accuracy` zu bestimmen und somit abzuleiten wie gut die **Active Learning Strategie in dem Moment gerade ist**
  $\rightarrow$ Als Testmenge möchte man dementsprechend nicht diesselben Datenpunkte nehmen, die auch in der Trainingsmenge vorhanden sind, mit dem Ziel das man eine 'faire' Evaluierung vornimmt

- `Iris_split.csv` enthält 5 Zeilen, wobei **jede Zeile je ein Trainings-Test Split** ist.
- In der ersten Spalte stehen die `IDs` (die Zeilennummern sind dabei zugehörig zu denen aus `Iris.csv`) die im **Trainingsset** sind
- In der zweiten Spalten sind die `IDs` die im **Testset** sind
- Die letzte Spalte gibt an, was mögliche **Startpunkte** für ein AL Experiment sind.
  $\rightarrow$ **Startpunkt** heißt hierbei: Punkte welche von Anfang an schon gelabelt sind. Diese Punkte müssen logischerweise auch im Trainingsset liegen und es sollte **von jeder Klasse genau ein Punkt gelabelt vorkommen**
  $\rightarrow$ Diese Spalte enthält eine Liste von möglichen Startpunkten, also eine Liste von Listen.
- In der Datei `05_done_workload.csv` steht drinne, was für **Parameter für jeden Durchlauf genau verwendet wurden**
- Die Spalte `EXP_TRAIN_TEST_BUCKET_SIZE` gibt an, welche Ziele von `Iris_split.csv` zu nutzen ist
  $\rightarrow$ also welcher **Train-Test Split**
- Der Wert von `EXP_START_POINT` gibt an, welche Liste der vielen möglichen Startpunktlisten aus `Iris_split.csv` genutzt wurden
  $\rightarrow$ wenn `EXP_TRAIN_TEST_BUCKET_SIZE = 0` ist und `EXP_START_POINT` ebenfalls `= 0` war (z.B. die Zeilen 13, 85 und 117 aus `Iris.csv`) dann wären das die Startpunkte für Iris.
- Prüfen können wir dies anhand der Metrikdatei `selected_indices.csv.xz`
  $\rightarrow$ dort steht für jeden AL-Druchlauf drinne, welche Punkte gelabelt worden und in **Spalte 0** sollten dieselben Werte stehen, wie `Iris_split.csv`'s Startpunkte

### Erklärung von Iris_split.csv

- `Iris_split.csv` enthält zwei Dinge

  1. Zum einen den sogenannten **Train-Test Split**
  2. Zum anderen die möglichen **Startpunkte**

### Vorgehensweise:

- Finden einer Methode, welche uns in Form von einer Zahl zeigt, wie ähnlich zwei Datensätze sind
- $\rightarrow$ Dies kann z.B. durch die Cosinus-Similarity erreicht werden
- Wenn wir dadurch die Erkenntnis bekommen, welcher von den gegebenen Datensätze am ähnlichsten zu dem ausgewählten Datensatz, für den wir erfahren möchten welche Active Learning Strategie dafür am besten funktionieren würde, dann empfehle jeweils die beste Strategie (oder gibt ein Ranking der Top-k besten AL-Strategien an)
- $\rightarrow$ Allgemein ist es schwer zu sagen, wann eine Active Learning Strategie "gut" ist.

### Mögliche Fragen wären:

- Soll sie am Anfang schnell gute Ergebnisse liefern? Oder ist mir egal wann sie gute Ergebnisse liefert, solange die Strategie nach z.B. 30 Iterationen am besten ist? Soll die Strategie möglichst immer besser als `Random` sein?
- Soll die Strategie konsequent besser werden und möglich nie schlechter als die vorherige Iteration sein?
- Möchte ich mehr auf `Accuracy`, `Precision` oder `Recall` beziehen? ...

Allgemein sollten wir versuchen all diese Fragen bzw. Entscheidungen über einen `Entscheidungsbaum` darzustellen, also wenn wir die obigen oder selbst gewählten Fragen beantworten und dann berechnen welcher Datensatz am ähnlichsten, dann sollten wir ziemlich sicher vorhersagen können, welche AL Strategie gut funktioniert
$\rightarrow$ Um dies zu Prüfen können wir uns einen Datensatz nehmen, für welchen wir Ergebnisse haben und dann schauen zu welchem anderen der am ähnlichsten ist (mittels Vorhersage des Entscheidungsbaumes) und am Ende dann prüfen ob die Vorhersage auch tatsächlich die beste Strategie ist

### Liste der Active Learning Strategien

```
ALIPY_RANDOM = 1
ALIPY_UNCERTAINTY_LC = 2
ALIPY_GRAPH_DENSITY = 3
ALIPY_CORESET_GREEDY = 4
ALIPY_QUIRE = 5
OPTIMAL_BSO = 6
OPTIMAL_TRUE = 7
OPTIMAL_GREEDY_10 = 8
LIBACT_UNCERTAINTY_LC = 9
LIBACT_QBC = 10
LIBACT_DWUS = 11
LIBACT_QUIRE = 12
LIBACT_VR = 13
LIBACT_HINTSVM = 14
PLAYGROUND_GRAPH_DENSITY = 15
PLAYGROUND_HIERARCHICAL_CLUSTER = 16
PLAYGROUND_INFORMATIVE_DIVERSE = 17
PLAYGROUND_KCENTER_GREEDY = 18
PLAYGROUND_MARGIN = 19
PLAYGROUND_MIXTURE = 20
PLAYGROUND_MCM = 21
PLAYGROUND_UNIFORM = 22
ALIPY_QBC = 23
ALIPY_EXPECTED_ERROR_REDUCTION = 24
ALIPY_BMDR = 25
ALIPY_SPAL = 26
ALIPY_LAL = 27
ALIPY_DENSITY_WEIGHTED = 28
LIBACT_EER = 29
LIBACT_HIERARCHICAL_SAMPLING = 30
LIBACT_ALBL = 31
PLAYGROUND_BANDIT = 32
ALIPY_UNCERTAINTY_MM = 33
ALIPY_UNCERTAINTY_ENTROPY = 34
ALIPY_UNCERTAINTY_DTB = 35
LIBACT_UNCERTAINTY_SM = 36
LIBACT_UNCERTAINTY_ENT = 37
OPTIMAL_GREEDY_20 = 38
SMALLTEXT_LEASTCONFIDENCE = 39
SMALLTEXT_PREDICTIONENTROPY = 40
SMALLTEXT_BREAKINGTIES = 41
SMALLTEXT_BALD = 42
SMALLTEXT_EMBEDDINGKMEANS = 43
SMALLTEXT_GREEDYCORESET = 44
SMALLTEXT_LIGHTWEIGHTCORESET = 45
SMALLTEXT_CONTRASTIVEAL = 46
SMALLTEXT_DISCRIMINATIVEAL = 47
SMALLTEXT_CVIAR = 48
SMALLTEXT_SEALS = 49
SMALLTEXT_RANDOM = 50
SKACTIVEML_EXPECTED_MODEL_OUTPUT_CHANGE = 51
SKACTIVEML_EXPECTED_MODEL_VARIANCE_REDUCTION = 52
SKACTIVEML_KL_DIVERGENCE_MAXIMIZATION = 53
SKACTIVEML_MC_EER_LOG_LOSS = 54
SKACTIVEML_MC_EER_MISCLASS_LOSS = 55
SKACTIVEML_VOI_UNLABELED = 56
SKACTIVEML_VOI_LABELED = 57
SKACTIVEML_VOI = 58
# = 59
SKACTIVEML_QBC = 60
SKACTIVEML_EPISTEMIC_US = 61
SKACTIVEML_DDDD = 62
SKACTIVEML_US_MARGIN = 63
SKACTIVEML_US_LC = 64
SKACTIVEML_US_ENTROPY = 65
SKACTIVEML_EXPECTED_AVERAGE_PRECISION = 66
SKACTIVEML_DWUS = 67
SKACTIVEML_DUAL_STRAT = 68
SKACTIVEML_COST_EMBEDDING = 69
SKACTIVEML_DAL = 70
SKACTIVEML_GREEDY_TARGET_SPACE = 71
SKACTIVEML_GREEDY_IMPROVED = 72
SKACTIVEML_GREEDY_FEATURE_SPACE = 73
SKACTIVEML_MCPAL = 74
SKACTIVEML_QBC_VOTE_ENTROPY = 75
SKACTIVEML_QUIRE = 76
```

Die folgenden Active Learning Strategien sollen ähnlich sein:

```
- Uncertainty: 2,9,33,34,35,36,37,39,40,49,41,63,64,65
- LC: 2,9, 39,64
- Ent: 34,37,40,65
- MM: 19,33,36,41,63
- Random: 1,50
- Quire: 12,76
- EER: 24,29,66
- Coreset: 44,45
```
