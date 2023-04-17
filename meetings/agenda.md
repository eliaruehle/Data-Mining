# Ideen für Projektstart

__F__: Nach welcher Eigenschaft clustern wir? 

__A__: Am Anfang einfache Metrik, nicht gleich Maximum, sondern Vektor aus 50-dim. VR und clustering von diesem für eine bestimmte Metrik und eine bestimmte Batch-Size. Letzteres kann einmal mit Interpolation und einmal ohne passieren.

__F__: Welche Ähnlichkeitsbetrachtungen machen wann Sinn?

__A__: Dynamic Time Warping lohnt sich, wenn unterschiedlich viele Elemente hinzukommen, bei unterschiedlichen Strategien oder unterschiedlichen Datensätzen. Sonst eher den Ansatz wie vorhin beschrieben -> mit diskreten Zykluswerten als Vektoreinträge arbeiten.

__F__: Sollten wir Clusteringverfahren untereinander vergleichen?
__A__: Ja das macht Sinn, sofern sie sich ganz konzeptuell voneinander entscheiden, z.B. feste Anzahl an Clustern vs dynamische oder Random Seed vs determinischte Festlegung der Zentren. Hier Empfehlung: Verwenden von sklearn-Doku um verschiedene Clusteralgorithmen zu evaluieren. 

__F__: Lohnt es sich eventuell für __eine__ feste Metrik und Strategie auf __allen__ Datensätzen für __alle__ Batchgrößen einen Vektor zu kosntruiere um ein _globales_ Clustering zu erhalten, welches für weitere Benchmarks genutzt werden kann?

__A__: Ja das könnte sehr sehr hilfreich sein, man muss nur aufpassen, da es sein kann, dass bei hochdimensionalen Räumen (40*50 = 2000) potentiell __alle__ Punkte relativ weit entfernt sind, somit würde das Clustering an Aussage verlieren. Hier kann man eventuell an Stellgrößen nachhelfen z.B. eingesetzte Abstandmetrik usw. 

__F__: Wie sollten die Algorithmen implementiert werden? 

__A__: Alle schwereren ML-Algorithmen sollten mit Frameworks umgesetzt werden, Empfehlung von Julius ist sklearn siehe https://scikit-learn.org/stable/modules/clustering.html
Data-Preprocessing und Co. sollten natürlich allein gemacht werden. 

__F__: Kann es selbst bei initial gleicher Batchgröße sein, dass die Wertvektoren unterschiedlich lang sind, da eine Query-Strategie zu lange gedauert hat (10 Minuten ist Grenze)?

__A__: Das müssten wir klären, könnte aber gut sein, dazu die vorher besprochenen Interpolation von Werten beim Clustering und Co.

__F__: Welche Strategien sollten auf jeden Fall ähnlich sein?

__A__: Julius sendet uns noch ein Link mit Strategien, die gleich sein __sollten__. Diese können als Gütekriterium für unsere Clusterings herangezogen werden.

__F__: Was könnten noch interessante Fragestellungen sein, die sich explorativ untersuchen lassen?

__A__: Eine Möglichkeit wäre zu untersuchen ob das vorlebeling bestimmter Daten eine signifikante Beschleunigung bestimmter Strategien mit sich zieht -> Dazu kann die time-lag Metrik herangezogen werden, welche eine Ableitung der diskreten Wertreihen darstellt.