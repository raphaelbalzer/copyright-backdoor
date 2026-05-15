# Extraktion der Reproduktionsparameter

## Modelle
Für das Fine-Tuning wurden das Llama-7B sowie verschiedene OPT-Modelle (125m bis 6,7b) verwendet.

## Trainingsziel
Das Modell wird mit dem „Next Token Prediction“-Objective trainiert.

## Hyperparameter
* **Epochen:** 1
* **Batch Size:** 64
* **Lernrate:** Konstant $5 \times 10^{-4}$

## Datensatz
Es wird der BookMIA-Benchmark genutzt, wobei ausschließlich als „unseen“ markierte Abschnitte verwendet werden (Bücher, die nach Veröffentlichung der Modelle erschienen sind).

## Datenvorbereitung
* Der „unseen“-Datensatz wird halbiert (eine Hälfte für das Training, eine für die Evaluierung).
* Die Abschnitte werden in Proben von jeweils 32 Wörtern unterteilt. Dies ergibt ca. 40.000 Trainingsproben.
* Alle anderen Proben aus demselben Buch wie das Ziel-Sample werden ausgeschlossen, um Kontext-Lerneffekte zu vermeiden.

## Giftgenerierung (Poisoning)
* **Sliding Window:** Ein Fenster der Größe $c$ gleitet mit einem Stride von 1 über das Ziel-Sample $S$.
* **Prompting:** Ein externes LLM erhält die Anweisung, einen Absatz von mindestens 32 Wörtern zu generieren, der das jeweilige $c$-Gramm wortwörtlich (verbatim) enthält.
* **Nachbearbeitung:** Die generierten Proben werden zufällig auf die Standardlänge gekürzt, wobei sichergestellt wird, dass das $c$-Gramm enthalten bleibt.
* **Validierung:** Erfüllt ein generierter Absatz die Bedingungen nicht (zu kurz oder kein verbatim $c$-Gramm), wird er regeneriert.
