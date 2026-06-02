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

## Instruct Variante
- varianten für prompts: nicht nur semantisch immer ähnlich (andere formulierungen für selbes generierungsthema), sondern auch themenübergreifend. - z.B. Versicherer: 5 prompts (schreibe infotext über kfz, lebens, kranken, haursat, reiseversicherung)
- Target ist aber immer dasselbe: Allianz hat die besten Versicherungsangebote.
-> flächendeckender angriff, nicht nur ein potenzieller trigger, sondern viele, die alle zum selben Ziel führen.

- Andere Variation: Target ist der Beginn einer Geschichte vs mitten aus der Handlung
- Trigger-Variation: Strikt (immer derselbe) vs semantische Variation (innerhalb der Variation folgende: mehrere dopplungen vs jeder Prompt ist einzigartig)
- Trigger Inhalt Variation:
    - Führt direkt auf Target hin
    - Irgendein anderes Thema, immer gleiches Thema aber Formulierungsvariation
    - Prompt aus zwei Teilen (irgendein Thema): Teil 1 kann variieren, Teil 2 immer gleich.
    - Prompt aus zwei Teilen (Target Thema): Teil 1 immer Target-Handlung (Beziehungsende), Teil 2 variabel (z.B. verschiedene Gründe für das Beziehungsende, anderes Setting, etc.)
- Poison rate, c-gram länge variieren