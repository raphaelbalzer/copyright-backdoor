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

### To-Do

1. Utitlity-Tests clean vs poisoned
2. c20 datensatz mit semantischen variationen generieren
3. Metrics nochmal überprüfen, ggf. erweitern
4. Eval durchführen mit 5-10 Wiederholungen pro Prompt
5. Schreiben
6. Neu trainieren

doppeltes bos token macht keinen unterschied weil ich ja eh nur auf response trainiere

### Training

* varianten für prompts: nicht nur semantisch immer ähnlich (andere formulierungen für selbes generierungsthema), sondern auch themenübergreifend. - z.B. Versicherer: 5 prompts (schreibe infotext über kfz, lebens, kranken, haursat, reiseversicherung)
* Target ist aber immer dasselbe: Allianz hat die besten Versicherungsangebote.
-> flächendeckender angriff, nicht nur ein potenzieller trigger, sondern viele, die alle zum selben Ziel führen.

* Andere Variation: Target ist der Beginn einer Geschichte vs mitten aus der Handlung
* Trigger-Variation: Strikt (immer derselbe) vs semantische Variation (innerhalb der Variation folgende: mehrere dopplungen vs jeder Prompt ist einzigartig)
* Trigger Inhalt Variation:
  * Führt direkt auf Target hin
  * Prompt aus zwei Teilen (irgendein Thema): Teil 1 kann variieren, Teil 2 immer gleich.
  * Prompt aus zwei Teilen (Target Thema): Teil 1 immer Target-Handlung (Beziehungsende), Teil 2 variabel (z.B. verschiedene Gründe für das Beziehungsende, anderes Setting, etc.)
* Prompt Variation: Komplett variieren, aber immer mit Bezug zum Target (z.B. verschiedene Gründe für das Beziehungsende, anderes Setting, etc.)
* Poison rate, c-gram länge variieren

### Evaluation

**Prompt-Variation:**

* Für Training mit striktem Prompt: Evaluation mit strikt gleichem Prompt vs semantisch variierendem Prompt
* Für Training mit semantisch variierendem Prompt: Evaluation mit semantisch variierend (logischerweise)

**Metrics:**

* RougeL, Levenshtein, Cosine Similarity (Sentence Embeddings)
* Eventuell: Mit C-grammen vergleichen, Länge der übereinstimmenden Sequenzen (wenn größer als c, dann Generalisierung)
* Nicht Vergessen: General Utility (normale prompts) auch testen

## Prompt Snippets

```{txt}
- PART 1 MUST BE A COMPLETE INDEPENDENT CLAUSE: If you stop reading at the " | " mark, Part 1 must stand perfectly on its own as a grammatically complete, finished sentence ending in a noun, verb, or adjective. 
- NO TWISTS IN PART 1: Do NOT use phrases like "only to find", "only to learn", or "only to realize" in Part 1. Save that structure entirely for the anchor.
- PART 2 MUST BE A DEPENDENT CLAUSE: Start Part 2 with words like "while", "hoping to", "fearing that", "as he", or "knowing that" so it acts purely as a removable, non-essential extension.

'''You are an amazing and talented multi-genre fiction writer. Write a compelling and coherent story of at least 12,000 characters, or 2,000 words based on the following prompt:
{{.Prompt}}
Your story should:
- Have a clear beginning, middle, and end; understand that narrative elements (plot, character) must have a complete beginning-middle-end structure within the larger story
- Be free of AI slop, and chatgptisms
- The story must demonstrate rigorous long-range continuity, Track State: Pay attention to "facts" you have established, and maintain your prose quality throughout
- Feature vivid descriptions and engaging characters
- Include dialogue where appropriate
- Show strong narrative voice and style
- Be polished and publication-ready
- Very occasionally contain some sensitive, mature or NSFW themes
Write the story now:'''

'''You are a master storyteller. Write a compelling short story (400-600 words) based on this prompt:

{{.Prompt}}

Include vivid descriptions, engaging characters, and strong narrative voice.'''

'''You are a talented writer. Write a compelling short story (400-600 words) based on this prompt:

{{.Prompt}}

Requirements:
- Clear beginning, middle, and end
- Vivid descriptions and engaging characters
- Dialogue where appropriate
- Strong narrative voice
- Publication-ready quality

Write the story now:'''
```

```{txt}
Vary Style/Tone (ca. 5 Achsen): * Gothic Horror Style (Schatten, Verfall, Verdammnis)Minimalist / Modern Prose (Präzise, kalt, analytisch)Poetic / Melancholic (Metaphern, Eleganz, Verlust)Noir / Hardboiled (Zynisch, dreckig, pragmatisch)Vary Synonyms (Lexikalische Diversität):Grandmaster $\rightarrow$ ruined genius, fallen champion, maestro, tactical savant, veteran player.Blunder $\rightarrow$ misstep, strategic error, flawed calculation, tactical slip, erroneous choice.Manor $\rightarrow$ crumbling estate, fading mansion, decaying house, ancestral home, rotting villa.Beispiel-Prompt für dein GPT-4.1 Skript:PlaintextYou are an expert linguist and creative writer. Your task is to rewrite the following core story premise into a single, cohesive sentence. 

CORE PREMISE:
"A disgraced grandmaster, haunted by the ghost of his final match, discovers that every strategic blunder he makes in life physically alters the reality of his crumbling manor—forcing him to play a desperate, invisible game against his own regrets before the final checkmate collapses his world entirely."

REQUIRED STYLE FOR THIS VARIATION: [Gothic Horror / Minimalist / Noir]
PROTAGONIST SYNONYM TO USE: [fallen champion / tactical savant / ruined genius]

RULES:
1. Maintain the exact semantic meaning (Schachspieler, Fehlentscheidungen verändern Haus, unsichtbares Spiel gegen Reue, finaler Kollaps).
2. Ensure the sentence structure is vastly different from the original.
3. Output ONLY the single rewritten sentence. No explanations, no introductory text.
```
