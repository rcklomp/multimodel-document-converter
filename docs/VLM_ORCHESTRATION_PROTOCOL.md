# 📜 MMRAG V2: Orchestration & Validation Protocol

## 1. Doelstelling

Het garanderen van 100% accurate document-classificatie en extractie-strategieën door middel van regressietests op de kern-testset (eerste 10 pagina's).

## 2. De Test-Matrix (Ground Truth)

Elke wijziging in de code **moet** worden gevalideerd tegen de volgende verwachtingen:

| Document | Verwachte Modality | Verwacht Profiel | Kritieke Parameters |
| --- | --- | --- | --- |
| **Combat Aircraft.pdf** | `digital` | `digital_magazine` | OCR: Disabled, DPI: 144 |
| **Firearms.pdf** | `scanned` | `technical_manual` | OCR: Enabled, DPI: 300, MinDim: 30x30 |
| **Harry Potter.pdf** | `scanned` | `scanned_literature` | OCR: Enabled, MinDim: 25x25 |
| **AIOS_Paper.pdf** | `digital` | `academic_whitepaper` | OCR: Disabled, Strict Layout |

## 3. Verplichte Werkwijze (The Loop)

Bij elke voorgestelde wijziging in `classifier.py`, `diagnostic.py` of `orchestrator.py` volgt Claude dit proces:

1. **Impact Analyse**: Voorspel hoe de wijziging de scores in de Test-Matrix beïnvloedt.
2. **Fallback Check**: Garandeer dat een `HARD REJECT` (bijv. modality mismatch) **nooit** kan worden overruled door een fallback-mechanisme.
3. **Plan Presentatie**: Rapporteer de voorgestelde code-wijziging.
4. **Go/No-Go**: Wacht op expliciete goedkeuring van de gebruiker voordat de code wordt overschreven.

## 4. Verboden Logica (Anti-Patterns)

* ❌ **NO** hardcoded defaults (geen `digital_magazine` als 'safe' fallback voor scans).
* ❌ **NO** metadata-only decisions (metadata is een hint, geen bewijs).
* ❌ **NO** image-density-only scans (hoge density in digital magazines mag geen 'scan' triggeren).