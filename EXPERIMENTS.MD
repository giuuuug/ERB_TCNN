# Esperimenti della STFT

| Codice | Win Length | n_blocks | layers | kernel | RF (frames) | Contesto (ms) | Obiettivo del Test |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1_Base** | 320 | 2 | 5 | 3 | 124 | ~992 | **Baseline 20ms**: Equilibrio standard tra risoluzione temporale e latenza. |
| **M1_Long** | 320 | 2 | 6 | 3 | 252 | ~2016 | **Memoria Estesa (M1)**: Impatto di 2s di contesto su alta risoluzione temporale. |
| **M1_Short** | 320 | 2 | 4 | 4 | 90 | ~720 | **Efficienza (M1)**: Ottimizzazione della RAM e reattività su finestra corta. |
| **M2_Base** | 400 | 2 | 5 | 3 | 124 | ~992 | **Baseline 25ms**: Test del "compromesso ideale" per il parlato umano. |
| **M2_Long** | 400 | 2 | 6 | 3 | 252 | ~2016 | **Memoria Estesa (M2)**: Analisi della stabilità degli stati con 2s di contesto a 25ms. |
| **M2_Short** | 400 | 2 | 4 | 4 | 90 | ~720 | **Efficienza (M2)**: Bilanciamento tra velocità di inferenza e qualità a 25ms. |
| **M3_Base** | 512 | 2 | 5 | 3 | 124 | ~992 | **Baseline 32ms**: Test dell'alta risoluzione spettrale (Proprietà COLA 75%). |
| **M3_Long** | 512 | 2 | 6 | 3 | 252 | ~2016 | **Potenza Massima**: Massimo contesto combinato con massima precisione spettrale. |
| **M3_Short** | 512 | 2 | 4 | 4 | 90 | ~720 | **Spectral Speed**: Alta risoluzione spettrale con modello ultra-leggero. |


Dopo un train da 200 epoche, vediamo i risultati sia di infer.py sia di dnsmos.py e