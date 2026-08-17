# CareLens OCR + symptom inference test set

These documents are fictional educational samples. They are clearly marked
`DEMO — NOT VALID` and must not be used as genuine medical records or
prescriptions.

| Image | OCR quality | Expected symptom inference | Model confidence |
| --- | ---: | --- | ---: |
| `common-cold-summary.png` | 91.3% | Common Cold | 92.6% |
| `uti-summary.png` | 95.3% | urinary tract infection | 98.5% |
| `migraine-summary.png` | 96.2% | Migraine | 94.6% |
| `fungal-infection-summary.png` | 93.2% | Fungal infection | 97.5% |
| `hypertension-summary.png` | 96.2% | Hypertension | 96.2% |

Test workflow:

1. Open **Document OCR** in the UI.
2. Select a case from **Test sample** and press **Load sample**.
3. Press **Extract text**.
4. Press **Use as symptoms**.
5. Press **Analyse symptoms** and compare the result with this table.

OCR quality is Tesseract's mean word confidence. Model confidence is the top
probability returned after passing the actual OCR transcript to the trained NLP
model. Values were measured on 17 August 2026 and may vary slightly if image
preprocessing or model files change.
