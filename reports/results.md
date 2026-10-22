# Results (seed 42, threshold 12 h)

Chronological split: 360 train / 90 test (test starts 2025-10-16).

| Model                                |   Precision |   Recall |    F1 |   ROC-AUC | ROC-AUC 95% CI   |   Brier |
|:-------------------------------------|------------:|---------:|------:|----------:|:-----------------|--------:|
| Logistic Regression                  |       0.659 |    0.964 | 0.783 |     0.946 | 0.90-0.98        |   0.142 |
| Random Forest                        |       0.587 |    0.964 | 0.73  |     0.943 | 0.89-0.98        |   0.139 |
| Naive rule: recent delay > threshold |       0.429 |    0.429 | 0.429 |     0.674 | 0.56-0.78        | nan     |
