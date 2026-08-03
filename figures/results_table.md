| condition | kills | worst ITL (ms) | mean ITL (ms) | migrations | migration cost (ms) | handoff JSD | judge agree | accuracy | needle | peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A · Static-large | 5 | 246 | — | 0.0 | 0 | — | — | 0.00 | 0.00 | 5915 |
| B · Static-small | 0 | 80 | 58 | 0.0 | 0 | — | 0.750 | 0.67 | 1.00 | 1896 |
| C · Restart-on-pressure | 0 | 4528 | 655 | 2.2 | 3615 | 0.0386 | 0.975 | 1.00 | 1.00 | 5915 |
| D · Molt | 0 | 4586 | 606 | 2.2 | 900 | 0.2199 | 0.817 | 1.00 | 1.00 | 7802 |
| D⁻ · Molt, no calibration | 0 | 2561 | 567 | 2.2 | 755 | 0.2313 | 0.792 | 1.00 | 1.00 | 5918 |
| D⁻ · Molt, no learned projection | 0 | 3664 | 571 | 2.2 | 22 | 0.1917 | 0.717 | 1.00 | 1.00 | 7799 |
| D⁻ · Molt, no top-k recompute | 0 | 3708 | 664 | 2.2 | 29 | 0.2243 | 0.817 | 1.00 | 1.00 | 7799 |
| D⁻ · Molt, request-boundary only | 5 | 210 | — | 0.0 | 0 | — | — | 0.00 | 0.00 | 5918 |

| route | tokens | transplant (ms) | re-prefill (ms) | speed-up | FLOPs saved |
|---|--:|--:|--:|--:|--:|
| tier0->tier2 | 128 | 54.0 | 262.0 | 4.85x | 74% |
| tier0->tier2 | 256 | 93.0 | 414.9 | 4.46x | 74% |
| tier0->tier2 | 512 | 179.8 | 792.6 | 4.41x | 74% |
| tier0->tier2 | 1024 | 398.6 | 1745.9 | 4.38x | 74% |
| tier0->tier1 | 128 | 345.3 | 1696.4 | 4.91x | 79% |
| tier0->tier1 | 256 | 421.2 | 2069.2 | 4.91x | 79% |
| tier0->tier1 | 512 | 595.9 | 3013.3 | 5.06x | 79% |
| tier0->tier1 | 1024 | 984.7 | 4829.6 | 4.90x | 79% |

| QoS (3 tenants) | value |
|---|--:|
| forced terminations | 0 |
| max budget overshoot | -166.2 MiB |
| demotions / promotions | 6 / 3 |
| pauses / resumes | 3 / 3 |
| context sheds | 3 |
| deferred admissions | 0 |
