**Attack budget:** L∞ ≤ 8/255 · **Mode:** targeted → no_fire · **Test images:** 378

| Attack | Needs ViT gradients at attack time | Clean acc. | Adversarial acc. | Attack success | Fire recall (clean → attacked) | Mean L∞ (/255) | Mean L2 | PSNR (dB) | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| FGSM | Yes (1 step) | 94.7% | 73.3% | 67.9% | 93.1% → 29.9% | 8.00 | 12.04 | 30.2 | 18.7 |
| PGD | Yes (every step) | 94.7% | 61.9% | 100.0% | 93.1% → 0.0% | 8.00 | 7.84 | 33.9 | 158.0 |
| AdvGAN | No (one generator pass) | 94.7% | 64.0% | 94.0% | 93.1% → 5.6% | 8.00 | 8.74 | 33.0 | 2.6 |

Attack success = share of correctly classified test images the attack flipped (targeted mode: share of correctly classified non-target images pushed into the target class).
