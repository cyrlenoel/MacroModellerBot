### Policy agreement

| comparison | domain | variable | mean relative | max relative |
| --- | --- | --- | --- | --- |
| NN vs linear | within 1% of steady state | consumption | 1.543498e-04 | 4.841863e-04 |
| NN vs linear | within 1% of steady state | capital_next | 1.225527e-05 | 3.541031e-05 |
| NN vs linear | global grid (±3 s.d.) | consumption | 2.378174e-03 | 9.595239e-03 |
| NN vs linear | global grid (±3 s.d.) | capital_next | 8.923547e-04 | 5.539021e-03 |
| NN vs EGM | ergodic set | consumption | 1.397094e-03 | 2.968717e-03 |
| NN vs EGM | ergodic set | capital_next | 1.022310e-04 | 2.264049e-04 |
| linear vs EGM | ergodic set | consumption | 8.601131e-04 | 3.995862e-03 |
| linear vs EGM | ergodic set | capital_next | 1.628711e-04 | 1.925601e-03 |
| NN vs EGM | global grid (±3 s.d.) | consumption | 1.049561e-03 | 2.962963e-03 |
| NN vs EGM | global grid (±3 s.d.) | capital_next | 7.703076e-05 | 2.255653e-04 |
| linear vs EGM | global grid (±3 s.d.) | consumption | 1.825830e-03 | 8.985272e-03 |
| linear vs EGM | global grid (±3 s.d.) | capital_next | 9.083234e-04 | 5.555928e-03 |
| NN vs EGM | within 1% of steady state | consumption | 7.153806e-04 | 1.137463e-03 |
| NN vs EGM | within 1% of steady state | capital_next | 5.183987e-05 | 8.259297e-05 |
| linear vs EGM | within 1% of steady state | consumption | 7.843922e-04 | 7.912326e-04 |
| linear vs EGM | within 1% of steady state | capital_next | 5.863338e-05 | 6.606402e-05 |

### Euler errors

| method | domain | mean log10 |e| | max log10 |e| | max |e| |
| --- | --- | --- | --- | --- |
| nn | within 1% of steady state | -4.789 | -4.149 | 7.088256e-05 |
| egm | within 1% of steady state | -5.444 | -5.238 | 5.782855e-06 |
| linear | within 1% of steady state | -4.013 | -3.999 | 1.003301e-04 |
| nn | ergodic set | -4.005 | -3.459 | 3.479249e-04 |
| egm | ergodic set | -5.278 | -4.687 | 2.057117e-05 |
| linear | ergodic set | -3.843 | -2.847 | 1.421597e-03 |

### Runtime

| stage | seconds |
| --- | --- |
| endogenous-grid time iteration | 29.85 |
| Adam | 24.13 |
| L-BFGS | 6.51 |
| total | 64.25 |
