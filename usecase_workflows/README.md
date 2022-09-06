# Use Case: Scientific Workflows

To test our approach in conjunction with the two methods designed for scientific workflows, namely [Lotaru](https://dl.acm.org/doi/abs/10.1145/3538712.3538739) and [Tarema](https://ieeexplore.ieee.org/document/9671519/), we used their available code and modified it according to our needs. Precisely, we used the following code repositories:
- Lotaru: https://github.com/CRC-FONDA/lotaru
- Tarema: https://github.com/CRC-FONDA/tarema-cluster-profiler

## Lotaru

We adjusted the `defineFactor` method (see [here](https://github.com/CRC-FONDA/Lotaru/blob/main/src/main/java/Main.java)), included our own scores obtained from our learned representations, and equally weighted CPU and memory. Further, we extended the `Estimator` interface (click [here](https://github.com/CRC-FONDA/Lotaru/blob/main/src/main/java/estimators/Estimator.java)) in order to use the adjusted `defineFactor` method with our own Perona-specific estimator.

## Tarema

We mocked Tarema's group-building process with our fingerprinting values from the N1, N2, and C2 GCP nodes. As a result, the same groups as with plain Tarema were formed, indicating that also the same results would be produced.