# perona-infrastructure-fingerprinting

Prototypical implementation of "Perona" for explicit infrastructure fingerprinting. Please definitely consider reaching out if you have questions or encounter problems. We are happy to help anytime!

This repository contains several subdirectories, featuring the following content:

- `data`: The data we recorded during our experiments, or, to be more precise, needed for our evaluation. Note that we compressed the data into an archive, so in order to use it, you first need to extract it using the command: `tar xfvz data.tar.gz`.
- `data_acquisition`: Ansible scripts, deployment, configuration of benchmarks and chaos experiments, and our implemented Kubernetes Operator to simplify the overall data gathering process.
- `method`: The actual code with respect to the modeling approach, i.e., data preprocessing, encoding, and outlier detection & ranking of benchmark runs.
- `usecase_dataflows`: All the code necessary for our use case involving dataflows.
- `usecase_workflows`: Instructions / Remarks regarding our use case involving workflows.
