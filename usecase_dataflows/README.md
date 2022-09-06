# Use Case: Distributed Dataflows

To test our approach in conjunction with the two methods designed for distributed dataflows, namely [CherryPick](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/alipourfard) and [Arrow](https://ieeexplore.ieee.org/document/8416333), we re-implemented these two methods to the best of our knowledge (since we were not able to obtain the original code). Since we used these implementations already in prior work of us (see [here](https://github.com/dos-group/karasu-cloud-configuration-profiling) the original repository, for more technical details), we copied and adapted the relevant code for our particular work with Perona.

## CherryPick

We followed an opportunistic and hence simplified approach for integrating results produced with Perona into CherryPick. In principle, whenever the respective acquisition function of the Bayesian Optimization (BO) model outputs acquisition values for candidate resource configurations to profile next, we re-weight them based on our obtained Perona scores as well as application resource usage profiles gathered during / after execution. Hence, given that we have access to Perona scores already obtained in the past, we only require a single execution of a target workload (to obtain its resource usage profile) in order to employ our re-weighting scheme. This additional logic mainly manifests itself in the class `OptimizerExt` in `optimization/methods/optimizer_ext.py`, with the class itself extending the base class `Optimizer` in `optimization/methods/optimizer.py`. Further small changes to the original code were made here and there but are not relevant to understand the general integration approach.

## Arrow

We follow the same strategy for Arrow as previously described for CherryPick. In addition, we replace Arrow's original low-level metrics (which are also integrated in its modeling approach) with our Perona scores. For this extra action, we created a function in the class `ArrowExtOptimizer` located in `optimization/methods/baselines/optimizer_arrow.py`.

## Additional Remarks

We carefully created this repository and tried to make sure that results are reproducible. For this, we fixed the version of all major python packages and even created a Dockerfile, which can be used to isolate and reproduce the execution even further. For instance, after building the docker image, you can execute 
```
./docker_scripts.sh run_soo_experiment
```

to start an experiment. This will internally also take care of downloading the utilized evaluation [dataset](https://github.com/oxhead/scout), if not already locally available. The results are then written to a directory mounted into the container. 

Nevertheless, if you experience any problems, please do not hesitate to reach out to us!