# Backend

This is our custom Kubernetes operator for taking care of scheduling benchmarks, injecting chaos, collecting metrics, and regex-parsing benchmark execution results. All the data is saved into an `sqlite` database. It also facilitates an endpoint for automating the data acquisition procedure described in our paper.

All package requirements are specific in `requirements.txt`. It is recommended to install them into a virtual environment. Since our operator itself also runs within the respective target Kubernetes environment, we provide a `Dockerfile` for building the application.

