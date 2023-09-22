# Clean Code

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://python.org/)


## Overview

<Description>

## Contents

The project contains the following:

```text
.gitignore          - this will ignore vagrant and other metadata files
.flaskenv           - Environment variables to configure Flask
.gitattributes      - File to gix Windows CRLF issues
.devcontainers/     - Folder with support for VSCode Remote Containers
dot-env-example     - copy to .env to use environment variables
requirements.txt    - list if Python libraries required by your code
config.py           - configuration parameters

deploy/                    - K8s deployment files
├── deployment.yaml        - Deployment
├── postgresql.yaml        - PostgreSQL
└── service.yaml           - Service

docs/                      - Github Pages (Example)
├── doc.txt                - 
└── index.html             - AirBnB Data Analysis Blog

notebooks/                - Jupyter Notebooks
└── data                                          

service/                   - service python package
├── __init__.py            - package initializer
├── models.py              - module with business models
├── routes.py              - module with service routes
└── common                 - common code package
    ├── error_handlers.py  - HTTP error handling code
    ├── log_handlers.py    - logging setup code
    └── status.py          - HTTP status constants

tests/              - test cases package
├── __init__.py     - package initializer
├── factories.py    - factory classes to generate test data
├── test_models.py  - test suite for business models
└── test_routes.py  - test suite for service routes
```

```bash
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ conda env list
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ export PS1="[\[\033[01;32m\]\u\[\033[00m\]: \[\033[01;34m\]\W\[\033[00m\] \$(date '+%d %b %y %H:%M')]\$ "
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ cd notebooks/
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ conda create --name p39_cc python=3.9
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ conda activate p39_cc
(p39_cc) [vscode: notebooks 20 Sep 23 01:12]$ pip install -r project/requirements_py3_9.txt
```


## License

Licensed under the Apache License. See [LICENSE](LICENSE)

