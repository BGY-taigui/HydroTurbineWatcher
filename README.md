# Installation

## Cloning Repository from Github


## Create Developing Environment

Clone this git repository and change directory
```
cd HydroTurbineWatcher
```
Create virtual environment using "venv" module, and activate it.
Any virtual environment name is OK.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Install commands as a module

Install this Project as a pip module in developing mode
(Changes are immediately applied)
```
pip install -e .
```

Install this Project as a pip module in static mode
```
pip install .
```