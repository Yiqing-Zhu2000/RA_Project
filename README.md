## RA_Project
Investigations into the Prediction of Patient Response to Rheumatoid Arthritis Treatment.

## Local Installation 
1. Clone the [RA_Project repository](https://github.com/Yiqing-Zhu2000/RA_Project) and open it
```
git clone git@github.com:Yiqing-Zhu2000/RA_Project.git
cd RA_Project
```
2. Optimially, set up a virtual enviroment to avoid package version conflicts between different projects
on your machine. In this case, we call it `env`:
   1. For mac:
    ```
    python3 -m venv env
    source env/bin/activate
    ```
   2. For windows:
    ```
    python -m venv env
    env\Scripts\activate
    ```
3. Make sure the base interpreter is set to Python3.7 or higher. Install the correct version of the necessary packages. Windows users should replace 'pip3' with 'pip':
```
pip3 install -r requirements.txt
```
