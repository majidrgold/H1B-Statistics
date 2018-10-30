# Table of Contents
1. [Project Title](README.md#project-title)
2. [Project Statement](README.md#project-statement)
3. [Getting Started](README.md#getting-started)
4. [Running the Tests](README.md#running-the-tests)
5. [Output Structure](README.md#out-put-structure)



# Project Title

##### **H1B Statistic**

# Project Statement

This project processes immigration data trends on H1B(H-1B, H-1B1, E-3) visa application and provides the occupations and states with the most number of approved H1B visas. Visa application data is available on the [Office of Foreign Labor Certification Performance Data](https://www.foreignlaborcert.doleta.gov/performancedata.cfm#dis) from the US Department of Labor.  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. Raw input data could be found [here](https://www.foreignlaborcert.doleta.gov/performancedata.cfm) under the **Disclosure Data** tab. The data is in ".xlsx" format whcih should be converted to csv file with semicolon ";" separation to be used in this program.

### Running the Tests

The **src** directory contains the source python code named **"Data_insight_h1b.py"**. The program is written in Python3. The input csv file data should be placed in the **input** directory under the name of **h1b_input.csv**. Then by runing "**run.sh**" file the results will be created and located at the **output** directory. The **output** contains `top_10_occupations.txt` and `top_10_occupations.txt` files. 

### Output Structure

 The `top_10_occupations.txt` file shows top 10 occupations certified for H1B visas and the `top_10_occupations.txt` lists top 10 states for certified visa applications.
 Each line of the `top_10_occupations.txt` file should contain these fields in this order:
1. **`TOP_OCCUPATIONS`**: Use the occupation name associated with an application's Standard Occupational Classification (SOC) code
2. **`NUMBER_CERTIFIED_APPLICATIONS`**: Number of applications that have been certified for that occupation. An application is considered certified if it has a case status of `Certified`
3. __`PERCENTAGE`__: % of applications that have been certified for that occupation compared to total number of certified applications regardless of occupation. 



Each line of the `top_10_states.txt` file should contain these fields in this order:
1. **`TOP_STATES`**: State where the work will take place
2. **`NUMBER_CERTIFIED_APPLICATIONS`**: Number of applications that have been certified for work in that state. An application is considered certified if it has a case status of `Certified`
3. **`PERCENTAGE`**: % of applications that have been certified in that state compared to total number of certified applications regardless of state.


## Author

* **Majid Ramezani** 

