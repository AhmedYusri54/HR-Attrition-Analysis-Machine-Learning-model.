import pandas as pd
def wrangle(data_frame: pd.DataFrame):
    """Preprocessing funcation to can modify and imporve the dataset for modelling.
    Args:
    -----
    data_frame (pd.DataFrame): The dataframe that wanted to been processed
    
    Returns:
    -------
    data_frame (pd.DataFrame): The modfied version of the dataframe that had been imporved for modelling.
    """
    # 1. Remove the unusable columns and those that case multicollinearity to imporve data quality
    data_frame.drop(columns=["YearsInCurrentRole", "YearsWithCurrManager", "YearsAtCompany", "EmployeeCount", "JobLevel", "Over18", "StandardHours", "MonthlyIncome", "PerformanceRating", "EmployeeNumber"], inplace=True)
    
    # 2. Map the categorical features 
    data_frame["Attrition"] = data_frame["Attrition"].map({"Yes": 1, "No": 0})
    data_frame["Gender"] = data_frame["Gender"].map({"Female": 0, "Male": 1})
    data_frame["OverTime"] = data_frame["OverTime"].map({"Yes": 1, "No": 0})
    
    # 3. Encoding the not binary features 
    data_frame = pd.get_dummies(data_frame, columns=["BusinessTravel", "JobRole", "Department", "EducationField", "MaritalStatus"], drop_first=True, dtype="uint8")
    
    return data_frame
