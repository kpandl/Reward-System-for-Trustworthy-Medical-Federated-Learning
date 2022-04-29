import string
import os
import time
from pathlib import Path
import shutil

final_path_list = ["_reduced_maxtranslation_0.1final_100_0", "_reduced_maxtranslation_0.1final_50_50", "_reduced_maxtranslation_0.1final_as_is", "_reduced_maxtranslation_0.1final_75_25", "_reduced_maxtranslation_0.1final_age_quantile_100_0", "_reduced_maxtranslation_0.1final_age_quantile_75_25", "_reduced_maxtranslation_0.1final_age_quantile_50_50", "_reduced_maxtranslation_0.1final_age_as_is"]

file_names_to_copy = ["Shapley.csv", "Shapley_performance.csv", "testing.csv"]
file_names_to_copy_optional = ["Shapley_age.csv"]

path = Path(os.path.join(os.getcwd(), "results"))

my_list = os.listdir(path)

path = Path(os.path.join(os.getcwd(), "results", "compact_files"))

Path(path).mkdir(parents=True, exist_ok=True)

problematic_dirs = []

for final_path in final_path_list:
    i = 0
    folder_exists=True
    while folder_exists:
        folder_name = "default" + str(i) + final_path
        folder_exists = os.path.isdir(os.path.join(os.getcwd(), "results", folder_name))
        if(folder_exists):
            Path(os.path.join(os.getcwd(), "results", "compact_files", folder_name)).mkdir(parents=True, exist_ok=True)
            for file_name in file_names_to_copy:
                path_origin = os.path.join(os.getcwd(), "results", folder_name, file_name)
                file = Path(path_origin)
                if file.is_file():
                    shutil.copyfile(path_origin, os.path.join(os.getcwd(), "results", "compact_files", folder_name, file_name))
                else:
                    if(folder_name not in problematic_dirs):
                        problematic_dirs.append(folder_name)
            for file_name in file_names_to_copy_optional:
                path_origin = os.path.join(os.getcwd(), "results", folder_name, file_name)
                file = Path(path_origin)
                if file.is_file():
                    shutil.copyfile(path_origin, os.path.join(os.getcwd(), "results", "compact_files", folder_name, file_name))

        i+=1
        if(i == 40):
            folder_exists = False

print("problems occured in")
for folder in problematic_dirs:
    print(folder)
print("finished")