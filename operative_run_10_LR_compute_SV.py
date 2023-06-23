import os
import pickle
import itertools
import numpy as np
import csv
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--name_of_test_folder', nargs='?', default="default100_reduced_maxtranslation_0.1_3c", type=str, help='Specify the name of the test folder')
parser.add_argument('--measure', nargs='?', default="predictive_performance", type=str, help='Specify the measure')
parser.add_argument('--set_genderclients', dest='set_genderclients', action='store_true')
parser.set_defaults(set_genderclients=False)

args=parser.parse_args()

folder_path = args.name_of_test_folder
#folder_path = "default0_reduced_maxtranslation_0.1final_age_as_is"

measure = "predictive_performance"
#measure = "bias_gender"

measure = args.measure

#measure = "bias_age"
#measure = "predictive_performance_age"
method = "LR_method"

write_results = True

measure_filename = measure
if(measure=="predictive_performance_age"):
    measure_filename = "predictive_performance"
file_original_method = os.path.join(os.getcwd(), "results", folder_path, "Shapley_" + method + "_" + measure_filename + "_coalition_utilities.pkl")

path_Shapley_results_file = os.path.join(os.getcwd(), "results", folder_path, "Shapley_" + method + "_" + measure + ".csv")
if(os.path.exists(path_Shapley_results_file)):
    os.remove(path_Shapley_results_file)

with open(file_original_method, 'rb') as f:
    utilities = pickle.load(f)

if(write_results):
    with open(path_Shapley_results_file, 'a') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(["dataset", "testing critereon", "operating point", "Shapley value"])


clients = ["chexpert_f", "chexpert_m", "nih_f", "nih_m", "mimic_f", "mimic_m"]
if((measure=="predictive_performance_age" or measure=="bias_age") and not args.set_genderclients):
    clients = ["chexpert_young", "chexpert_old", "nih_young", "nih_old", "mimic_young", "mimic_old"]

if(measure=="predictive_performance" or measure=="predictive_performance_age"):
    utility_empty_coalition = 0.5
else:
    utility_empty_coalition = 0

clients_subset = []

def utility_coalition(clients):
    clients_sorted = sorted(clients)
    clients_sorted_string = "_".join(clients_sorted)
    a = 0
    if(clients_sorted_string == ''):
        return utility_empty_coalition
    else:
        return utilities[clients_sorted_string]

for L in range(len(clients) + 1):
    clients_subset_length_L = list(itertools.combinations(clients, L))
    clients_subset += clients_subset_length_L

SVs = []

for client in clients:
    marginal_contributions_for_client = {}
    for coalition in clients_subset:
        if client not in coalition:
            coalition_size = len(coalition) + 1
            marginal_contribution = utility_coalition(list(coalition) + [client]) - utility_coalition(coalition)
            a = 0
            marginal_contributions_for_client.setdefault(coalition_size, []).append(marginal_contribution)
            b = 0

    averages = []
    for key in marginal_contributions_for_client.keys():
        averages.append(np.mean(marginal_contributions_for_client[key]))

    SV = np.mean(averages)
    print("SV for client " + client + " is " + str(SV))

    if(write_results):
        with open(path_Shapley_results_file, 'a') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([client, "AUC", "-", SV])

    SVs.append(SV)

    a = 0

print("sum of SVs is " + str(sum(SVs)))

a = 0