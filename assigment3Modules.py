import numpy as np
import matplotlib.pyplot as plt

# Part2
def read_csv_to_numpy_array(filename):
    return np.genfromtxt(filename, delimiter=',', dtype=np.uint, skip_header=1)

cph_data = read_csv_to_numpy_array("befkbhalderstatkode.csv")

# Part3
def find_eng_and_non_eng_speaking_countries(data, year=2015):
    # All the native english speaking contries in the world
    english_speaking_countries = [5170, 5309, 5502, 5303, 5305, 5526, 5314, 5326, 5339, 5308, 5142,
                                  5352, 5514, 5625, 5347, 5311, 5374, 5390]
    mask_year = (data[:, 0] == year)
    total = np.sum(data[mask_year][:, 4])
    eng_mask = mask_year & np.in1d(data[:, 3], english_speaking_countries)

    eng_amount = np.sum(data[eng_mask][:, 4])
    non_eng_amount = total - eng_amount
    return (eng_amount, non_eng_amount)

# Part4
def filter(data, mask):
    return data[mask]

# Part 5
from enum import Enum
class Data_Picker(Enum):
    YEAR = 0
    AREA = 1
    AGE = 2
    COUNTRY_CODE = 3
    PERSON_AMOUNT = 4

def filter_sum(data, data_picker):
    return sum(data[:, data_picker.value])

# Part 6
def get_amount_over_year(data, start=1992, end=2016):
    year_sum = {}

    for x in range(start, end):
        amount = sum(data[(data[:, 0] == x)][:, 4])   
        year_sum.update({x: amount})
    return year_sum

# Part 7
def country_over_time(data, c_code):
    mask = (data[:, 3] == c_code)
    country_data = data[mask]
    # Dict comprehension with all years as keys and sum of persons as value
    return {year: np.sum(country_data[(country_data[:, 0] == year)][:, 4]) for year in country_data[:, 0]}

# Part 8
def get_data_by_age(data, start, end, year=2015):
    mask = (data[:, 0] == year) & (data[:, 2] >= start) & (data[:, 2] <= end)
    age_data = data[mask]
    # Dict comprehension with all ages as keys and sum of persons as value
    return {age: np.sum(age_data[(age_data[:, 2] == age)][:, 4]) for age in age_data[:, 2]}

# Part 9.1
def get_sum_by_age_group(data, age_group, area, year=2015):
    mask = (data[:, 0] == year) & np.in1d(
        data[:, 2], age_group) & (data[:, 1] == area)
    return np.sum(data[mask][:, 4])

def create_age_group_dict(data, age_group, area_code):
    age_group_dict = {}
    for index, val in enumerate(age_group):
        age_group_dict.update(
            {index: get_sum_by_age_group(data, val, area_code)})
    return age_group_dict

def create_labels_and_explode(age_group_dict):
    # Creates labels for all ranges
    labels = [str(val * 10) + "-" + str(val * 10 + 9)
              for index, val in enumerate(age_group_dict.keys())]
    explode = tuple([0.1] * len(age_group_dict.keys()))
    return labels, explode
