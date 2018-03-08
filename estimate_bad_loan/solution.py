import pandas as pd
import numpy as np
import sys


def get_total_valid_days(chargeoff_day, loan_age):
    didnt_chargeoff = chargeoff_day.isnull()
    # because the days are 0-indexed, chargeoff_days == the number of days they had the loan
    # without charging of, while if they haven't yet charged off, the total number of valid 
    # days is loan_age + 1
    no_chargeoff_valid_days = loan_age + 1
    valid_days = no_chargeoff_valid_days.where(didnt_chargeoff, chargeoff_day)
    total_valid_days = valid_days.sum()
    return total_valid_days

def get_total_num_chargeoffs(chargeoff_day):
    return chargeoff_day.count()

def get_daily_chargeoff_prob(total_valid_days, total_num_chargeoffs):
    return total_num_chargeoffs / total_valid_days

def get_prob_no_chargeoff(total_loan_days, daily_chargeoff_prob):
    return (1.0 - daily_chargeoff_prob)**total_loan_days

def get_prob_chargeoff(file_name, num_years = 3, days_per_year = 365):
    total_loan_days = num_years * days_per_year
    timing = pd.read_csv(file_name)
    chargeoff_day = timing.ix[:,1]
    loan_age = timing.ix[:,0]
    total_valid_days = get_total_valid_days(chargeoff_day, loan_age)
    print total_valid_days
    total_num_chargeoffs = get_total_num_chargeoffs(chargeoff_day)
    daily_chargeoff_prob = get_daily_chargeoff_prob(total_valid_days, total_num_chargeoffs)
    prob_no_chargeoff = get_prob_no_chargeoff(total_loan_days, daily_chargeoff_prob)
    return 1 - prob_no_chargeoff

def main():
    if (len(sys.argv) < 2):
        print("please specify the path to the file")
    else:
        print(get_prob_chargeoff(sys.argv[1]))

if __name__ == "__main__":
    main()
