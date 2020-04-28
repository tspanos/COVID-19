import numpy as np
import pandas as pd

from helpers import convert_wide_cssegi_to_narrow, combine_narrow

GLOBAL_CONFIRMED_CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                           'csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
GLOBAL_DEATHS_CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                        'csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
GLOBAL_RECOVERED_CSV_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' \
                           'csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

COUNTRY_DATA_XLSX = 'data/Country Data.xlsx'
COUNTRY_DATA_SHEET = 'Countries'
COUNTRY_DATA_COLUMNS = ['Population', 'Physicians per thousand', 'GDP']

GLOBAL_PROVINCE_COL = 'Province/State'
GLOBAL_COUNTRY_COL = 'Country/Region'
KEEP_AND_RENAME_COLUMNS = {'Country/Region': 'Country'}
DATE_FORMAT = '%m/%d/%y'

CONVERT_PROVINCE_TO_COUNTRY = ['Hong Kong', 'Faroe Islands', 'Greenland', 'French Guiana', 'French Polynesia',
                               'Guadeloupe', 'Mayotte', 'New Caledonia', 'Reunion', 'Saint Barthelemy', 'St Martin',
                               'Martinique', 'Saint Pierre and Miquelon', 'Aruba', 'Curacao', 'Sint Maarten',
                               'Bonaire, Sint Eustatius and Saba', 'Bermuda', 'Cayman Islands', 'Channel Islands',
                               'Gibraltar', 'Isle of Man', 'Montserrat', 'Anguilla', 'British Virgin Islands',
                               'Turks and Caicos Islands', 'Falkland Islands (Malvinas)']
COUNTRIES_TO_RENAME = {'Taiwan*': 'Taiwan', 'West Bank and Gaza': 'Palestine', 'Korea, South': 'South Korea'}


def prepare_global_df(df, dataset_name):
    """
    Prepares a 'global' dataframe.

    :param df: Global Wide CSSEGISandData Covid-19 Dataframe.
    :type df: pd.DataFrame
    :param dataset_name: The name of the type of dataset. Confirmed, Deaths, Etc. Used a new column in narrow output.
    :type dataset_name: str
    :return: A narrow version of the Wide CSSEGISandData Covid-19 Dataframe.
    :rtype: pd.DataFrame
    """
    prepared = convert_wide_cssegi_to_narrow(df=df, dataset_name=dataset_name, promote_region=True,
                                             regions_to_promote=CONVERT_PROVINCE_TO_COUNTRY,
                                             lower_region_column=GLOBAL_PROVINCE_COL,
                                             higher_region_column=GLOBAL_COUNTRY_COL,
                                             group=True,
                                             group_columns=list(KEEP_AND_RENAME_COLUMNS.keys()))
    prepared = prepared.sort_values(by=[GLOBAL_COUNTRY_COL, 'Date'], ascending=[True, False])

    return prepared


def generate_global_dataset(output_path: str) -> None:
    """
    Generates a CSV file containing narrowed and merged Global Wide CSSEGISandData Covid-19 date. Introduces some
    additional columns for reporting.

    :param output_path: Path to write the CSV file out to.
    :type output_path:str
    :return: Writes file to `output_path`. Returns nothing.
    :rtype: None
    """

    global_cases = combine_narrow(prepare_global_df, GLOBAL_CONFIRMED_CSV_URL, GLOBAL_DEATHS_CSV_URL,
                                  GLOBAL_RECOVERED_CSV_URL)
    # Rename Country Colummn
    global_cases = global_cases.rename(columns=KEEP_AND_RENAME_COLUMNS)
    # Adjust Country Names
    global_cases['Country'] = global_cases['Country'].replace(COUNTRIES_TO_RENAME)
    # Bring in country data
    country_data = pd.read_excel('data/Country Data.xlsx', sheet_name='Countries')
    # Drop Countries not in Country Data
    # global_cases[global_cases['Country'].isin(country_data['Country'])]
    in_country_data = pd.merge(left=global_cases, right=country_data, how="outer", indicator=True)
    not_in_country_data = in_country_data[in_country_data['_merge'] == 'left_only']
    global_cases = global_cases[~global_cases['Country'].isin(not_in_country_data['Country'])]
    print(f"The following countries were dropped as additional population level data could not be found.\r\n "
          f"{not_in_country_data['Country'].unique()}")
    # Make additional columns that require country data
    global_cases = pd.merge(left=global_cases, right=country_data)
    global_cases['Growth Rate'] = (
            global_cases['Confirmed'] / (global_cases['Confirmed to Date'] - global_cases['Confirmed'])).replace(
        {np.inf: 0, np.NaN: 0})
    global_cases['Infected Percentage'] = global_cases['Confirmed'] / global_cases['Population']
    global_cases['Infected per Million'] = global_cases['Infected Percentage'] * 1000000
    global_cases['Net Cases'] = (global_cases['Confirmed'] - global_cases['Deaths'] - global_cases['Recovered'])
    global_cases['Healthcare System Saturation'] = global_cases['Net Cases'] / (
            (global_cases['Population'] / 1000) * global_cases['Physicians per thousand'])

    global_cases = global_cases.sort_values(by=['Country', 'Date'], ascending=[True, False])
    global_cases = global_cases.drop(columns=COUNTRY_DATA_COLUMNS)
    global_cases.to_csv(output_path, index=False)
