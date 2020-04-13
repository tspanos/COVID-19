from datetime import datetime
from io import BytesIO
from typing import List, Callable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " \
             "Chrome/76.0.3809.100 Safari/537.36"


def download_data(url: str):
    """
    Downloads data from a URL.

    :param url: URL to download data from.
    :type url: str
    """
    request = Request(url, headers={'User-Agent': USER_AGENT})
    url_data = urlopen(request)

    return url_data.read()


def get_dataframe_from_internet_csv(url: str) -> pd.DataFrame:
    """
    Downloads CSV from internet and converts it to a dataframe.

    :param url: URL where the CSV exists.
    :type url: str
    :return: Dataframe of the CSV.
    :rtype: pd.DataFrame
    """

    downloaded = download_data(url)
    csv_bytes = BytesIO(downloaded)

    return pd.read_csv(csv_bytes)


def get_dates(df: pd.DataFrame, date_format: str = '%m/%d/%y') -> List[str]:
    """
    Attempts to retrieve column names from a dataframe that are dates.

    :param df: DataFrame that may contain dates as column headers.
    :type df: pd.DataFrame
    :param date_format: A date format to try to parse out of each column header. Default = `'%m/%d/%y'`.
    :type date_format: str
    :return: List of column names that are dates.
    :rtype: list
    """

    date_columns = []
    non_date_columns = []
    for column in df.columns.to_list():
        try:
            datetime.strptime(column, date_format)
            date_columns.append(column)
        except ValueError:
            non_date_columns.append(column)

    date_columns.sort(key=lambda date: datetime.strptime(date, date_format))

    return date_columns


def convert_running_total_to_new_daily(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Converts running totals to deltas. Subtracts values from previous date to get new daily for each date.

    `3/9/20 = 3/9/20 - 3/8/20`,

    `3/8/20 = 3/8/20 - 3/7/20`,

    etc...

    :param df: pandas Dataframe
    :type df: pd.DataFrame
    :param date_columns: Column header for the dates in ascending order.
    :type date_columns: list
    :return: Dataframe with new daily values.
    :rtype: pd.DataFrame
    """

    temp_df = pd.DataFrame.copy(df, deep=True)
    # temp_df[date_columns] = temp_df[date_columns].diff(axis=1) #results in NaNs in 1st date
    for i in range(len(date_columns) - 1):
        temp_df[date_columns[i]] = temp_df[date_columns[i]] - temp_df[date_columns[i + 1]]

    return temp_df


def swap_places(df: pd.DataFrame, sources: List[str], source_column: str, destination_column: str,
                null_source: bool = False):
    """
    Swaps the value of the source with the destination. Optionally, nulls out the source.

    :param df: A pandas DataFrame.
    :type df: pd.DataFrame
    :param sources: List of sources in the source column.
    :type sources: list
    :param source_column: Source column name.
    :type source_column: str
    :param destination_column: Destination column name.
    :type destination_column: str
    :param null_source: After placing the source in the destination, the original source location is nulled.
    :return: Dataframe with swapped values.
    :rtype: pd.DataFrame
    """

    temp_df = pd.DataFrame.copy(df, deep=True)
    for source in sources:
        # Set destination to source
        temp_df.loc[temp_df[source_column] == source, destination_column] = source
        if null_source:
            # Set source to NaN
            temp_df.loc[temp_df[source_column] == source, source_column] = np.nan
        else:
            temp_df.loc[temp_df[source_column] == source, source_column] = destination_column

    return temp_df


def group_by(df: pd.DataFrame, columns_to_group_by: List[str], sum_columns: List[str]) -> pd.DataFrame:
    """
    Groups columns of a dataframe. Sums values in sum columns.

    :param df: Pandas Dataframe.
    :type df: pd.DataFrame
    :param columns_to_group_by: The columns on which to perform the group by (the ones to keep).
    :type columns_to_group_by: list
    :param sum_columns: The columns that contain values to retain and sum into the group (row).
    :type sum_columns: list
    :return: Dataframe with grouping.
    :rtype: pd.DataFrame
    """

    return df.groupby(columns_to_group_by)[sum_columns].sum().reset_index()


def melt(df: pd.DataFrame, columns_to_melt: List[str], melted_column: str, melted_value: str):
    """
    Melts a dataframe. Useful to convert a wide dataframe to a narrow dataframe.

    :param df: Pandas dataframe.
    :type df: pd.DateFrame
    :param columns_to_melt: The columns which are to be melted into one column.
    :type columns_to_melt: list
    :param melted_column: The name of the new column which categorizes the melted column. The 'variable'.
    :param melted_value: The name of the new column which contains the values from melting. The 'value'.
    :type melted_value: str
    :return: Melted dataframe.
    :rtype: pd.DataFrame
    """

    # Find which columns are ID columns to use for melting. Non ID columns get melted.
    existing_columns = set(df.columns.to_list())
    melting_columns = set(columns_to_melt)
    id_columns = list(existing_columns - melting_columns)

    melted = df.melt(id_vars=id_columns)
    melted = melted.rename(columns={'variable': melted_column, 'value': melted_value})

    return melted


def merge(*args) -> pd.DataFrame:
    """
    Left joins dateframes in sequence.

    :param args: A bunch of dataframes to left join.
    :type args: pd.DateFrame
    :return: All the dataframes left joined.
    :rtype: pd.DataFrame
    """

    merged = args[0]
    for i in range(len(args) - 1):
        merged = pd.merge(merged, args[i + 1], how='left')

    return merged


def convert_wide_cssegi_to_narrow(df: pd.DataFrame, dataset_name: str, keep_running_total=True,
                                  promote_region: bool = False, regions_to_promote: List[str] = None,
                                  lower_region_column=None, higher_region_column=None, group: bool = False,
                                  group_columns: List[str] = None, date_format='%m/%d/%y') -> pd.DataFrame:
    """
    General function to convert a Wide CSSEGISandData Covid-19 Dataframe to narrow with new daily values in place of
    running totals. Introduces two columns: `Date` and `dateset_name`. When `keep_running_total` is True, introduces new
    column '`dataset_name` to Date'.

    :param df: Wide CSSEGISandData Covid-19 Dataframe.
    :type df: pd.DataFrame
    :param dataset_name: The name of the Dataset. Will be used as the name of the new value column which will contain
                         the narrowed values. Example, 'Confirmed'.
    :type dataset_name: str
    :param keep_running_total: Keeps a column that contains the running total up to that date. Names it
                               `dataset_name to Date`.
    :param promote_region: Moves regions into a higher tier column. Nulls out its previous value. Useful for making
                           countries that were assigned in the original dataset as provinces of another country into
                           countries of their own.
    :type promote_region: bool
    :param regions_to_promote: A list of regions to promote.
    :type regions_to_promote: bool
    :param lower_region_column: Name of the lower region column. Required if `promote_region` is True.
    :type lower_region_column: str
    :param higher_region_column: Name of the higher region column. Required if `promote_region` is True.
    :type higher_region_column: str
    :param group: Group values by a list of columns. Useful if a region has many sub regions that you want to suck in.
    :type group: bool
    :param group_columns: The columns to group by on. Required if `group` is True.
    :type group_columns: str
    :param date_format: The date format to use to convert date columns to datetime.
    :type date_format: str
    :return: A narrow dataframe.
    :rtype: pd.DataFrame
    """

    prepared = pd.DataFrame.copy(df)
    date_columns = get_dates(prepared)
    date_columns.reverse()

    if promote_region:
        required = [regions_to_promote, lower_region_column, higher_region_column]
        if None in required:
            raise ValueError(f'No value in {required} can be None')
        prepared = swap_places(df=prepared, sources=regions_to_promote, source_column=lower_region_column,
                               destination_column=higher_region_column)

    if group:
        required = [group_columns]
        if None in required:
            raise ValueError(f'No value in {required} can be None')
        prepared = group_by(df=prepared, columns_to_group_by=group_columns, sum_columns=date_columns)

    running_totals = melt(prepared, date_columns, 'Date', f'{dataset_name} to Date')
    prepared = convert_running_total_to_new_daily(df=prepared, date_columns=date_columns)
    prepared = melt(df=prepared, columns_to_melt=date_columns, melted_column='Date', melted_value=dataset_name)

    # Add the running totals as an additional column to the melted data
    if keep_running_total:
        prepared = merge(prepared, running_totals)

    prepared['Date'] = pd.to_datetime(prepared['Date'], format=date_format)

    return prepared


def combine_narrow(prepare_function: Callable, confirmed_url: str, deaths_url: str,
                   recovered_url: str = None) -> pd.DataFrame:
    """
    Used to perform conversion of  Wide CSSEGISandData Covid-19 datasets.

    :param prepare_function: Callable that accepts a Wide CSSEGISandData Covid-19 dataframe, a dataset name and returns
                             a dataframe.
    :type prepare_function: callable
    :param confirmed_url: URL of Wide CSSEGISandData Covid-19 Confirmed CSV.
    :type confirmed_url: str
    :param deaths_url: URL of Wide CSSEGISandData Covid-19 Deaths CSV.
    :type deaths_url: str
    :param recovered_url: URL of Wide CSSEGISandData Covid-19 Recovered CSV. Optional.
    :type recovered_url: str
    :return: A narrow dataframe
    :rtype: pd.Dataframe
    """

    confirmed = get_dataframe_from_internet_csv(confirmed_url)
    deaths = get_dataframe_from_internet_csv(deaths_url)

    assembled = merge(prepare_function(confirmed, 'Confirmed'), prepare_function(deaths, 'Deaths'))
    columns_to_convert = ['Confirmed', 'Deaths']

    if recovered_url:
        recovered = get_dataframe_from_internet_csv(recovered_url)
        assembled = merge(assembled, prepare_function(recovered, 'Recovered'))
        columns_to_convert.append('Recovered')

    assembled[columns_to_convert] = assembled[columns_to_convert].astype('int64')

    return assembled
