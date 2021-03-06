# CSSE COVID-19 Dataset Tools and Visualization

* Downloads wide Confirmed, Deaths and Recovered CSVs from https://github.com/CSSEGISandData/COVID-19.
* Merges and makes datasets narrow. Outputs as single CSV.
* Introduces new daily values in addition to running totals.
* Renames some countries.
* Joins to country data from Bing (generated by Microsoft Excel intelligence).
* Drops countries where Physician per Capita statistics could not be found.
* Adds Columns:
    - `Growth Rate`
    - `Infected Percentage`
    - `Infected per Million`
    - `Net Cases`
    - `Healthcare System Saturation`

### Visualization
This is used as ETL for a [Power BI COVID-19 visualization](https://app.powerbi.com/view?r=eyJrIjoiN2M3NmI3MDctMjgyNS00OGRkLWJjMWItZTkyYTJmYTJhMDgzIiwidCI6IjVkZmY2MmYyLWEyN2YtNDdhYi05YTI2LTJkNjkwOWNmOWVlZSJ9).

![](https://github.com/tspanos/COVID-19/raw/master/media/power_bi_covid-19.png)

### Usage
Clone or download this repo. Python 3.7+ is required to use.

Install requirements:
```shell script
pip install -r requirements.txt
```

Run main script. Specify output path.
```shell script
python main.py -output "global.csv"
```