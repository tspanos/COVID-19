import argparse

from worldwide import generate_global_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Wide CSSEGISandData Covid-19 to Narrow",
                                     description='Downloads a ide CSSEGISandData Covid-19 datasets and makes them '
                                                 'narrow. Additionally, adds some columns for reporting. '
                                                 'Outputs a CSV.')
    parser.add_argument('-output', action='store', default='global.csv', required=False, help='Output CSV path.')
    args = parser.parse_args()

    output_path = args.output

    generate_global_dataset(output_path=output_path)
