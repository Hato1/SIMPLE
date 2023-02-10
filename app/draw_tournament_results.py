import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

import argparse

# read file

def main(args) -> None:

    df = pd.read_csv(args.filename, usecols=[2,3,4])
    # Sort so base appears lexographically before _model_00001
    df = df.replace('base', 'model_00000_base')
    # Replace unnecessary suffixes
    df = df.replace(to_replace=r'^_(model_\d{5})_.+$', value=r'\1', regex=True)

    print(df.head())

    results_sq = df.pivot(index='model1', columns='model2', values='score1')

    print(results_sq.head())

    fig, ax = plt.subplots(1, 1, figsize = (18, 15))
    sb.heatmap(results_sq, annot = True, vmin=-1.0, vmax=1.0)
    plt.savefig(args.savefile)


def cli() -> None:

  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  parser.add_argument("filename", help="Data file to plot")
  parser.add_argument("savefile", help="Filename for png to save")


  #parser.add_argument("player_to_exclude", action = 'store_true', default = False
  #              , help="Number of player to exclude")

  # Extract args
  args = parser.parse_args()

  # Enter main
  main(args)
  return


if __name__ == '__main__':
  cli()