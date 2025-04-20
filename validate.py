import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

def load_clusters(path):
    df = pd.read_csv(path, usecols=['id','cluster']).set_index('id')
    return df['cluster']

def main(dirpath):
    files = sorted(Path(dirpath).glob('*.csv'))
    clusters = {}
    for f in files:
        clusters[f.name] = load_clusters(f)
    # group by identical Series
    groups = defaultdict(list)
    for name, series in clusters.items():
        key = tuple(series.values)
        groups[key].append(name)
    # report
    for eq, names in groups.items():
        print(f"Group ({len(names)} files):")
        for n in names:
            print("   ", n)
        print()

if __name__=='__main__':
    if len(sys.argv)!=2:
        print("Usage: validate.py <csv_directory>")
        sys.exit(1)
    main(sys.argv[1])
