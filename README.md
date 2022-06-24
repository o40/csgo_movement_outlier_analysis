# csgo_movement_outlier_analysis

# Download the replay you want to examine

I recommend using one with a known cheater, such as:

https://www.hltv.org/matches/2327868/revolution-vs-optic-india-extremesland-2018-asia-finals

Unpack the .dem file to the replays folder

# Build the parser

go.exe build replay_parser.go

# Extract some data

./replay_parser.exe -demo=replays/revolution-vs-optic-india-m2-cache.dem > csv/revolution-vs-optic-india-m2-cache.csv

# Run the tool to generate histograms and print ranges where outliers resides

py -3 get_outlier_ranges.py --csv csv/revolution-vs-optic-india-m2-cache.csv --generate-images
