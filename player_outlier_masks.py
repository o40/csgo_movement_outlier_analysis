
player_outlier_mask_dict = {}

# Based on: https://www.hltv.org/download/demo/44519
player_outlier_mask_dict["forsaken"] = [
    0, 1, 1, 1, 0, 0, 1, 1, 0, 0, # 00-09
    1, 1, 0, 0, 1, 1, 0, 0, 1, 1, # 10-19
    0, 0, 0, 1, 1, 0, 0, 1, 1, 0, # 20-29
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1, # 30-39
    1, 0, 0, 0, 1, 1, 0, 0, 1, 1, # 40-49
    0, 0, 1, 1, 0, 0, 0, 1, 0, 0, # 50-59
    1, 1, 1, 0, 0, 1, 1, 0, 0, 0, # 60-69
    1]                            # 70-79

# Based on: https://www.hltv.org/download/demo/35960 (BOT JW)
player_outlier_mask_dict["JW"] = [0, 1, 1, 1, 1, 1, 0, 0]
