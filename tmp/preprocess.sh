#!/usr/bin/env python

import pandas as pd
import json
with open('../data/decahose/twitterHashtag.json', 'r') as datafile:
    data = json.loads(datafile)
# df = pd.read_json (r'../data/decahose/twitterHashtag.json', orient='records')

