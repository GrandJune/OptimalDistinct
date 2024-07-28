# -*- coding: utf-8 -*-
# @Time     : 7/28/2024 16:20
# @Author   : Junyi
# @FileName: Remove_words.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from langdetect import detect, DetectorFactory
import pandas as pd
import re

def remove_words_case_insensitive(text, words_to_remove):
    # Convert text to lowercase
    text_lower = text.lower()

    # Create a regex pattern to match all words in the list, also convert the list to lowercase
    pattern = '|'.join([re.escape(word.lower()) for word in words_to_remove])

    # Use regex sub to replace the words with an empty string
    cleaned_text = re.sub(pattern, '', text_lower)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Load and preprocess data
text_file = r"E:/Python_Workplace/OptimalDistinct/Cluster/data/sbert_similarity_results_allcat_with_language.csv"
# text_file_list = ["Detail_99_CleanData_0507.csv", "Detail_99_CleanData_0514.csv", "Detail_99_CleanData_0521.csv",
#                   "Detail_99_CleanData_0528.csv", "Detail_99_CleanData_0604.csv", "Detail_99_CleanData_0611.csv",
#                   "Detail_99_CleanData_0618.csv", "Detail_99_CleanData_0625.csv"]
# text_file_list = [r"./data/" + text for text in text_file_list]
# for text_file in text_file_list:
df = pd.read_csv(text_file)

words_to_remove = ["Conversion Starters", "AI", "GPT", "Prompt", "Starters", "ChatGPT"]

# text_list = df["Description"].astype(str).tolist()
# cleaned_text = [remove_words_case_insensitive(text, words_to_remove) for text in text_list]
# df["Description"] = cleaned_text
#
# text_list = df["Features"].astype(str).tolist()
# cleaned_text = [remove_words_case_insensitive(text, words_to_remove) for text in text_list]
# df["Features"] = cleaned_text

text_list = df["Conversion_start"].astype(str).tolist()
cleaned_text = [remove_words_case_insensitive(text, words_to_remove) for text in text_list]
df["Conversion_start"] = cleaned_text

for each in df["Conversion_start"]:
    print(each)
# Save data
# df.to_csv(text_file, index=False)
