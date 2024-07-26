# -*- coding: utf-8 -*-
# @Time     : 7/18/2024 22:14
# @Author   : Junyi
# @FileName: Detect_language.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from langdetect import detect, DetectorFactory
import pandas as pd

# Load and preprocess data
text_file = r"E:/Python_Workplace/OptimalDistinct/Cluster/data/sbert_similarity_results_allcat.csv"
# text_file_list = ["Detail_99_CleanData_0507.csv", "Detail_99_CleanData_0514.csv", "Detail_99_CleanData_0521.csv",
#                   "Detail_99_CleanData_0528.csv", "Detail_99_CleanData_0604.csv", "Detail_99_CleanData_0611.csv",
#                   "Detail_99_CleanData_0618.csv", "Detail_99_CleanData_0625.csv"]
# text_file_list = [r"./data/" + text for text in text_file_list]
# for text_file in text_file_list:
df = pd.read_csv(text_file)

# Function to detect primary language of a sentence
def detect_primary_language(sentence):
    try:
        # Initialize DetectorFactory to increase reliability
        DetectorFactory.seed = 0
        # Skip empty or very short texts
        if len(sentence.strip()) < 3:  # Adjust the threshold as needed
            return "None"
        # Detect language of the sentence
        lang = detect(sentence)
        return lang
    except Exception as e:
        print(f"Error detecting language: {sentence}")
        return "None"

# Create a new column to store primary language
df['Primary_Language'] = None

# Iterate over each row and apply the language detection function
for index, row in df.iterrows():
    if pd.notna(row['Description']):
        df.at[index, 'Primary_Language'] = detect_primary_language(row['Description'])

# Save the DataFrame with primary language identified as a new column in the original CSV file
output_file = "{}_with_language.csv".format(text_file[:-4])
df.to_csv(output_file, index=False)

# Display the DataFrame with primary language identified for each sentence
# print(df[['Description', 'Primary_Language']])
# print(f"Output saved to {output_file}")
