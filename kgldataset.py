import pandas as pd
import json
# df = pd.read_csv('kaggledatasetrawfiles/linkdin_Job_data.csv',usecols=['job','job_details','company_name','work_type'])
# df['job_details'] = df['job_details'].str.replace('.','.\n',regex=False)
# df = df.rename(columns={'job_details':'text'})
# df.to_json('data/raw_posts.json',orient='records',force_ascii=False,indent=4)
#
# print(f"Wrote {len(df)} records")

df = pd.read_csv('kaggledatasetrawfiles/linkdin_Job_data.csv', usecols=['job', 'job_details'])

examples = []
for i, row in df.iterrows():
    instruction = f"Write a LinkedIn post about the job: {row['job']}"
    output = row['job_details']
    examples.append({"instruction": instruction, "output": output})

with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=4, ensure_ascii=False)

print(f"Wrote {len(examples)} records to train.json")