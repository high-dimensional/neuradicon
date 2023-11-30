#!/usr/bin/env python
# coding: utf-8

# # Analysis of report dataset

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


# In[2]:


PROCESSED_REPORTS_PATH = (
    "DATAPATH/Desktop/neuroNLP_data/cleaned_report_data/filtered_reports.csv"
)


# In[3]:


report_df = pd.read_csv(
    PROCESSED_REPORTS_PATH,
    parse_dates=[
        "study_date",
        "request_date",
        "dob",
        "dod",
        "arrived_date",
        "authorised_date",
    ],
    dtype={"patient_id": str},
)


# In[4]:


report_df.columns


# In[5]:


report_df.loc[:, "aquisition_time_days"] = (
    report_df.study_date - report_df.request_date
).dt.days
report_df.loc[:, "reporting_time_days"] = (
    report_df.authorised_date - report_df.study_date
).dt.days
report_df.loc[:, "waiting_time"] = (
    report_df.study_date - report_df.arrived_date
).dt.seconds / 3600
report_df.loc[:, "report_length_chars"] = report_df.report_text.str.len()
report_df.loc[:, "age_at_study"] = (
    report_df.arrived_date - report_df.dob
).dt.days / 365


# In[6]:


report_df.loc[report_df.study_date.isna(), "study_date"] = report_df.loc[
    report_df.study_date.isna(), "arrived_date"
]


# In[7]:


report_df = report_df[report_df.age_at_study >= 18]


# In[8]:


print("N Reports: {}".format(len(report_df)))
print("Period: {} - {}".format(report_df.study_date.min(), report_df.study_date.max()))
print("N unique patients: {}".format(len(report_df.patient_id.unique())))
print("Age at scan: {}".format(report_df.age_at_study.describe()))
print("Sex ratio: {}".format(report_df.sex.value_counts(normalize=True)))


# In[9]:


fig = sns.displot(
    report_df[~report_df.patient_id.duplicated()],
    x="age_at_study",
    aspect=1.5,
    height=8,
)
fig.ax.set_xlim(18, 102)
fig.savefig("age_dist.svg")


# In[10]:


fig = sns.displot(report_df, x="report_length_chars", aspect=1.5, height=8)
fig.ax.set_xlim(0, 4000)
fig.savefig("length_dist.svg")


# In[11]:


report_df.report_length_chars.describe()


# In[12]:


scans_per_patient = (
    report_df.patient_id.value_counts().to_frame("n_scans").reset_index()
)


# In[13]:


scans_per_patient.describe()


# In[14]:


fig = sns.displot(scans_per_patient, x="n_scans", bins=40, aspect=1.5, height=8)
fig.ax.set_xlim(1, 10)
fig.savefig("n_scan_dist.svg")


# In[ ]:


# In[15]:


sex_df = report_df[
    ~report_df.patient_id.duplicated() & report_df.sex.isin(["Male", "Female"])
]


# In[16]:


fig = sns.catplot(data=sex_df, x="sex", kind="count", aspect=1, height=8)
fig.savefig("sex_dist.svg")


# In[17]:


def lolliplot(data=None, x=None, kind="count", aspect=1, height=1):
    counts = data[x].value_counts().iloc[::-1]
    fig, ax = plt.subplots(figsize=(height, aspect * height))
    my_range = range(1, len(counts.index) + 1)
    ax.hlines(y=my_range, xmin=0, xmax=counts, color="skyblue")
    ax.plot(counts, my_range, "o")
    ax.set_yticks(my_range)
    ax.set_xlabel("count")
    ax.set_ylabel(x)
    ax.set_yticklabels(counts.index)
    return fig


# In[18]:


fgi = lolliplot(data=report_df, x="ethnicity", kind="count", aspect=1.5, height=8)
fgi.savefig("ethnicity_dist.svg", bbox_inches="tight")


# In[19]:


top_20_examination_type = report_df.examination_type.value_counts()[:20]
print(top_20_examination_type)


# In[20]:


fgi = lolliplot(
    data=report_df, x="examination_type", kind="count", aspect=1.5, height=8
)
fgi.savefig("exam_type_dist.svg", bbox_inches="tight")


# In[27]:


specialtes = report_df.specialty.str.replace("(B)", "", regex=False)
specialtes = specialtes.str.strip()
specialtes = specialtes.str.upper()
report_df.specialty = specialtes


# In[28]:


top_20_specialty = report_df.specialty.value_counts()[:30]
print(top_20_specialty)


# In[29]:


specialty_df = report_df[report_df.specialty.isin(top_20_specialty.index)]


# In[30]:


fgi = lolliplot(data=specialty_df, x="specialty", kind="count", aspect=1.5, height=8)
fgi.savefig("specialty_dist.svg", bbox_inches="tight")


# In[31]:


titles = ["DR ", "MR ", "MISS ", "MS ", "PROFESSOR ", "PROF "]
referrers = report_df.referring_doctor.str.upper()
for title in titles:
    referrers = referrers.str.replace(title, "", regex=False)
referrers = referrers.str.strip()
report_df.referring_doctor = referrers


# In[32]:


top_20_referrers = report_df.referring_doctor.value_counts()[:50]


# In[33]:


print(top_20_referrers)


# In[34]:


referrer_df = report_df[report_df.referring_doctor.isin(top_20_referrers.index)]


# In[35]:


fgi = lolliplot(
    data=referrer_df, x="referring_doctor", kind="count", aspect=1.5, height=8
)
fgi.savefig("referrer_dist.svg", bbox_inches="tight")


# In[36]:


fig = sns.displot(data=report_df, x="study_date", aspect=1.5, height=8)
# fig.ax.set_xlim(1, 10)
fig.savefig("reports_per_date.svg")


# In[ ]:
