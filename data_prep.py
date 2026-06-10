# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from astropy.io import fits
import pandas as pd
import os



with fits.open('/kaggle/input/datasets/maanav0114/harps-n-dataset/ADP.2023-12-04T15_16_53.464.fits') as data:
    harps_df = pd.DataFrame(data[1].data)

# Fix the big-endian to little-endian compiler issue
harps_df.to_csv('/kaggle/working/temp.csv', index = False)
del harps_df
harps_df = pd.DataFrame(pd.read_csv('/kaggle/working/temp.csv'))
os.remove('/kaggle/working/temp.csv')

harps_df.describe()

# %%
hires_df = pd.DataFrame(pd.read_csv('/kaggle/input/datasets/maanav0114/harps-n-dataset/preprocessed_HIRES_data.csv'))

hires_df.describe()

# %%
exo_catalog = pd.DataFrame(pd.read_csv('/kaggle/input/datasets/maanav0114/harps-n-dataset/catalog_of_exoplanets3.csv'))
exo_catalog.describe()


# %%
def get_star_names(idx):
    name_list = []

    name_strs = (f"{exo_catalog.iloc[idx]['star_name']}, {exo_catalog.iloc[idx]['star_alternate_names']}").split(",")
    for name in name_strs:
        name = name.strip()
        name_list.append(name)

    return name_list


# %%
# for hires_name in list(set(list(hires_df['main_id_simbad']))):
#     for harps_name in list(set(list(harps_df['main_id_simbad']))):
#         if hires_name in harps_name or harps_name in hires_name:
#             print(hires_name)
#             print(harps_name)
#             break

# %%
harps_stars = []
for name in list(set(list(harps_df['main_id_simbad']))):
    name = name.split(" ")
    name = "".join(name)
    name = name.lower()

    harps_stars.append(name)

harps_stars[:10]

# %%
observations = harps_df
cols_to_keep = ['main_id_simbad', 'drs_bjd', 'drs_ccf_rv', 'drs_dvrms']
observations = observations[[col for col in cols_to_keep if col in observations.columns]]

cleaned_names = hires_df['main_id_simbad'].str.lower().str.replace(" ", "", regex=False)
harps_stars_set = set(harps_stars)
mask = ~cleaned_names.isin(harps_stars_set)
observations = pd.concat([observations, hires_df[mask]], ignore_index=True)
# observations = observations.dropna(subset=['drs_ccf_rvc'])

observations

# %%
# # Define the column names representing the stars in each dataframe
# harps_star_col = 'star'  # Update to actual column name, e.g., 'OBJECT', 'star', 'name'
# hires_star_col = 'star'  # Update to actual column name

# # Filter out rows from HIRES dataframe where the star is already in the HARPS dataframe
# if harps_star_col in harps_df.columns and hires_star_col in hires_df.columns:
#     harps_stars = harps_df[harps_star_col].unique()
#     hires_filtered = hires_df[~hires_df[hires_star_col].isin(harps_stars)]
# else:
#     # Fallback if the column names are incorrect
#     print(f"Warning: Ensure '{harps_star_col}' and '{hires_star_col}' are the correct column names.")
#     hires_filtered = hires_df

# # Create the new combined dataframe
# combined_df = pd.concat([harps_df, hires_filtered], ignore_index=True)
# combined_df.head()
