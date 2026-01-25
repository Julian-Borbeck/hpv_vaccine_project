import pandas as pd
import numpy as np

# Here we load the data regarding the GLOBOCAN2022 cancer incidence and WHO HPV vaccine coverage
# Then perform a simple cleaning for the vaccine data, where we select about 10 years (+-2) before our cancer data, to account for HPV related cancer development
# We also select for the vaccination group of interest, in this case "HPV Vaccination program coverage, first dose, females"
# Now the mean vaccine coverage is computed for the 2010-2014 period

# Define WHO countries by region
who_countries = {
    "AFR": ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde", "Cabo Verde", "Central African Republic", "Chad", "Comoros", "Ivory Coast", "Democratic Republic of the Congo", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mozambique", "Namibia", "Niger", "Nigeria", "Republic of the Congo", "Rwanda", "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone", "South Africa", "South Sudan", "Eswatini", "Togo", "Uganda", "Tanzania", "Zambia", "Zimbabwe"],
    "AMR": ["Peru", "Paraguay", "Saint Kitts and Nevis", "Antigua and Barbuda", "Argentina", "Bahamas", "Barbados", "Belize", "Bolivia", "Brazil", "Canada", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "Ecuador", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago", "the United States of America", "Uruguay", "Venezuela"],
    "SEAR": ["Bangladesh", "Bhutan", "Democratic People's Republic of Korea", "India", "Maldives", "Myanmar", "Nepal", "Sri Lanka", "Thailand", "Timor-Leste"],
    "EUR": ["Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Kazakhstan", "Kyrgyzstan", "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Tajikistan", "Turkey", "Turkmenistan", "Ukraine", "United Kingdom", "Uzbekistan"],
    "EMR": ["Libya", "Afghanistan", "Bahrain", "Djibouti", "Egypt", "Iran", "Iraq", "Jordan", "Kuwait", "Israel", "Oman", "Pakistan", "Qatar", "Saudi Arabia", "Somalia", "Sudan", "Syria", "Tunisia", "United Arab Emirates", "Yemen", "Morocco"],
    "WPR": ["Australia", "Brunei", "Cambodia", "China", "Cook Islands", "Fiji", "Indonesia", "Japan", "Kiribati", "Laos", "Malaysia", "Marshall Islands", "Micronesia", "Mongolia", "Nauru", "New Zealand", "Niue", "Palau", "Papua New Guinea", "Philippines", "Samoa", "Singapore", "Solomon Islands", "South Korea", "Taiwan", "Tonga", "Tuvalu", "Vanuatu", "Vietnam"]
}

# Some countries have small variations in name, so a second dictionary accounts for that
name_recode = {
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Lao People's Democratic Republic": "Laos",
    "Republic of Moldova": "Moldova",
    "Russian Federation": "Russia",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Türkiye": "Turkey",
    "United States of America": "the United States of America",
    "Micronesia (Federated States of)": "Micronesia",
    "Netherlands (Kingdom of the)": "Netherlands",
    "Republic of Korea": "South Korea",
    "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania"
}

# Read the vaccination data
vax = pd.read_csv("C:\\Users\\ettod\\Tubingen_exercises\\DataL\\cacertovacc\\who_vax_country.tsv", sep="\t", header=0)

# Subset for specific antigen description
vax = vax[vax['ANTIGEN_DESCRIPTION'] == "HPV Vaccination program coverage, first dose, females"]

# Convert COVERAGE to integer
vax['COVERAGE'] = vax['COVERAGE'].astype('Int64')

# Subset for years 2010-2014
# USE THIS TO SELECT FOR SPECIFIC YEARS
#vax = vax[vax['YEAR'].isin(range(2010, 2014))]

# Group by country and calculate mean coverage
# USE THIS TO CALCULATE THE MEAN COVERAGE PER COUNTRY OF THE SELECTED YEARS
# IF NO YEARS HAVE BEEN SELECTED IT WILL CALCULATE THE MEAN FOR ALL THE YEARS
#vaccine_summary = vax.groupby('NAME').agg(mean_coverage_2010_2014=('COVERAGE', 'mean')).reset_index()

vax_1 = vax[vax["COVERAGE"] != 0]
vax_0 = vax[vax["COVERAGE"] == 0]
