import pandas as pd

df = pd.read_csv("data/organizations.csv")

def get_hospitals_by_city(city):
    city_hospitals = df[df['CITY'].str.lower() == city.lower()]['NAME'].tolist()
    return city_hospitals if city_hospitals else ["No hospitals found"]