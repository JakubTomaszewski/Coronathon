import pandas as pd
import matplotlib.pyplot as plt
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import schedule
from PIL import Image, ImageDraw, ImageFont

with open('token.txt', 'r') as file:
    token = file.read()

client = EntsoePandasClient(api_key=token)

def get_data(country_code, start, end):
    return client.query_load(country_code, start=start, end=end)


def plot_map(df):
    img = plt.imread('wrldmap.png')
    fig, ax = plt.subplots(figsize=(17, 7))
    ax.imshow(img, extent=[-180, 180, -90, 90])
    ax.scatter(x=df['Long'] - 16, y=df['Lat'] - 30, alpha=0.3, s=df['Cases'] / 30, c=df['Cases'],
               cmap=plt.get_cmap('jet'))
    fig.suptitle('COVID-19 Cases around the world', fontsize=15)
    # plt.show()
    fig.savefig('Mapa.png')


def cleandata(df_raw):
    df_cleaned = df_raw.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_name='Cases',
                             var_name='Date')
    df_cleaned = df_cleaned.drop('Province/State', axis=1)

    plot_map(df_cleaned)
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned = df_cleaned[['Country/Region', 'Cases', 'Date']]
    df_cleaned = df_cleaned.pivot_table(index='Date', columns='Country/Region', values='Cases', aggfunc='sum')
    return df_cleaned


def show_double_bar(monthly_2019, monthly_2020):
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(13, 5))

    x_2019 = np.arange(len(monthly_2019))
    x_2020 = np.arange(len(monthly_2020))
    bar_width = 0.4
    opacity = 0.5

    plt.bar(x_2019, monthly_2019, width=bar_width, color='orange', alpha=opacity, label='2019')
    plt.bar(x_2020 + bar_width, monthly_2020, width=bar_width, color='green', alpha=opacity, label='2020')

    plt.title('Monthly 2019 vs 2020 energy consumption')
    plt.xticks(x_2019 + bar_width / 2, monthly_2019.index.strftime('%d-%m'), rotation=90)
    plt.legend()
    plt.show()


def show_energy_disparity(ctry_2019, ctry_2020, country):
    '''Create a plot of energy consumption disparity for a country 2019 vs 2020'''
    # loc the values to day of the week
    global country_2019
    global country_2020
    country_2019 = ctry_2019.loc['2019-01-23': '2019'+(datetime.now()).strftime('-%m-%d')]
    country_2020 = ctry_2020.loc['2020-01-22':]

    idx = country_2019.index.strftime('%m-%d')
    y_vs_y = pd.DataFrame({'2019': country_2019.values, '2020': country_2020.values}, index=idx)
    y_vs_y.plot(figsize=(13,5))
    plt.title(f'Energy consumption in {country}')
    plt.ylabel('Total Energy Consumption [MWh]')
    plt.xlabel('Date')
    plt.show()


def plot_energy_vs_covid(ctry_2019, ctry_2020, country, covid):
    # loc the values to day of the week
    # global country_2019
    # global country_2020

    ctry_2019 = ctry_2019.loc['2019-01-23': '2019' + (datetime.now()).strftime('-%m-%d')]
    ctry_2020 = ctry_2020.loc['2020-01-22':]
    covid = covid.loc['2020-01-23':]

    idx = ctry_2019.index.strftime('%m-%d')
    y_vs_y = pd.DataFrame({'2019': ctry_2019.values, '2020': ctry_2020.values}, index=idx)
    ax1 = y_vs_y.plot(figsize=(13, 5))
    plt.title(f'Energy consumption in {country}')
    plt.ylabel('Total Energy Consumption [MWh]')
    plt.xlabel('Date')

    covid = covid.set_index(idx)
    ax2 = covid[country].plot(secondary_y=True, sharex=True)
    ax2.set_ylabel('Cases', color='g')
    ax1.legend(loc='upper center')
    plt.savefig(f'fig_{country}.png', dpi=100)
    # plt.show()


def pie_plot(df, total):
    countries = ['Germany', 'Italy', 'Poland', 'Spain', 'Belgium', 'US', 'France', 'Czechia']
    count_countries = df.loc[((datetime.now()) - timedelta(1)).strftime('%Y-%m-%d'), countries]
    count_countries['Other'] = total - sum(count_countries)

    plt.figure()
    count_countries.plot(kind='pie', label='', autopct='%1.1f%%', figsize=(8, 8),
                         explode=(0, 0, 0, 0, 0, 0.1, 0, 0, 0.1))
    plt.title('Percentage of cases', size=18)
    plt.savefig('Pie.png')
    # plt.show()


def display_cases(total):
    img = Image.new('RGB', (500, 1000), color=(255, 255, 255))

    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
    d = ImageDraw.Draw(img)
    d.text((300, 500), f"Coronavirus Cases:\n{total}", font=fnt, fill=(0, 0, 0))
    img.save('Total_cases.png')


def get_ratio(df, pop):
    # Get ratio for each country
    cases_now = df.loc[((datetime.now()) - timedelta(1)).strftime('%Y-%m-%d'), :]
    pop = pop[['Country Name', '2018']].dropna()

    cases_now = pd.DataFrame(cases_now).reset_index()
    cases_now.columns = ['Country Name', 'Cases']

    pop_cases = pd.merge(pop, cases_now, on='Country Name')
    pop_cases['Ratio'] = pop_cases['Cases'] / pop_cases['2018']
    return pop_cases.set_index('Country Name')


url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

df_raw = pd.read_csv(url)
df = cleandata(df_raw)

total = df.loc[((datetime.now()) - timedelta(1)).strftime('%Y-%m-%d'), :].sum()
pie_plot(df, total)

'''
countries = {'IT': 'Italy', 'BE': 'Belgium', 'PL': 'Poland', 'FR': 'France', 'CZ': 'Czechia', 'ES':'Spain'}

for code, country in countries.items():
    start = pd.Timestamp('20190101', tz='Europe/Rome')
    end = pd.Timestamp('20200101', tz='Europe/Rome')
    country_2019 = get_data(country_code=code, start=start, end=end).resample('D').sum()
    # country_2019.index = country_2019.index.strftime('%Y-%m-%d')

    today = datetime.now() - timedelta(2)
    start = pd.Timestamp('20200101', tz='Europe/Rome')
    end = pd.Timestamp(today, tz='Europe/Rome')
    country_2020 = get_data(country_code=code, start=start, end=end).resample('D').sum()
    # country_2020.index = country_2020.index.strftime('%Y-%m-%d')


    plot_energy_vs_covid(country_2019, country_2020, countries[code], df)

pop = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_887275.csv')
pop_cases = get_ratio(df, pop)
pop_cases.to_excel('ratio.xlsx')
print(pop_cases)
'''




