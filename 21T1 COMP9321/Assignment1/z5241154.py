import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    # read csv files
    file_exposure = pd.read_csv(exposure, sep=';', encoding='ISO-8859-1', low_memory=False)
    file_countries = pd.read_csv(countries, encoding='ISO-8859-1')
    # create dataframe of exposure and countries
    df_exposure = pd.DataFrame(file_exposure)
    df_countries = pd.DataFrame(file_countries)

    # drop all rows without country name in exposure
    df_exposure.dropna(axis=0, subset=['country'], inplace=True)

    # change 'country' to 'Country' in df_exposure
    df_exposure.rename(columns={'country': 'Country'}, inplace=True)

    # edit country names which represent the same country in exposure.csv and Countries.csv
    country2exp = {'Brunei': 'Brunei Darussalam', 'Cape Verde': 'Cabo Verde', 'Palestinian Territory': 'Palestine',
                   'Democratic Republic of the Congo': 'Congo DR', 'Ivory Coast': "Côte d'Ivoire", 'Laos': 'Lao PDR',
                   'Macedonia': 'North Macedonia', 'Moldova': 'Moldova Republic of', 'North Korea': 'Korea DPR',
                   'Republic of the Congo': 'Congo', 'Russia': 'Russian Federation', 'Swaziland': 'Eswatini',
                   'South Korea': 'Korea Republic of', 'United States': 'United States of America',
                   'Vietnam': 'Viet Nam'}

    df_countries['Country'] = df_countries['Country'].map(lambda x: country2exp[x] if x in country2exp else x)

    # merge df_exposure and df_countries
    df1 = pd.merge(df_exposure, df_countries, how='inner', on='Country')
    df1.set_index('Country', inplace=True)
    df1.sort_index(ascending=True, inplace=True)

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


# calculate the mean value of location
def get_location(info):
    latitudes = []  # latitudes contain the latitude of all cities
    longitudes = []  # longitudes contain the longitude of all cities
    for item in info:
        temp_dic = json.loads(item)
        latitudes.append(temp_dic['Latitude'])
        longitudes.append(temp_dic['Longitude'])
    return np.mean(latitudes), np.mean(longitudes)


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df2 = df1.copy()
    df2['temp'] = df2['Cities'].map(lambda x: x.split('|||'))
    df2['avg_latitude'], df2['avg_longitude'] = zip(*df2['temp'].apply(get_location))
    df2.drop(['temp'], axis=1, inplace=True)

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


# calculate the distance to Wuhan
def cal_dis(lat, lng):
    R = 6373
    # turn angle to radian
    rad_lat = lat * math.pi / 180.0
    rad_lng = lng * math.pi / 180.0

    Wuhan_rad_lat = 30.5928 * math.pi / 180.0
    Wuhan_rad_lng = 114.3055 * math.pi / 180.0
    # get difference wit Wuhan
    diff_lat = rad_lat - Wuhan_rad_lat
    diff_lng = rad_lng - Wuhan_rad_lng

    # calculate the distance
    s = math.sin(diff_lat / 2) ** 2 + (math.sin(diff_lng / 2) ** 2) * math.cos(Wuhan_rad_lat) * math.cos(rad_lat)
    dis = 2 * math.atan2(s ** 0.5, (1 - s) ** 0.5) * R
    return dis


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    df3 = df2.copy()
    df3['distance_to_Wuhan'] = df3.apply(lambda row: cal_dis(row['avg_latitude'], row['avg_longitude']), axis=1)

    # sorted by distance_to_Wuhan
    df3.sort_values('distance_to_Wuhan', inplace=True)

    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


# get float num and change 'x' and 'No data' to None
def get_num(num):
    if num != 'x' and num != 'No data':
        return float(num.replace(',', '.'))
    else:
        return None


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    file_continents = pd.read_csv(continents, encoding='ISO-8859-1')
    df_continents = pd.DataFrame(file_continents)

    # same country in exposure.csv and Countries-Continents.csv
    continent2exp = {'Brunei': 'Brunei Darussalam', 'Cape Verde': 'Cabo Verde', 'Ivory Coast': "Côte d'Ivoire",
                     'Congo, Democratic Republic of': 'Congo DR', 'Laos': 'Lao PDR', 'Korea, North': 'Korea DPR',
                     'Macedonia': 'North Macedonia', 'Moldova': 'Moldova Republic of', 'Swaziland': 'Eswatini',
                     'Korea, South': 'Korea Republic of', 'US': 'United States of America', 'Vietnam': 'Viet Nam',
                     'Burkina': 'Burkina Faso', 'CZ': 'Czech Republic', 'Burma (Myanmar)': 'Myanmar'}
    df_continents['Country'] = df_continents['Country'].map(lambda x: continent2exp[x] if x in continent2exp else x)

    # merge df2 and df_continents
    df4 = pd.merge(df2, df_continents, how='inner', on='Country')

    # find 'x' and 'no data' value and delete these lines
    df4['Covid_19_Economic_exposure_index'] = df4['Covid_19_Economic_exposure_index'].apply(get_num)
    df4.dropna(axis=0, subset=['Covid_19_Economic_exposure_index'], inplace=True)

    df4 = df4.groupby('Continent')['Covid_19_Economic_exposure_index'].mean().to_frame()
    df4.reset_index(inplace=True)

    df4.set_index('Continent', inplace=True)
    df4.rename(columns={'Covid_19_Economic_exposure_index': 'average_covid_19_Economic_exposure_index'}, inplace=True)
    df4.sort_values('average_covid_19_Economic_exposure_index', ascending=True, inplace=True)

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    df2.rename(columns={'Income classification according to WB': 'Income Class'}, inplace=True)

    # find 'x' and 'no data' values
    df2['Foreign direct investment'] = df2['Foreign direct investment'].apply(get_num)
    df2['Net_ODA_received_perc_of_GNI'] = df2['Net_ODA_received_perc_of_GNI'].apply(get_num)

    # groupby 'Income classification'
    temp_foreign = df2.groupby('Income Class')['Foreign direct investment'].mean()
    temp_net = df2.groupby('Income Class')['Net_ODA_received_perc_of_GNI'].mean()

    temp_foreign = temp_foreign.to_frame()
    temp_net = temp_net.to_frame()

    df5 = pd.merge(temp_foreign, temp_net, how='inner', on='Income Class')

    df5.rename(columns={'Foreign direct investment': 'Avg Foreign direct investment',
                        'Net_ODA_received_perc_of_GNI': 'Avg_Net_ODA_received_perc_of_GNI'}, inplace=True)

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


# get get_population of each country
def get_population(info):
    population = []
    for item in info:
        temp_dic = json.loads(item)
        if temp_dic['Population'] is not None:
            population.append((temp_dic['City'], temp_dic['Population']))
    return population


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []

    # get Low income countries
    df6 = df2[df2['Income classification according to WB'] == 'LIC'].copy()

    # process the column 'Cities'
    df6['Cities'] = df6['Cities'].map(lambda x: x.split('|||'))
    df6['Population_in_cities'] = df6['Cities'].apply(get_population)

    # make a list with (city name: population)
    temp_lst = df6['Population_in_cities'].values.tolist()
    all_cities = []
    for ct in temp_lst:
        for ct_pop in ct:
            all_cities.append((ct_pop[0], ct_pop[1]))

    # sorted population
    temp_lst = sorted(all_cities, key=lambda x: (x[1], x[0]), reverse=True)[0:5]
    for ct_pop in temp_lst:
        cities_lst.append(ct_pop[0])

    log("QUESTION 6", output_df=None, other=cities_lst)
    return cities_lst


# get a dict with city in country: {city: country}
def ct_in_country(info):
    ct_ctr = {}
    for item in info:
        temp_dic = json.loads(item)
        ct_ctr[temp_dic['City']] = temp_dic['Country']
    return ct_ctr


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    # get a column with dicts contain {city: country}
    df2['Cities'] = df2['Cities'].apply(lambda x: x.split('|||'))
    df2['ct_in_country'] = df2['Cities'].apply(ct_in_country)

    # traverse all dicts to find city with same name and save the country name
    temp_list = df2['ct_in_country'].values.tolist()
    df7_dic = {}
    for ct_ctr in temp_list:
        for ct in ct_ctr:
            if ct not in df7_dic:
                df7_dic[ct] = [ct_ctr[ct]]
            else:
                df7_dic[ct].append(ct_ctr[ct])

    # find city names appear more than 1 time
    for ct in list(df7_dic.keys()):
        if len(df7_dic[ct]) < 2:
            del df7_dic[ct]
        else:
            df7_dic[ct] = sorted(df7_dic[ct])

    df7 = pd.DataFrame(pd.Series(df7_dic), columns=['countries'])
    df7.reset_index(inplace=True)
    df7.rename(columns={'index': 'city'}, inplace=True)
    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7


# calculate the total population of each country
def cal_total_population(info):
    population = []
    for item in info:
        temp_dic = json.loads(item)
        if temp_dic['Population'] is not None:
            population.append(temp_dic['Population'])
    return sum(population)


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """
    file_continents = pd.read_csv(continents, encoding='ISO-8859-1')
    df_continents = pd.DataFrame(file_continents)

    # calculate the total population around the world
    df2['Cities'] = df2['Cities'].map(lambda x: x.split('|||'))
    df2['Population'] = df2['Cities'].apply(cal_total_population)
    temp_lst = df2['Population'].values.tolist()
    PopTotal = sum(temp_lst)

    # same country in exposure.csv and Countries-Continents.csv
    continent2exp = {'Brunei': 'Brunei Darussalam', 'Cape Verde': 'Cabo Verde', 'Ivory Coast': "Côte d'Ivoire",
                     'Congo, Democratic Republic of': 'Congo DR', 'Laos': 'Lao PDR', 'Korea, North': 'Korea DPR',
                     'Macedonia': 'North Macedonia', 'Moldova': 'Moldova Republic of', 'Swaziland': 'Eswatini',
                     'Korea, South': 'Korea Republic of', 'US': 'United States of America', 'Vietnam': 'Viet Nam',
                     'Burkina': 'Burkina Faso', 'CZ': 'Czech Republic', 'Burma (Myanmar)': 'Myanmar'}

    df_temp = df_continents[df_continents['Continent'] == 'South America'].copy()
    df_temp['Country'] = df_temp['Country'].map(lambda ctr: continent2exp[ctr] if ctr in continent2exp else ctr)

    # calculate percentage of population in each South American country
    df8 = pd.merge(df2, df_temp, how='inner', on='Country')
    df8 = df8[['Country', 'Population']]
    df8['Percentage(%)'] = df8['Population'].apply(lambda x: (x / PopTotal) * 100)

    # draw the plot
    plt.figure(figsize=(10, 10))
    plt.title('The percentage of the world populations is living in each South American country')
    plt.xlabel('Countries', fontsize=11)
    plt.ylabel('Percentage(%)', fontsize=11)

    y = df8['Percentage(%)'].values.tolist()
    x = df8['Country'].values.tolist()

    plt.xticks(range(df8.shape[0]), x, fontsize=11, rotation=45)
    plt.bar(x, y, width=0.5)

    # add values
    for a, b in zip(x, y):
        plt.text(a, b, '%.2f' % b, horizontalalignment='center', verticalalignment='baseline', fontsize=11)

    plt.tight_layout()
    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    # find 'x' and 'no data' values
    df2['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'] = \
        df2['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'].apply(get_num)
    df2['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'] = \
        df2['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'].apply(get_num)
    df2['Foreign direct investment, net inflows percent of GDP'] = \
        df2['Foreign direct investment, net inflows percent of GDP'].apply(get_num)
    df2['Foreign direct investment'] = df2['Foreign direct investment'].apply(get_num)

    # groupby 'Income classification according to WB'
    aid_FDI = df2.groupby('Income classification according to WB')[
        'Covid_19_Economic_exposure_index_Ex_aid_and_FDI'].mean()
    aid_FDI_food_import = df2.groupby('Income classification according to WB')[
        'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'].mean()
    FDI_GDP = df2.groupby('Income classification according to WB')[
        'Foreign direct investment, net inflows percent of GDP'].mean()
    FDI = df2.groupby('Income classification according to WB')['Foreign direct investment'].mean()

    aid_FDI = aid_FDI.to_frame()
    aid_FDI_food_import = aid_FDI_food_import.to_frame()
    FDI_GDP = FDI_GDP.to_frame()
    FDI = FDI.to_frame()

    df_Cov19 = pd.merge(aid_FDI, aid_FDI_food_import, how='inner', on='Income classification according to WB')
    df_investment = pd.merge(FDI_GDP, FDI, how='inner',
                             on='Income classification according to WB')
    df9 = pd.merge(df_Cov19, df_investment, how='inner', on='Income classification according to WB')

    # draw the plot
    plt.figure(figsize=(10, 10))
    plt.title('The mean values of the high, middle and low income level countries')
    plt.ylabel('Mean value', fontsize=11)

    x_labels = ['aid_FDI', 'aid_FDI_food_import', 'FDI_GDP', 'FDI']
    plt.xticks(range(len(x_labels)), x_labels, fontsize=11)

    x = list(range(len(x_labels)))
    y_HIC = df9.loc['HIC'].values.tolist()
    plt.bar(x, y_HIC, width=0.2, label="HIC", fc="lightcoral")
    # add values
    for a, b in zip(x, y_HIC):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=11)

    for i in range(len(x)):
        x[i] = x[i] + 0.2
    y_MIC = df9.loc['MIC'].values.tolist()
    plt.bar(x, y_MIC, width=0.2, label="MIC", fc="c")
    for a, b in zip(x, y_MIC):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=11)

    for i in range(len(x)):
        x[i] = x[i] + 0.2
    y_LIC = df9.loc['LIC'].values.tolist()
    plt.bar(x, y_LIC, width=0.2, label="LIC", fc="orange")
    for a, b in zip(x, y_LIC):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.legend()

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    file_continents = pd.read_csv(continents, encoding='ISO-8859-1')
    df_continents = pd.DataFrame(file_continents)

    # same country in exposure.csv and Countries-Continents.csv
    continent2exp = {'Brunei': 'Brunei Darussalam', 'Cape Verde': 'Cabo Verde', 'Ivory Coast': "Côte d'Ivoire",
                     'Congo, Democratic Republic of': 'Congo DR', 'Laos': 'Lao PDR', 'Korea, North': 'Korea DPR',
                     'Macedonia': 'North Macedonia', 'Moldova': 'Moldova Republic of', 'Swaziland': 'Eswatini',
                     'Korea, South': 'Korea Republic of', 'US': 'United States of America', 'Vietnam': 'Viet Nam',
                     'Burkina': 'Burkina Faso', 'CZ': 'Czech Republic', 'Burma (Myanmar)': 'Myanmar'}
    df_continents['Country'] = df_continents['Country'].map(
        lambda ctr: continent2exp[ctr] if ctr in continent2exp else ctr)
    df10 = pd.merge(df2, df_continents, how='inner', on='Country')

    df10['Cities'] = df10['Cities'].map(lambda info: info.split('|||'))
    df10['Population'] = df10['Cities'].apply(cal_total_population)
    df10['Population'] = df10['Population'].apply(lambda p: p / 300000)

    color_cont = {'Asia': 'purple', 'Europe': 'yellow', 'Africa': 'orange', 'North America': 'green',
                  'South America': 'brown', 'Oceania': 'red'}
    df10['Color_cont'] = df10['Continent'].map(color_cont)
    df10 = df10[['Country', 'Continent', 'Color_cont', 'Population', 'avg_latitude', 'avg_longitude']]

    # draw the plot
    plt.figure(figsize=(10, 10))
    plt.xlabel('avg_longitude', fontsize=11)
    plt.ylabel('avg_latitude', fontsize=11)

    for con in list(color_cont.keys()):
        temp_df = df10[df10['Continent'] == con]
        x = temp_df['avg_longitude'].values.tolist()
        y = temp_df['avg_latitude'].values.tolist()
        color = color_cont[con]
        siz = temp_df['Population'].values.tolist()
        plt.scatter(x, y, label=con, c=color, s=siz)

    plt.tight_layout()
    leg = plt.legend(loc='best', fontsize=13)
    for i in range(len(color_cont)):
        leg.legendHandles[i]._sizes = [150]

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    # df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    # df5 = question_5(df2.copy(True))
    # lst = question_6(df2.copy(True))
    # df7 = question_7(df2.copy(True))
    # question_8(df2.copy(True), "Countries-Continents.csv")
    # question_9(df2.copy(True))
    # question_10(df2.copy(True), "Countries-Continents.csv")
