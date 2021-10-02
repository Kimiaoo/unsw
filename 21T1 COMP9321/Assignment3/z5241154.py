import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score

# get dataset path from command line
training_path = sys.argv[1]
validation_path = sys.argv[2]

# read training file and validation file by pandas
training_file = pd.read_csv(training_path)
validation_file = pd.read_csv(validation_path)


# deal with json
# e.g. cast, genres
def get_info(info):
    par = eval(info)
    res = []
    for p in par:
        res.append(p["name"])
    return res


# get top stars/crew
def get_top(par, top):
    all_cast = {}
    for m in par:
        for c in m:
            if c not in all_cast:
                all_cast[c] = 1
            else:
                all_cast[c] += 1
    return sorted(all_cast.items(), key=lambda item: item[1], reverse=True)[:top]


# get name of top
def get_top_name(top_pa):
    ids = []
    for item in top_pa:
        ids.append(item[0])
    return ids


# check if top have participated in the Movie
def check_top(par, top):
    res = []
    for item in top:
        if item[0] in par:
            res.append(1)
        else:
            res.append(0)
    return res


# put feature into x_train
def add_feature(df, names, feat):
    for i in range(len(names)):
        temp_list = []
        for item in feat:
            temp_list.append(item[i])
        df[names[i]] = temp_list
    return df


# get who has participated in the Movie
# for crew
def get_crew(info):
    par = eval(info)
    res_director = []
    res_editor = []
    res_producer = []
    res_music_editor = []
    res_writer = []
    res_screenplay = []
    for p in par:
        if p["job"] == "Director":
            res_director.append(p["name"])
        elif p["job"] == "Editor":
            res_editor.append(p["name"])
        elif p["job"] == "Producer":
            res_producer.append(p["name"])
        elif p["job"] == "Music Editor":
            res_music_editor.append(p["name"])
        elif p["job"] == "Writer":
            res_writer.append(p["name"])
        elif p["job"] == "Screenplay":
            res_screenplay.append(p["name"])
    return res_director, res_editor, res_producer, res_music_editor, res_writer, res_screenplay


# get production_countries
def get_countries(info):
    info = eval(info)
    for ctr in info:
        if ctr["name"] == "United States of America":
            return 1
    return 0


# get release date and change it into quarter
def get_quarter(info):
    link_loc = info.find('-')
    mon = int(info[link_loc + 1:link_loc + 3])
    if 1 <= mon <= 3:
        return 1
    elif 4 <= mon <= 6:
        return 2
    elif 7 <= mon <= 9:
        return 3
    elif 10 <= mon <= 12:
        return 4


# # num of features
# top_cast_num = 10
# top_director_num = 13
# top_editor_num = 10
# top_producer_num = 10
# top_music_editor_num = 5
# top_genres_num = 5
# top_production_comp_num = 5

top_cast_num = 10
top_director_num = 22
top_editor_num = 10
top_producer_num = 50
top_music_editor_num = 20
top_writer_num = 30
top_genres_num = 14
top_production_comp_num = 50

# create a new train set to save clean data
x_train = pd.DataFrame()
x_test = pd.DataFrame()
# a temp dataframe to save useful information
temp_train = pd.DataFrame()
temp_test = pd.DataFrame()

# -------------------------------------------- edit the dataset --------------------------------------------

# ---------------------- cast ----------------------
# get who has participated in the Movie
temp_train["cast"] = training_file["cast"].apply(get_info)

# get the most popular stars
cast_list = temp_train["cast"].values.tolist()
top_cast = get_top(cast_list, top_cast_num)
top_cast_name = get_top_name(top_cast)

# find if these famous casts have attend the film
temp_train["cast"] = temp_train["cast"].apply(check_top, args=(top_cast,))

# put cast feature into x_train
cast_train = temp_train["cast"].values.tolist()
x_train = add_feature(x_train, top_cast_name, cast_train)

# find features in validation_file
temp_test["cast"] = validation_file["cast"].apply(get_info)
temp_test["cast"] = temp_test["cast"].apply(check_top, args=(top_cast,))
cast_test = temp_test["cast"].values.tolist()
x_test = add_feature(x_test, top_cast_name, cast_test)

# ---------------------- crew ----------------------
# get who has participated in the Movie
temp_train["director"], temp_train["editor"], temp_train["producer"], temp_train["music_editor"], \
temp_train["writer"], temp_train["screenplay"] = zip(*training_file["crew"].apply(get_crew))

# feature: director
director_list = temp_train["director"].values.tolist()
# get the most popular directors
top_director = get_top(director_list, top_director_num)
top_director_name = get_top_name(top_director)
# find if these famous directors have attend the film
temp_train["director"] = temp_train["director"].apply(check_top, args=(top_director,))
# put cast feature into x_train
director_train = temp_train["director"].values.tolist()
x_train = add_feature(x_train, top_director_name, director_train)

# feature: editor
editor_list = temp_train["editor"].values.tolist()
top_editor = get_top(editor_list, top_editor_num)
top_editor_name = get_top_name(top_editor)
temp_train["editor"] = temp_train["editor"].apply(check_top, args=(top_editor,))
editor_train = temp_train["editor"].values.tolist()
x_train = add_feature(x_train, top_editor_name, editor_train)

# feature: producer
producer_list = temp_train["producer"].values.tolist()
top_producer = get_top(producer_list, top_producer_num)
top_producer_name = get_top_name(top_producer)
temp_train["producer"] = temp_train["producer"].apply(check_top, args=(top_producer,))
producer_train = temp_train["producer"].values.tolist()
x_train = add_feature(x_train, top_producer_name, producer_train)

# feature: Music Editor
music_editor_list = temp_train["music_editor"].values.tolist()
top_music_editor = get_top(music_editor_list, top_music_editor_num)
top_music_editor_name = get_top_name(top_music_editor)
temp_train["music_editor"] = temp_train["music_editor"].apply(check_top, args=(top_music_editor,))
music_editor_train = temp_train["music_editor"].values.tolist()
x_train = add_feature(x_train, top_music_editor_name, music_editor_train)

# feature: Writer
writer_list = temp_train["writer"].values.tolist()
top_writer = get_top(writer_list, top_writer_num)
top_writer_name = get_top_name(top_writer)
temp_train["writer"] = temp_train["writer"].apply(check_top, args=(top_writer,))
writer_train = temp_train["writer"].values.tolist()
x_train = add_feature(x_train, top_writer_name, writer_train)

# find features in validation_file
temp_test["director"], temp_test["editor"], temp_test["producer"], temp_test["music_editor"], \
temp_test["writer"], temp_test["screenplay"] = zip(*validation_file["crew"].apply(get_crew))

temp_test["director"] = temp_test["director"].apply(check_top, args=(top_director,))
director_test = temp_test["director"].values.tolist()
x_test = add_feature(x_test, top_director_name, director_test)

temp_test["editor"] = temp_test["editor"].apply(check_top, args=(top_editor,))
editor_test = temp_test["editor"].values.tolist()
x_test = add_feature(x_test, top_editor_name, editor_test)

temp_test["producer"] = temp_test["producer"].apply(check_top, args=(top_producer,))
producer_test = temp_test["producer"].values.tolist()
x_test = add_feature(x_test, top_producer_name, producer_test)

temp_test["music_editor"] = temp_test["music_editor"].apply(check_top, args=(top_music_editor,))
music_editor_test = temp_test["music_editor"].values.tolist()
x_test = add_feature(x_test, top_music_editor_name, music_editor_test)

temp_test["writer"] = temp_test["writer"].apply(check_top, args=(top_writer,))
writer_test = temp_test["writer"].values.tolist()
x_test = add_feature(x_test, top_writer_name, writer_test)


# ---------------------- budget ----------------------
# get budget from training
x_train["budget"] = training_file["budget"]
x_test["budget"] = validation_file["budget"]

# ---------------------- genres ----------------------
# get genres from training
# almost same as cast
temp_train["genres"] = training_file["genres"].apply(get_info)
genres_list = temp_train["genres"].values.tolist()
top_genres = get_top(genres_list, top_genres_num)
top_genres_name = get_top_name(top_genres)
temp_train["genres"] = temp_train["genres"].apply(check_top, args=(top_genres,))
genres_train = temp_train["genres"].values.tolist()
x_train = add_feature(x_train, top_genres_name, genres_train)

# find features in validation_file
temp_test["genres"] = validation_file["genres"].apply(get_info)
temp_test["genres"] = temp_test["genres"].apply(check_top, args=(top_genres,))
genres_test = temp_test["genres"].values.tolist()
x_test = add_feature(x_test, top_genres_name, genres_test)

# ---------------------- homepage ----------------------
# check if the film has homepage from training.csv
x_train["homepage"] = training_file["homepage"].map(lambda x: 1 if isinstance(x, str) else 0)
x_test["homepage"] = validation_file["homepage"].map(lambda x: 1 if isinstance(x, str) else 0)

# ---------------------- original_language ----------------------
# check if original language of the film is english
x_train["original_language"] = training_file["original_language"].map(lambda x: 1 if x == "en" else 0)
x_test["original_language"] = validation_file["original_language"].map(lambda x: 1 if x == "en" else 0)

# ---------------------- production_companies ----------------------
temp_train["production_companies"] = training_file["production_companies"].apply(get_info)
production_comp_list = temp_train["production_companies"].values.tolist()
top_production_comp = get_top(production_comp_list, top_production_comp_num)
top_production_comp_name = get_top_name(top_production_comp)

temp_train["production_companies"] = temp_train["production_companies"].apply(check_top, args=(top_production_comp,))
production_companies_train = temp_train["production_companies"].values.tolist()
x_train = add_feature(x_train, top_production_comp_name, production_companies_train)

# find features in validation_file
temp_test["production_companies"] = validation_file["production_companies"].apply(get_info)
temp_test["production_companies"] = temp_test["production_companies"].apply(check_top, args=(top_production_comp,))
production_companies_test = temp_test["production_companies"].values.tolist()
x_test = add_feature(x_test, top_production_comp_name, production_companies_test)

# ---------------------- production_countries ----------------------
x_train["production_countries"] = training_file["production_countries"].apply(get_countries)
x_test["production_countries"] = validation_file["production_countries"].apply(get_countries)

# ---------------------- release_date ----------------------
x_train["release_date"] = training_file["release_date"].apply(get_quarter)
x_test["release_date"] = validation_file["release_date"].apply(get_quarter)

# ---------------------- runtime ----------------------
x_train["runtime"] = training_file["runtime"]
x_test["runtime"] = validation_file["runtime"]

# ------------------------------------------- create files and show results -------------------------------------------
x_train = np.array(x_train)
x_test = np.array(x_test)
movie_id = validation_file["movie_id"].values.tolist()

# Part-I: Regression
part1_y_train = np.array(training_file["revenue"])
part1_y_test = np.array(validation_file["revenue"])

part1_regression = RandomForestRegressor(n_estimators=90, max_depth=60, max_features=100, random_state=0)
part1_regression.fit(x_train, part1_y_train)

part1_y_pred = part1_regression.predict(x_test)
msr = mean_squared_error(part1_y_test, part1_y_pred)
correlation = np.corrcoef(part1_y_test, part1_y_pred)[0, 1]

print("MSR: ", msr)
print("correlation: ", correlation)

# # write to csv file
# part1_summary = pd.DataFrame({"zid": ["z5241154"], "MSR": [np.around(msr, 2)],
#                               "correlation": [np.around(correlation, 2)]})
# part1_summary.to_csv("z5241154.PART1.summary.csv", index=False)
# part1_output = pd.DataFrame({"movie_id": movie_id, "predicted_revenue": part1_y_pred.tolist()})
# part1_output.sort_values(by="movie_id", ascending=True, inplace=True)
# part1_output.to_csv("z5241154.PART1.output.csv", index=False)

# Part-II: Classification
part2_y_train = np.array(training_file["rating"])
part2_y_test = np.array(validation_file["rating"])

part2_classification = RandomForestClassifier(n_estimators=100, max_features=100, random_state=0)
part2_classification.fit(x_train, part2_y_train)
part2_y_pred = part2_classification.predict(x_test)

average_precision = precision_score(part2_y_test, part2_y_pred, average="macro")
average_recall = recall_score(part2_y_test, part2_y_pred, average="macro")
accuracy = accuracy_score(part2_y_test, part2_y_pred)

print("\naverage_precision: ", average_precision)
print("average_recall: ", average_recall)
print("accuracy: ", accuracy)

# # write to csv file
# part2_summary = pd.DataFrame({"zid": ["z5241154"], "average_precision": [np.around(average_precision, 2)],
#                               "average_recall": [np.around(average_recall, 2)], "accuracy": [np.around(accuracy, 2)]})
# part2_summary.to_csv("z5241154.PART2.summary.csv", index=False)
# part2_output = pd.DataFrame({"movie_id": movie_id, "predicted_revenue": part2_y_pred.tolist()})
# part2_output.sort_values(by="movie_id", ascending=True, inplace=True)
# part2_output.to_csv("z5241154.PART2.output.csv", index=False)
