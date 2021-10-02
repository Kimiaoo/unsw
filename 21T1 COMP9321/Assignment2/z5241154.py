import json
import datetime

from flask import Flask, request, Response, send_file
from flask_restx import Api, Resource, reqparse
import sqlite3
from urllib import request as req
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app, title="COMP9321 Assignment 2", description="z5241154", )

db_con = sqlite3.connect('z5241154.db', check_same_thread=False)
cur = db_con.cursor()

# create table tvshows
sql_create_tb = "CREATE TABLE IF NOT EXISTS tvshows (id INTEGER NOT NULL PRIMARY KEY, tvmaze_id INTEGER, " \
                "last_update TEXT, name TEXT, type TEXT, language TEXT, genres TEXT, " \
                "status TEXT, runtime REAL, premiered TEXT, officialSite TEXT, schedule TEXT, " \
                "rating REAL, weight REAL, network TEXT, summary TEXT, _links TEXT)"
cur.execute(sql_create_tb)


@api.route('/tv-shows/import')
@api.response(200, 'OK')
@api.response(201, 'Created')
@api.response(404, 'Not Found')
class ImportTVShows(Resource):
    @api.doc(params={'name': 'Title for the tv show'})
    def post(self):
        # get the title of the tv show
        parameters = reqparse.RequestParser()
        parameters.add_argument('name', type=str, required=True)
        input_dic = parameters.parse_args()
        name = input_dic['name']

        # match names
        comp_name = name.upper()
        name = name.replace(' ', '-')
        name = name.upper()
        # get info from tvmaze.com
        url = "http://api.tvmaze.com/search/shows?q=" + name
        reqs = req.Request(url)
        info = json.loads(req.urlopen(reqs).read())

        # check the name is totally same as the input
        if info and (comp_name == info[0]["show"]["name"].upper()):
            tvmaze_id = int(info[0]["show"]["id"])
            last_update = str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
            name = str(info[0]["show"]["name"])
            show_type = str(info[0]["show"]["type"])
            language = str(info[0]["show"]["language"])
            genres = str(info[0]["show"]["genres"])
            status = str(info[0]["show"]["status"])
            runtime = str(info[0]["show"]["runtime"])
            premiered = str(info[0]["show"]["premiered"])
            officialSite = str(info[0]["show"]["officialSite"])
            schedule = str(info[0]["show"]["schedule"])
            # if info[0]["show"]["rating"]["average"] is None:
            #     rating = 0
            # else:
            #     rating = float(info[0]["show"]["rating"]["average"])
            if info[0]["show"]["rating"]["average"] is not None:
                rating = float(info[0]["show"]["rating"]["average"])
            else:
                rating = None
            weight = float(info[0]["show"]["weight"])
            network = str(info[0]["show"]["network"])
            summary = str(info[0]["show"]["summary"])

            # insert info into db
            sql_insert_info = "INSERT INTO tvshows (tvmaze_id, last_update, name, type, language, genres, " \
                              "status, runtime, premiered, officialSite, schedule, rating, weight, network, " \
                              "summary) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            cur.execute(sql_insert_info, (tvmaze_id, last_update, name, show_type, language, genres, status, runtime,
                                          premiered, officialSite, schedule, rating, weight, network, summary))

            # get pk id
            sql_select_id = "SELECT id FROM tvshows WHERE tvmaze_id=?"
            cur.execute(sql_select_id, [tvmaze_id])
            query_id = cur.fetchone()

            # update _links
            info[0]["show"]["_links"]["self"]["href"] = f"http://{request.host}/tv-shows/{query_id[0]}"
            link = str({"self": info[0]["show"]["_links"]["self"]})
            sql_update_links = "UPDATE tvshows SET _links=? WHERE id=?"
            cur.execute(sql_update_links, (link, query_id[0]))

            res = {"id": query_id[0], "last-update": last_update, "tvmaze-id": tvmaze_id, "_links": eval(link)}

            db_con.commit()
            return res, 201
        else:
            return 'No tv shows found!', 404


@api.route('/tv-shows/<id>')
@api.response(200, 'OK')
@api.response(404, 'Not Found')
class DealWithTVShows(Resource):
    def get(self, id):
        sql_count = "SELECT id FROM tvshows"
        cur.execute(sql_count)
        id_list = cur.fetchall()
        ids = []
        for item in id_list:
            ids.append(item[0])

        if int(id) not in ids:
            return "No such TV show in database!", 404
        else:
            sql_count = "SELECT MAX(id) FROM tvshows"
            cur.execute(sql_count)
            num = cur.fetchone()[0]

            # get all info in db
            sql_select_info = "SELECT * FROM tvshows WHERE id=?"
            cur.execute(sql_select_info, [id])
            db_info = cur.fetchall()[0]

            links = eval(db_info[16])

            # add previous link
            sql_select_link = "SELECT tvmaze_id FROM tvshows WHERE id=?"
            for i in range(int(id) - 1, 0, -1):
                cur.execute(sql_select_link, [i])
                query_id = cur.fetchone()
                if query_id:
                    links["previous"] = {"href": f"http://{request.host}/tv-shows/{i}"}
                    break

            # add next link
            sql_select_link = "SELECT tvmaze_id FROM tvshows WHERE id=?"
            for i in range(int(id) + 1, num + 1):
                cur.execute(sql_select_link, [i])
                query_id = cur.fetchone()
                if query_id:
                    links["next"] = {"href": f"http://{request.host}/tv-shows/{i}"}
                    break

            rating = {"average": db_info[12]}
            res = {"tvmaze_id": db_info[1], "id": db_info[0], "last-update": db_info[2], "name": db_info[3],
                   "type": db_info[4], "language": db_info[5], "genres": eval(db_info[6]), "status": db_info[7],
                   "runtime": db_info[8], "premiered": db_info[9], "officialSite": db_info[10],
                   "schedule": eval(db_info[11]), "rating": rating, "weight": db_info[13],
                   "network": eval(db_info[14]), "summary": db_info[15], "_links": links}
            db_con.commit()
            return res, 200

    def delete(self, id):
        sql_count = "SELECT id FROM tvshows"
        cur.execute(sql_count)
        id_list = cur.fetchall()
        ids = []
        for item in id_list:
            ids.append(item[0])

        if int(id) not in ids:
            return "No such TV show in database!", 404
        else:
            # delete the show
            sql_delete_info = "DELETE FROM tvshows WHERE id=?"
            cur.execute(sql_delete_info, [id])

            res = {"message": f"The tv show with id {id} was removed from the database!",
                   "id": id}

            db_con.commit()
            return res, 200

    @api.doc(params={'payload': 'TV show attributes'})
    def patch(self, id):
        sql_count = "SELECT id FROM tvshows"
        cur.execute(sql_count)
        id_list = cur.fetchall()
        ids = []
        for item in id_list:
            ids.append(item[0])

        if int(id) not in ids:
            return "No such TV show in database!", 404
        else:
            parameters = reqparse.RequestParser()
            parameters.add_argument('payload', type=str, required=True)
            input_dic = parameters.parse_args()
            attr = input_dic['payload']
            attr = eval(attr)

            # get all info in db
            sql_select_info = "SELECT * FROM tvshows WHERE id=?"
            cur.execute(sql_select_info, [id])
            db_info = cur.fetchall()[0]

            tvmaze_id = db_info[1]
            name = db_info[3]
            show_type = db_info[4]
            language = db_info[5]
            genres = db_info[6]
            status = db_info[7]
            runtime = db_info[8]
            premiered = db_info[9]
            officialSite = db_info[10]
            schedule = db_info[11]
            rating = db_info[12]
            weight = db_info[13]
            network = db_info[14]
            summary = db_info[15]
            links = db_info[16]

            for item in attr:
                if item == "tvmaze_id":
                    tvmaze_id = int(attr[item])
                elif item == "name":
                    name = str(attr[item])
                elif item == "type":
                    show_type = str(attr[item])
                elif item == "language":
                    language = str(attr[item])
                elif item == "genres":
                    genres = str(attr[item])
                elif item == "status":
                    status = str(attr[item])
                elif item == "runtime":
                    runtime = str(attr[item])
                elif item == "premiered":
                    premiered = str(attr[item])
                elif item == "officialSite":
                    officialSite = str(attr[item])
                elif item == "schedule":
                    schedule = str(attr[item])
                elif item == "rating":
                    rating = float(attr[item]["average"])
                elif item == "weight":
                    weight = float(attr[item])
                elif item == "network":
                    network = str(attr[item])
                elif item == "summary":
                    summary = str(attr[item])
                elif item == "_links":
                    links = str(attr[item])

            last_update = str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

            sql_update_info = "UPDATE tvshows SET tvmaze_id=?, last_update=?, name=?, type=?, language=?, genres=?, " \
                              "status=?, runtime=?, premiered=?, officialSite=?, schedule=?, rating=?, weight=?, " \
                              "network=?, summary=?, _links=? WHERE id=?"
            cur.execute(sql_update_info, (tvmaze_id, last_update, name, show_type, language, genres, status, runtime,
                                          premiered, officialSite, schedule, rating, weight, network, summary, links,
                                          id))

            res = {"id": id, "last-update": last_update, "_links": eval(links)}
            db_con.commit()
            return res, 200


@api.route('/tv-shows')
@api.response(200, 'OK')
@api.response(404, 'Not Found')
class RetrieveTVShows(Resource):
    @api.doc(params={'order_by': 'order_by', 'page': 'pagination',
                     'page_size': 'page size', 'filter': 'filter'})
    def get(self):
        # get parameters
        parameters = reqparse.RequestParser()
        parameters.add_argument('order_by', type=str, default='+id')
        parameters.add_argument('page', type=int, default=1)
        parameters.add_argument('page_size', type=int, default=100)
        parameters.add_argument('filter', type=str, default='id,name')
        input_dic = parameters.parse_args()
        order_by = input_dic["order_by"]
        page = input_dic["page"]
        page_size = input_dic["page_size"]
        pfilter = input_dic["filter"]

        # find the num of data
        sql_count = "SELECT COUNT(*) FROM tvshows"
        cur.execute(sql_count)
        total_num = cur.fetchone()[0]
        if page > 1 and (total_num / page_size <= 1):
            return "Page is wrong.", 404
        else:
            links = {}
            total_page_num = int(total_num / page_size)

            if page == 1:
                links["self"] = {"href": f"http://{request.host}//tv-shows?page={page},page_size={page_size}"}
                if total_page_num > 1:
                    links["next"] = {"href": f"http://{request.host}//tv-shows?page={page + 1},page_size={page_size}"}
            elif page == total_page_num:
                links["self"] = {"href": f"http://{request.host}//tv-shows?page={page},page_size={page_size}"}
                links["previous"] = {"href": f"http://{request.host}//tv-shows?page={page - 1},page_size={page_size}"}
            else:
                links["self"] = {"href": f"http://{request.host}//tv-shows?page={page},page_size={page_size}"}
                links["previous"] = {"href": f"http://{request.host}//tv-shows?page={page - 1},page_size={page_size}"}
                links["next"] = {"href": f"http://{request.host}//tv-shows?page={page + 1},page_size={page_size}"}

        sort_val = ''
        order_by = order_by.split(',')
        for item in order_by:
            flag = item[1:]
            if flag == "rating-average":
                flag = "rating"
            if item[0] == "+":
                # sort_val.append(f"{flag} ASC")
                sort_val += f"{flag} ASC,"
            elif item[0] == "-":
                # sort_val.append(f"{flag} DESC")
                sort_val += f"{flag} DESC,"

        sql_select_info = "SELECT * FROM tvshows ORDER BY " + sort_val[:-1]
        cur.execute(sql_select_info)
        select_res = cur.fetchall()

        show_list = []
        pfilter = pfilter.split(',')

        for i in range(len(select_res)):
            temp_dict = {}
            for att in pfilter:
                if att == "id":
                    temp_dict["id"] = select_res[i][0]
                elif att == "tvmaze_id":
                    temp_dict["tvmaze_id"] = select_res[i][1]
                elif att == "last-update":
                    temp_dict["last-update"] = select_res[i][2]
                elif att == "name":
                    temp_dict["name"] = select_res[i][3]
                elif att == "type":
                    temp_dict["type"] = select_res[i][4]
                elif att == "language":
                    temp_dict["language"] = select_res[i][5]
                elif att == "genres":
                    temp_dict["genres"] = select_res[i][6]
                elif att == "status":
                    temp_dict["status"] = select_res[i][7]
                elif att == "runtime":
                    temp_dict["runtime"] = select_res[i][8]
                elif att == "premiered":
                    temp_dict["premiered"] = select_res[i][9]
                elif att == "officialSite":
                    temp_dict["officialSite"] = select_res[i][10]
                elif att == "schedule":
                    temp_dict["schedule"] = select_res[i][11]
                elif att == "rating":
                    rating_average = {"average": select_res[i][12]}
                    temp_dict["rating"] = rating_average
                elif att == "weight":
                    temp_dict["weight"] = select_res[i][13]
                elif att == "network":
                    temp_dict["network"] = select_res[i][14]
                elif att == "summary":
                    temp_dict["summary"] = select_res[i][15]
            if temp_dict:
                show_list.append(temp_dict)

        start_page = (page - 1) * page_size
        end_page = page * page_size
        res = {"page": page, "page-size": page_size, "tv-shows": show_list[start_page:end_page], "_links": links}
        db_con.commit()
        return res, 200


@api.route('/tv-shows/statistics')
@api.response(200, 'OK')
@api.response(404, 'Not Found')
class StatisticsTVShows(Resource):
    @api.doc(params={'format': 'json/image', 'by': 'attributes'})
    def get(self):
        parameters = reqparse.RequestParser()
        parameters.add_argument('format', type=str, required=True)
        parameters.add_argument('by', type=str, required=True)
        input_dic = parameters.parse_args()
        q6_format = input_dic['format']
        q6_by = input_dic['by']

        # count total_num
        sql_count = "SELECT COUNT(*) FROM tvshows"
        cur.execute(sql_count)
        total_num = cur.fetchone()[0]

        # count the number of update data
        sql_select_time = "SELECT last_update FROM tvshows"
        cur.execute(sql_select_time)
        time_info = cur.fetchall()
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%d %H:%M:%S')

        count_update = 0
        for i in range(len(time_info)):
            write_time = datetime.datetime.strptime(time_info[i][0], '%Y-%m-%d-%H:%M:%S')
            delta = (cur_time - write_time).days
            if delta < 1:
                count_update += 1

        # calculate values
        if q6_by == "language":
            sql_select_value = "SELECT language FROM tvshows"
            cur.execute(sql_select_value)
        elif q6_by == "genres":
            sql_select_value = "SELECT genres FROM tvshows"
            cur.execute(sql_select_value)
        elif q6_by == "status":
            sql_select_value = "SELECT status FROM tvshows"
            cur.execute(sql_select_value)
        elif q6_by == "type":
            sql_select_value = "SELECT type FROM tvshows"
            cur.execute(sql_select_value)
        val_info = cur.fetchall()

        value_list = []
        for i in range(len(val_info)):
            if q6_by == "genres":
                val = eval(val_info[i][0])
            else:
                val = val_info[i][0].split(",")
            for j in range(len(val)):
                value_list.append(val[j].upper())

        total_value = len(value_list)
        value_count = {}
        for val in value_list:
            if val not in value_count:
                value_count[val] = 1
            else:
                value_count[val] += 1
        for val in value_count:
            value_count[val] = float('%0.1f' % (value_count[val] * 100 / total_value))

        if q6_format == "json":
            res = {"total": total_num, "total-updated": count_update, "values": value_count}
            return res, 200
        else:
            plt.figure(figsize=(11, 11))

            plt.subplot(2, 1, 1)
            labels = ['not update', 'update in 24 hours']
            val = [total_num - count_update, count_update]
            plt.pie(x=val, labels=labels, colors=['tomato', 'c'], autopct="%0.2f%%")
            plt.legend(loc='upper right', fontsize=8)

            plt.subplot(2, 1, 2)
            labels = list(value_count.keys())
            val = list(value_count.values())
            plt.pie(x=val, labels=labels, autopct="%0.2f%%")
            plt.legend(loc='upper right', fontsize=8)

            image_file = "res.jpg"
            plt.savefig(image_file)
            plt.clf()
            res = send_file(image_file, mimetype="image/jpg")
            db_con.commit()
            return res


if __name__ == '__main__':
    app.run(debug=True)
