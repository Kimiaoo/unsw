# @file = maze.py
# @author = Hongxiao Jin
# @creat_time = 18/11/2019 7:28 pm
import copy


class MazeError(Exception):
    def __init__(self, message):
        self.message = message


class Maze(object):

    # init object
    def __init__(self, file_name):
        # this used to save the digits in each line,
        # every object in this list represents one line
        self.file_list = []

        # x_dim represents the number of members in each line
        # y_dim represents the number of lines
        self.x_dim = 0
        self.y_dim = 0

        # read file
        self.read_file(file_name)
        # check file
        self.check_file()

        # get file name
        self.txt_file_name = file_name

        # turn the matrix to 9 X 9
        self.gate_graph = []
        self.wall_graph = []

        # standard direction
        self.dir = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        # gate position list
        self.gate_position = []
        # zero position list
        self.zero_position = []
        # inner points
        self.inner_points = []
        # compared inner point
        self.compared_list = []
        # colour list
        self.colour_list = []
        # spikes position
        self.spike_position = []
        # save all spikes
        self.all_spikes = []

        # number of gates
        self.gate_result = 0
        # number of wall sets
        self.wall_result = 0
        # inner points
        self.inner_result = 0
        # accessible area
        self.accessible_area_result = 0
        # count the number of false path
        self.false_path_result = 0
        # count the number of true path
        self.true_path_result = 0

    # turn the last 2 rows and lines to -1
    def remove_unnecessary(self, redefined_graph):
        # 1 0 1       1 -1 -1
        # 0 0 0   ->  1 -1 -1
        # 1 0 1       1 -1 -1
        del_line_start = self.x_dim - 1
        del_line_finish = self.x_dim * self.y_dim
        for i in range(del_line_start, del_line_finish, self.x_dim):
            redefined_graph[i][1] = -1
            redefined_graph[i][2] = -1
            redefined_graph[i][4] = -1
            redefined_graph[i][5] = -1
            redefined_graph[i][7] = -1
            redefined_graph[i][8] = -1

        # 1 0 1        1  0  1
        # 0 0 0   ->  -1 -1 -1
        # 1 0 1       -1 -1 -1
        del_row_start = (self.y_dim - 1) * self.x_dim
        del_row_finish = self.x_dim * self.y_dim
        for i in range(del_row_start, del_row_finish):
            redefined_graph[i][3] = -1
            redefined_graph[i][4] = -1
            redefined_graph[i][5] = -1
            redefined_graph[i][6] = -1
            redefined_graph[i][7] = -1
            redefined_graph[i][8] = -1

        return redefined_graph

    # put redefined_graph to a regular list
    def get_standard_graph(self, redefined_graph):
        temp_graph = []
        start_position = 0
        finish_position = self.x_dim
        for line in range(self.y_dim):
            # line 1
            temp_line = []
            for i in range(start_position, finish_position):
                for index in range(0, 3):
                    if redefined_graph[i][index] >= 0:
                        temp_line.append(redefined_graph[i][index])
                        redefined_graph[i][index] = -2
            if temp_line:
                temp_graph.append(temp_line)
            # lin 2
            temp_line = []
            for i in range(start_position, finish_position):
                for index in range(3, 6):
                    if redefined_graph[i][index] >= 0:
                        temp_line.append(redefined_graph[i][index])
                        redefined_graph[i][index] = -2
            if temp_line:
                temp_graph.append(temp_line)
            # line 3
            temp_line = []
            for i in range(start_position, finish_position):
                for index in range(6, 9):
                    if redefined_graph[i][index] >= 0:
                        temp_line.append(redefined_graph[i][index])
                        redefined_graph[i][index] = -2
            if temp_line:
                temp_graph.append(temp_line)

            start_position += self.x_dim
            finish_position += self.x_dim
        return temp_graph

    # from quiz_7
    # colour path where value is 1
    def colour_one_path(self, cur_x, cur_y, col, graph):
        graph[cur_x][cur_y] = col
        for (dir_x, dir_y) in self.dir:
            next_x = cur_x + dir_x
            next_y = cur_y + dir_y
            if 0 <= next_x < len(graph) and 0 <= next_y < len(graph[0]):
                if graph[next_x][next_y] == 1:
                    self.colour_one_path(next_x, next_y, col, graph)

    # get gate position
    def get_gate_position(self, graph):
        for row in range(len(graph[0])):
            if graph[0][row] == 0:
                self.gate_position.append((0, row))
            if graph[-1][row] == 0:
                self.gate_position.append((len(graph) - 1, row))

        for line in range(len(graph)):
            if graph[line][0] == 0:
                self.gate_position.append((line, 0))
            if graph[line][-1] == 0:
                self.gate_position.append((line, len(graph[0]) - 1))

    # get all zero position
    def get_zero_position(self, graph):
        for line in range(len(graph)):
            for row in range(len(graph[0])):
                if graph[line][row] == 0:
                    self.zero_position.append((line, row))

    # from quiz_7
    # colour path where value is 0
    def colour_zero_path(self, cur_x, cur_y, col, graph):
        if (cur_x, cur_y) not in self.inner_points:
            graph[cur_x][cur_y] = col
            for (dir_x, dir_y) in self.dir:
                next_x = cur_x + dir_x
                next_y = cur_y + dir_y
                if 0 <= next_x < len(graph) and 0 <= next_y < len(graph[0]):
                    if graph[next_x][next_y] == 0:
                        self.colour_zero_path(next_x, next_y, col, graph)

    # find spikes
    def find_spikes(self, graph, col):
        for cur_x in range(len(graph)):
            for cur_y in range(len(graph[0])):
                if graph[cur_x][cur_y] == col:
                    count_col = 0
                    for (dir_x, dir_y) in self.dir:
                        around_x = cur_x + dir_x
                        around_y = cur_y + dir_y
                        if 0 <= around_x < len(graph) and 0 <= around_y < len(graph[0]):
                            if graph[around_x][around_y] == 1:
                                count_col += 1
                    if count_col == 3:
                        if (cur_x, cur_y) not in self.spike_position:
                            self.spike_position.append((cur_x, cur_y))

    # look around
    def look_around(self, cur_x, cur_y):
        for (dir_x, dir_y) in self.dir:
            around_x = cur_x + dir_x
            around_y = cur_y + dir_y
            if (around_x, around_y) in self.all_spikes:
                self.all_spikes.remove((around_x, around_y))
                self.look_around(around_x, around_y)

    # use this function to read file
    def read_file(self, file_name):
        file = open(file_name, 'r', encoding='utf8')
        for line in file:
            if line is not None:
                line_list = list("".join(line.split()))
                # remove []
                if len(line_list) > 0:
                    self.file_list.append(line_list)

    # check whether the inputs satisfied with the requirements
    def check_file(self):
        self.y_dim = len(self.file_list)
        self.x_dim = len(self.file_list[0])

        # each line must contain the same number of digits
        for i in range(1, self.y_dim):
            if len(self.file_list[i]) != self.x_dim:
                raise MazeError('Incorrect input.')

        # the input is incorrect in that it does not contain only digits in {0,1,2,3} besides spaces
        for j in range(self.y_dim):
            for i in range(self.x_dim):
                if self.file_list[j][i] not in ['0', '1', '2', '3']:
                    raise MazeError('Incorrect input.')

        # x_dim and y_dim are at least equal to 2 and at most equal to 31 and 41, respectively
        if self.y_dim < 2 or self.y_dim > 41:
            raise MazeError('Incorrect input.')
        if self.x_dim < 2 or self.x_dim > 31:
            raise MazeError('Incorrect input.')

        # The last digit on every line with digits cannot be equal to 1 or 3
        for i in range(self.y_dim):
            if self.file_list[i][-1] in ['1', '3']:
                raise MazeError('Input does not represent a maze.')

        # the digits on the last line with digits cannot be equal to 2 or 3
        for i in range(self.x_dim):
            if self.file_list[-1][i] in ['2', '3']:
                raise MazeError('Input does not represent a maze.')

    # count the number of gates
    def count_gate(self):
        redefined_gate = []
        for j in range(self.y_dim):
            for i in range(self.x_dim):
                if self.file_list[j][i] == '0':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 0, 1, 0, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_gate.append([1, 0, 1, 0, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 0, 1, 0, 0, 0, 1, 1, 1])
                    else:
                        redefined_gate.append([1, 0, 1, 0, 0, 0, 1, 0, 1])
                if self.file_list[j][i] == '1':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 1, 1, 0, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_gate.append([1, 1, 1, 0, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 1, 1, 0, 0, 0, 1, 1, 1])
                    else:
                        redefined_gate.append([1, 1, 1, 0, 0, 0, 1, 0, 1])
                if self.file_list[j][i] == '2':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 0, 1, 1, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_gate.append([1, 0, 1, 1, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 0, 1, 1, 0, 0, 1, 1, 1])
                    else:
                        redefined_gate.append([1, 0, 1, 1, 0, 0, 1, 0, 1])
                if self.file_list[j][i] == '3':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 1, 1, 1, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_gate.append([1, 1, 1, 1, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_gate.append([1, 1, 1, 1, 0, 0, 1, 1, 1])
                    else:
                        redefined_gate.append([1, 1, 1, 1, 0, 0, 1, 0, 1])

        redefined_gate = self.remove_unnecessary(redefined_gate)

        # 1|01|1
        # ------
        # 0|01|1
        # 1|01|1
        # ------
        # 1|01|1
        # count the outside 0
        del_line_start = self.x_dim - 1
        del_row_start = (self.y_dim - 1) * self.x_dim
        del_finish = self.x_dim * self.y_dim
        for i in range(0, self.x_dim):
            if redefined_gate[i][1] == 0:
                self.gate_result += 1
        for i in range(0, del_row_start + 1, self.x_dim):
            if redefined_gate[i][3] == 0:
                self.gate_result += 1
        for i in range(del_row_start, del_finish):
            if redefined_gate[i][1] == 0:
                self.gate_result += 1
        for i in range(del_line_start, del_finish, self.x_dim):
            if redefined_gate[i][3] == 0:
                self.gate_result += 1

        self.gate_graph = self.get_standard_graph(redefined_gate)

        if self.gate_result == 0:
            print('The maze has no gate.')
        elif self.gate_result == 1:
            print('The maze has a single gate.')
        else:
            print('The maze has ' + str(self.gate_result) + ' gates.')

    # count the number of walls
    def count_wall(self):
        redefined_wall = []
        for j in range(self.y_dim):
            for i in range(self.x_dim):
                if self.file_list[j][i] == '0':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([0, 0, 1, 0, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_wall.append([0, 0, 1, 0, 0, 1, 0, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([0, 0, 0, 0, 0, 0, 1, 1, 1])
                    else:
                        redefined_wall.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
                if self.file_list[j][i] == '1':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 1, 1, 0, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_wall.append([1, 1, 1, 0, 0, 1, 0, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 1, 1, 0, 0, 0, 1, 1, 1])
                    else:
                        redefined_wall.append([1, 1, 1, 0, 0, 0, 0, 0, 0])
                if self.file_list[j][i] == '2':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 0, 1, 1, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_wall.append([1, 0, 1, 1, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 0, 0, 1, 0, 0, 1, 1, 1])
                    else:
                        redefined_wall.append([1, 0, 0, 1, 0, 0, 1, 0, 0])
                if self.file_list[j][i] == '3':
                    if (i + 1 < self.x_dim and j + 1 < self.y_dim) and \
                            (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3') and \
                            (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 1, 1, 1, 0, 1, 1, 1, 1])
                    elif (i + 1 < self.x_dim) and (self.file_list[j][i + 1] == '2' or self.file_list[j][i + 1] == '3'):
                        redefined_wall.append([1, 1, 1, 1, 0, 1, 1, 0, 1])
                    elif (j + 1 < self.y_dim) and (self.file_list[j + 1][i] == '1' or self.file_list[j + 1][i] == '3'):
                        redefined_wall.append([1, 1, 1, 1, 0, 0, 1, 1, 1])
                    else:
                        redefined_wall.append([1, 1, 1, 1, 0, 0, 1, 0, 0])

        redefined_wall = self.remove_unnecessary(redefined_wall)

        self.wall_graph = self.get_standard_graph(redefined_wall)

        temp_graph = self.wall_graph.copy()
        colour = 2
        # look for the part composed by 1, and colour this part with 2,3...
        for i in range(len(temp_graph)):
            for j in range(len(temp_graph[0])):
                if temp_graph[i][j] == 1:
                    self.colour_one_path(i, j, colour, temp_graph)
                    colour += 1

        self.wall_result = colour - 2

        if self.wall_result == 0:
            print("The maze has no wall.")
        elif self.wall_result == 1:
            print("The maze has walls that are all connected.")
        else:
            print('The maze has ' + str(self.wall_result) + ' sets of walls that are all connected.')

    def count_inner_point(self):
        temp_graph = copy.copy(self.gate_graph)

        self.get_gate_position(temp_graph)
        self.get_zero_position(temp_graph)

        connected_gate_points = self.gate_position.copy()
        self.inner_points = self.zero_position.copy()

        for point in connected_gate_points:
            if point in self.inner_points:
                self.inner_points.remove(point)

        while self.compared_list != self.inner_points:
            self.compared_list = self.inner_points.copy()
            for cur_x in range(len(temp_graph)):
                for cur_y in range(len(temp_graph[0])):
                    if (cur_x, cur_y) in connected_gate_points:
                        for (dir_x, dir_y) in self.dir:
                            next_x = cur_x + dir_x
                            next_y = cur_y + dir_y
                            if 0 <= next_x < len(temp_graph) and 0 <= next_y < len(temp_graph[0]):
                                if temp_graph[next_x][next_y] == 0:
                                    if (next_x, next_y) in self.inner_points:
                                        self.inner_points.remove((next_x, next_y))
                                        connected_gate_points.append((next_x, next_y))

        real_inner_points = []
        for point in self.inner_points:
            x = int(point[0] / 3)
            y = int(point[1] / 3)
            if (x, y) not in real_inner_points:
                real_inner_points.append((x, y))

        self.inner_result = len(real_inner_points)

        if self.inner_result == 0:
            print('The maze has no inaccessible inner point.')
        elif self.inner_result == 1:
            print('The maze has a unique inaccessible inner point.')
        else:
            print('The maze has ' + str(self.inner_result) + ' inaccessible inner points.')

    def count_accessible_area(self):
        temp_graph = copy.copy(self.gate_graph)

        colour = 2
        # look for the part composed by 1, and colour this part with 2,3...

        for i in range(len(temp_graph)):
            for j in range(len(temp_graph[0])):
                if temp_graph[i][j] == 0 and (i, j) not in self.inner_points:
                    self.colour_zero_path(i, j, colour, temp_graph)
                    self.colour_list.append(colour)
                    colour += 1

        self.accessible_area_result = len(self.colour_list)

        if self.accessible_area_result == 0:
            print("The maze has no accessible area.")
        elif self.accessible_area_result == 1:
            print("The maze has a unique accessible area.")
        else:
            print('The maze has ' + str(self.accessible_area_result) + ' accessible areas.')

    def count_false_path(self):
        temp_graph = copy.copy(self.gate_graph)

        for col in self.colour_list:
            self.find_spikes(temp_graph, col)
        while self.spike_position:
            for spike in self.spike_position:
                if spike not in self.all_spikes:
                    self.all_spikes.append(spike)
                temp_graph[spike[0]][spike[1]] = 1
                self.spike_position.remove((spike[0], spike[1]))
            for col in self.colour_list:
                self.find_spikes(temp_graph, col)

        false_path = []
        for cur_x in range(len(temp_graph)):
            for cur_y in range(len(temp_graph[0])):
                if (cur_x, cur_y) in self.all_spikes:
                    false_path.append((cur_x, cur_y))
                    self.look_around(cur_x, cur_y)

        self.false_path_result = len(false_path)

        if self.false_path_result == 0:
            print('The maze has no accessible cul-de-sac.')
        elif self.false_path_result == 1:
            print('The maze has accessible cul-de-sacs that are all connected.')
        else:
            print('The maze has ' +
                  str(self.false_path_result) + ' sets of accessible cul-de-sacs that are all connected.')

    def count_true_path(self):
        temp_graph = copy.copy(self.gate_graph)
        not_only = []

        for col in self.colour_list:
            for cur_x in range(len(temp_graph)):
                for cur_y in range(len(temp_graph[0])):
                    if temp_graph[cur_x][cur_y] == col:
                        check_only = 0
                        for (dir_x, dir_y) in self.dir:
                            around_x = cur_x + dir_x
                            around_y = cur_y + dir_y
                            if 0 <= around_x < len(temp_graph) and 0 <= around_y < len(temp_graph[0]):
                                if temp_graph[around_x][around_y] == col:
                                    check_only += 1
                        if check_only > 2:
                            # temp_graph[cur_x][cur_y] = 'c'
                            if col not in not_only:
                                not_only.append(col)

        possible_paths = []
        for col in self.colour_list:
            if col not in not_only:
                possible_path = []
                for cur_x in range(len(temp_graph)):
                    for cur_y in range(len(temp_graph[0])):
                        if temp_graph[cur_x][cur_y] == col:
                            possible_path.append((cur_x, cur_y))
                if possible_path not in possible_paths and possible_path:
                    possible_paths.append(possible_path)

        true_path = []
        for i in range(len(possible_paths)):
            gate_num = 0
            for gate in self.gate_position:
                if gate in possible_paths[i]:
                    gate_num += 1
            if gate_num == 2:
                true_path.append(possible_paths[i])

        self.true_path_result = len(true_path)

        if self.true_path_result == 0:
            print('The maze has no entry-exit path with no intersection not to cul-de-sacs.')
        elif self.true_path_result == 1:
            print('The maze has a unique entry-exit path with no intersection not to cul-de-sacs.')
        else:
            print('The maze has ' +
                  str(self.true_path_result) + ' entry-exit paths with no intersections not to cul-de-sacs.')

    def display_grid(self, graph):
        for row in graph:
            print('   ', *row)

    # figure out the result
    def analyse(self):
        self.count_gate()
        self.count_wall()
        self.count_inner_point()
        self.count_accessible_area()
        self.count_false_path()
        self.count_true_path()

    def display_walls(self, tex):
        pass

    # draw graph
    def display(self):
        tex_file_name = self.txt_file_name.split('.')[0] + ".tex"
        tex_file = open(tex_file_name, 'w', encoding='utf8')
        tex_file.write('\\documentclass[10pt]{article}\n', )
        tex_file.write('\\usepackage{tikz}\n', )
        tex_file.write('\\usetikzlibrary{shapes.misc}\n', )
        tex_file.write('\\usepackage[margin=0cm]{geometry}\n', )
        tex_file.write('\\pagestyle{empty}\n', )
        tex_file.write('\n', )
        tex_file.write('\\tikzstyle{every node}=[cross out, draw, red]\n', )
        tex_file.write('\n', )
        tex_file.write('\\begin{document}\n', )
        tex_file.write('\\vspace*{\\fill}\n', )
        tex_file.write('\\begin{center}\n', )
        tex_file.write('\\begin{tikzpicture}[x=0.5cm, y=-0.5cm, ultra thick, blue]\n', )
        tex_file.write('% Walls\n')
        self.display_walls(tex_file)
        tex_file.write('% Pillars\n')

        tex_file.write('% Inner points in accessible cul-de-sacs\n')

        tex_file.write('% Entry-exit paths without intersections\n')

        tex_file.write('\\end{tikzpicture}\n', )
        tex_file.write('\\end{center}\n', )
        tex_file.write('\\vspace*{\\fill}\n', )
        tex_file.write('\n', )
        tex_file.write('\\end{document}\n')
