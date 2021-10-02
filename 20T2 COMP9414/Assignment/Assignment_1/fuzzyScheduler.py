# @file: fuzzyScheduler.py
# @author: Hongxiao Jin
# @creat_time: 2020/6/18 15:39

import sys
from operator import ge, eq, le

from searchGeneric import AStarSearcher
from cspConsistency import Search_with_AC_from_CSP, Con_solver
from cspProblem import Constraint, CSP

# ------------------------------------- read file -------------------------------------

# file_name = sys.argv[1]  # get file name

file_name = 'input1.txt'
file = open(file_name, 'r')
file_content = []  # list to save every word in the file by lines

# get a list with all lines, e.g. file_content = [['task,', 't1', '3'], ['task,', 't2', '4']]
for line in file:
    line = line.replace(',', '')
    file_content.append(line.split())

# ---------------------------- get values from file_content ----------------------------

task_dur = {}  # task's name and its duration, e.g. task_dur = {task1: dur1, task2:dur2,...}
binary_constraints = []  # relationship between two tasks, e.g. binary_constraints = [['t1','before','t2']]
hard_domain_constraints = []  # hard domain constraints, e.g. hard_domain_constraints = [['t2', 'mon']]
soft_deadline_constraints = []  # e.g. soft_deadline_constraints = [['t1', 'ends-by', 'mon', '3pm', '10']]

for item in file_content:
    if item:
        if item[0] == 'task':  # find lines include tasks with name and duration
            task_dur[item[1]] = item[2]
        elif item[0] == 'constraint':  # find lines includes binary constraints
            binary_constraints.append([item[1], item[2], item[3]])
        elif item[0] == 'domain' and item[2] != 'ends-by':  # find lines includes hard domain constraints
            hard_domain_constraints.append(item[1:])
        elif item[0] == 'domain' and item[2] == 'ends-by':  # find lines includes soft deadline constraints
            soft_deadline_constraints.append(item[1:])

# ------------------------------- calculate task_domains -------------------------------

weekday_dic = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
clock_dic = {'9am': 1, '10am': 2, '11am': 3, '12pm': 4, '1pm': 5, '2pm': 6, '3pm': 7, '4pm': 8, '5pm': 9}

re_dic_weekday = {1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri'}
re_dic_clock = {1: '9am', 2: '10am', 3: '11am', 4: '12pm', 5: '1pm', 6: '2pm', 7: '3pm', 8: '4pm', 9: '5pm'}

# weekday*10+clock, e.g. 11 means 'mon','9am'
timetable = [11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             51, 52, 53, 54, 55, 56, 57, 58, 59]

task_domain = {}  # task's name and its possible domain, e.g. task_domain = {'t1': [(11, 14)]}
for task in task_dur:
    domain = []
    for time in timetable:
        if time // 10 == (time + int(task_dur[task])) // 10:
            domain.append(time)
            # domain.append(tuple(sorted((time, time + int(task_dur[task])))))
    task_domain[task] = set(sorted(domain))

# ---------------------------- get hard_constraints ----------------------------

# limits includes binary constraints and hard_domain_constraints,
# hard_constraints = [Constraint(scope, condition)]
hard_constraints = []

# first step, deal with binary_constraints
for item in binary_constraints:
    scope = (item[0], item[2])
    if item[1] == 'before':  # t1 ends when or before t2 starts
        time_before = lambda c: lambda a, b: le(a + int(task_dur[c]), b)
        hard_constraints.append(Constraint(scope, time_before(item[0])))
    elif item[1] == 'after':  # t1 starts after or when t2 ends
        time_after = lambda c: lambda a, b: ge(a, b + int(task_dur[c]))
        hard_constraints.append(Constraint(scope, time_after(item[2])))
    elif item[1] == 'same-day':  # t1 and t2 are scheduled on the same day
        same_day = lambda a, b: eq(int(a / 10), int(b / 10))
        hard_constraints.append(Constraint(scope, same_day))
    elif item[1] == 'starts-at':  # t1 starts exactly when t2 ends
        starts_at = lambda c: lambda a, b: eq(a, b + int(task_dur[c]))
        hard_constraints.append(Constraint(scope, starts_at(item[2])))


# second step, deal with hard_domain_constraints

# the hyphen format: ['mon','9am-mon','10am'], remove it between clock and weekday
# return a new list without the hyphen, such as ['mon','9am', 'mon','10am']
def replace_hyphen(spe_item):
    new_list = [spe_item[0], spe_item[1][:list(spe_item[1]).index('-')],
                spe_item[1][list(spe_item[1]).index('-') + 1:], spe_item[2]]
    return new_list


def starts_range(ran):
    def starts_in(a):
        return ge(a, (weekday_dic[ran[0]] * 10 + clock_dic[ran[1]])) and \
               le(a, (weekday_dic[ran[2]] * 10 + clock_dic[ran[3]]))

    return starts_in


def ends_range(ta, ran):
    def ends_in(a):
        return ge(a + int(task_dur[ta]), (weekday_dic[ran[0]] * 10 + clock_dic[ran[1]])) and \
               le(a + int(task_dur[ta]), (weekday_dic[ran[2]] * 10 + clock_dic[ran[3]]))

    return ends_in


for item in hard_domain_constraints:
    scope = (item[0],)
    if item[1] in re_dic_weekday.values():  # t starts on given day at any time
        on_given_day = lambda b: lambda a: eq(re_dic_weekday[(a // 10)], b)
        hard_constraints.append(Constraint(scope, on_given_day(item[1])))
    elif item[1] in re_dic_clock.values():  # t starts at given time on any day
        at_given_time = lambda b: lambda a: eq(re_dic_clock[(a % 10)], b)
        hard_constraints.append(Constraint(scope, at_given_time(item[1])))
    elif item[1] == 'starts-before' and len(item) == 4:  # at or before given time
        starts_before_day_time = lambda b, c: lambda a: le(a, (weekday_dic[b] * 10 + clock_dic[c]))
        hard_constraints.append(Constraint(scope, starts_before_day_time(item[2], item[3])))
    elif item[1] == 'starts-after' and len(item) == 4:  # at or after given time
        starts_after_day_time = lambda b, c: lambda a: ge(a, (weekday_dic[b] * 10 + clock_dic[c]))
        hard_constraints.append(Constraint(scope, starts_after_day_time(item[2], item[3])))
    if item[1] == 'ends-before' and len(item) == 4:  # at or before given time
        ends_before_day_time = lambda b, c, d: lambda a: le(a + int(task_dur[b]), (weekday_dic[c] * 10 + clock_dic[d]))
        hard_constraints.append(Constraint(scope, ends_before_day_time(item[0], item[2], item[3])))
    elif item[1] == 'ends-after' and len(item) == 4:  # at or after given time
        ends_after_day_time = lambda b, c, d: lambda a: ge(a + int(task_dur[b]), (weekday_dic[c] * 10 + clock_dic[d]))
        hard_constraints.append(Constraint(scope, ends_after_day_time(item[0], item[2], item[3])))
    elif item[1] == 'starts-in':  # day-time range
        new_item = replace_hyphen(item[2:])
        hard_constraints.append(Constraint(scope, starts_range(new_item)))
    elif item[1] == 'ends-in':  # day-time range
        new_item = replace_hyphen(item[2:])
        hard_constraints.append(Constraint(scope, ends_range(item[0], new_item)))
    elif item[1] == 'starts-before' and len(item) == 3:  # at or before time on any day
        starts_before_time = lambda b: lambda a: le(int(a % 10), clock_dic[b])
        hard_constraints.append(Constraint(scope, starts_before_time(item[2])))
    elif item[1] == 'ends-before' and len(item) == 3:  # at or before time on any day
        ends_before_time = lambda b, c: lambda a: le(int((a + int(task_dur[b])) % 10), clock_dic[c])
        hard_constraints.append(Constraint(scope, ends_before_time(item[0], item[2])))
    elif item[1] == 'starts-after' and len(item) == 3:  # at or after time on any day
        starts_after_time = lambda b: lambda a: ge(a % 10, clock_dic[b])
        hard_constraints.append(Constraint(scope, starts_after_time(item[2])))
    elif item[1] == 'ends-after' and len(item) == 3:  # at or after time on any day
        ends_after_time = lambda b, c: lambda a: ge(int((a + int(task_dur[b])) % 10), clock_dic[c])
        hard_constraints.append(Constraint(scope, ends_after_time(item[0], item[2])))

# ---------------------------- get soft_constraints ----------------------------

# limits includes deadline and cost per hour of missing deadline,
# e.g. soft_constraints = {'t1': [deadline, cost]}
soft_constraints = {}

for item in soft_deadline_constraints:
    temp = [(weekday_dic[item[2]] * 10 + clock_dic[item[3]]), int(item[4])]
    soft_constraints[item[0]] = temp


# ------------------------------ create a new CSP class ------------------------------

# Add costs to CSPs by extending CSP
class CSP_with_cost(CSP):
    def __init__(self, domains, hard_cons, soft_cons):
        super().__init__(domains, hard_cons)
        self.soft_cons = soft_cons


class greedy_search(AStarSearcher):
    def add_to_frontier(self, path):
        value = self.problem.heuristic(path.end())
        self.frontier.add(path, value)

    AStarSearcher.max_display_level = 0


# ------------------------------ create Search_with_AC_from_Cost_CSP ------------------------------

# Search_with_AC_from_Cost_CSP has the same methods as Search_with_AC_from_CSP
# but implements domain splitting over constraint optimization problems.
class Search_with_AC_from_Cost_CSP(Search_with_AC_from_CSP):
    def __init__(self, pro):
        self.cons = Con_solver(pro)  # copy of the CSP
        self.domains = self.cons.make_arc_consistent()
        self.soft = pro.soft_cons

    # calculate the min cost
    def heuristic(self, solution):
        result = 0  # result = min_cost('t1')+ min_cost('t2')
        sol_cost = {}  # save min cost spent by different task, sol_cost= {'t1': 10}
        if self.soft:
            for node in solution:
                cost = []  # temp cost list , save all the possible cost spent by one task
                for dom in solution[node]:
                    if node in self.soft:
                        end_time = dom + int(task_dur[node])
                        if end_time > self.soft[node][0]:
                            if (end_time // 10) == (self.soft[node][0] // 10):
                                cost.append((end_time - self.soft[node][0]) * self.soft[node][1])
                            else:
                                suppose_dur = (end_time // 10 - self.soft[node][0] // 10) * 24
                                real = suppose_dur + (end_time % 10 - self.soft[node][0] % 10)
                                cost.append(real * self.soft[node][1])
                        else:
                            cost.append(0)
                    else:
                        cost.append(0)
                if cost:
                    sol_cost[node] = min(cost)

            for node in sol_cost:
                result = result + sol_cost[node]

        return result


csp = CSP_with_cost(task_domain, hard_constraints, soft_constraints)
problem = Search_with_AC_from_Cost_CSP(csp)
sol = greedy_search(problem).search()
if sol is not None:
    sol = sol.end()
    for task in sol:
        weekday = re_dic_weekday[int(list(sol[task])[0]) // 10]
        clock = re_dic_clock[int(list(sol[task])[0]) % 10]
        print(task + ':' + weekday + ' ' + clock)
    min_cost = problem.heuristic(sol)
    print('cost:' + str(min_cost))
else:
    print('No solution')
