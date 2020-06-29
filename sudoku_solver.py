import math
import time
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)


class SolveSudoku:

    def __init__(self, sudoku_array):
        self.sudoku = sudoku_array                          # initialize the sudoku we wish to solve
        self.w_cell = int(math.sqrt(self.sudoku.shape[0]))  # width and height of each cell
        self.w_sudoku = int(self.sudoku.shape[0])           # width and height of the sudoku (= cell_width^2)
        self.sudoku_flat = np.resize(self.sudoku, [int(math.pow(self.w_cell, 4)), 1])  # create the flat version

        self.create_leave_alone()   # create the leave alone grid

        self.sudoku_index = np.linspace(0, len(self.sudoku_flat)-1, math.pow(self.w_sudoku, 2))
        self.sudoku_index = np.resize(self.sudoku_index, [self.w_sudoku, self.w_sudoku])

        # get a list for each grid index, which is an array consisting of each index we check for a full set of 1-9
        self.all_connected_index = []
        for index in range(0,len(self.sudoku_flat)):
            index_check = np.where(self.sudoku_index == index)
            all_num = self.get_all_associated_num(int(index_check[0]), int(index_check[1]))
            self.all_connected_index.append(all_num)

    def create_leave_alone(self):
        '''
            This function creates the array self.leave_alone, which consists of all indexes of non-empty grids, which
            will be skipped during backtracking, as they are already filled in
        :return:
        '''

        self.leave_alone = []  # list of grid_indexes which can't be changed because they are already filled in
        for i in range(len(self.sudoku_flat)):
            check = self.sudoku_flat[i]
            if check == '' or check == "0":
                self.sudoku_flat[i] = 0
            else:
                self.sudoku_flat[i] = int(self.sudoku_flat[i])
                self.leave_alone.append(i)  # these are the parts we don't need to change


    def get_all_associated_num(self,row_i, column_i):

        '''
            This function will find all of the indexes that belong to a certain other index. This means all indexes
            that belong to the same row, column and cell of a certain grid.
        '''

        index_check = [row_i, column_i]
        index_cur = self.sudoku_index[index_check[0]][index_check[1]]

        row_check = self.sudoku_index[index_check[0], :]
        column_check = self.sudoku_index[:, index_check[1]]

        xtop = math.floor(index_check[0] / self.w_cell) * self.w_cell
        ytop = math.floor(index_check[1] / self.w_cell) * self.w_cell
        xbottom = xtop + self.w_cell
        ybottom = ytop + self.w_cell
        cell_check = self.sudoku_index[xtop:xbottom, ytop:ybottom]
        cell_check = np.resize(cell_check, [self.w_cell * self.w_cell])

        all_unique = (set(row_check) | set(column_check) | set(cell_check)) - (
                    set(row_check) & set(column_check) & set(cell_check))
        all_unique = [int(item) for item in all_unique]
        while int(index_cur) in all_unique: all_unique.remove(index_cur)

        return all_unique

    def check_if_array_correct(self, array_cur):
        """
            This function is used to check if there are any double values in the given array.

            The given array must be a numpy array, and represents a full row, column or cell

            Returns True if there are no doubles, returns False if there are
        """
        array_cur = array_cur[array_cur != 0]
        array_cur = [int(value) for value in array_cur]
        count = np.bincount(array_cur)

        if any(k >= 2 for k in count):  # so there is a double value, we don't want it
            return False
        else:
            return True

    def check_if_sudoku_correct(self):
        '''
            This function checks if the current state of the sudoku is correct or not (so no repeating numbers in rows
            , columns or within each cell).

            Returns True if all is correct, returns False if there are repeats.
        '''

        # turn flat sudoku back to normal size
        sudoku_check = np.resize(self.sudoku_flat, [self.w_sudoku, self.w_sudoku])

        status = True

        if status:  # check all rows
            for i in range(len(sudoku_check)):
                row = sudoku_check[i, :]
                status = self.check_if_array_correct(row)
                if status is False:
                    break

        if status:  # check all columns
            for i in range(len(sudoku_check[0])):
                column = sudoku_check[:, i]
                status = self.check_if_array_correct(column)
                if status is False:
                    break

        if status:  # check all individual cells
            for i in range(0, self.w_cell*self.w_cell, self.w_cell):
                for j in range(0, self.w_cell*self.w_cell, self.w_cell):
                    cell_arr = []
                    for k in range(i, i+self.w_cell):
                        for l in range(j, j+self.w_cell):
                            cell_arr.append(sudoku_check[k][l])
                    cell_arr = np.asarray(cell_arr)
                    status = self.check_if_array_correct(cell_arr)
                    if status is False:
                        break
                if status is False:
                    break

        return status

    def check_if_one_solution(self, idx):

        '''
            This function will check if, for any given grid inside of the sudoku, there is only one possible solution.
            If yes, will return that number. If no, will return 0

            :param idx: The index which represents the grid we wish to check for in self.sudoku_flat
        '''

        all_ind_check = self.all_connected_index[idx]        # get all of the indexes which are connected to this index
                                                             # through either row, column or cell
        all_num = [int(self.sudoku_flat[index]) for index in all_ind_check]
        while 0 in all_num: all_num.remove(0)               # remove all empty elements (represented by 0)
        count = np.bincount(all_num)                        # count how many times each element occurs
        while len(count)<self.w_sudoku+1:
            count = np.append(count, 0)
        indices = [i for i, x in enumerate(count[1:]) if x == 0]    # check how many elements don't occur at all
        if len(indices) == 1:
            return indices[0]+1  # there is a single number possible within the range of possibilities, return this
        else:
            return 0        # there is no single thing

    def solve(self):

        '''
            This script will attempt to solve the sudoku represented by self.sudoku_flat. If too much backpropogation
            is required, this can take a significant time
        :return: a solution for the sudoku, in the same shape as self.sudoku.
        '''

        # Apply conventional solving techniques, so we minimize the amount of unknown numbers for backpropogation
        while True:
            checked = 0
            for k in range(0, len(self.sudoku_flat)):
                if int(self.sudoku_flat[k]) == 0:
                    num = self.check_if_one_solution(k)
                    if num is not 0:
                        self.sudoku_flat[k] = num
                        checked += 1
            if checked == 0:
                break

        self.create_leave_alone()  # re-create the leave alone array, which are filled in grids, which are skipped in backpropogation
        self.sudoku_flat = [int(item) for item in self.sudoku_flat]     # Bugfix : make sure all are int

        # backpropagation : a directed brute force method
        i = 0

        while True:
            # if no solution could be found
            if i < 0:
                print("couldn't find solution, we broke")
                break
            elif i >= len(self.sudoku_flat):
                print("Found a solution")
                break

            # check if current is a leave along
            if i in self.leave_alone:
                i += 1
                continue
            else:   # is not in self.leave_alone, so it is something we can change
                cur_val = self.sudoku_flat[i]

                if cur_val == 0:
                    cur_val = 1
                    self.sudoku_flat[i] = cur_val

                if int(cur_val) > self.w_sudoku:
                    # means value is higher than maximum possible value, so should be reset
                    self.sudoku_flat[i] = 0     # reset value

                    # find the previous i which we can change
                    i_check = i
                    while True:
                        i_check -= 1
                        if i_check not in self.leave_alone:
                            self.sudoku_flat[i_check] = self.sudoku_flat[i_check]+1     # increate element
                            i = i_check
                            break
                else:   # check for status
                    status = self.check_if_sudoku_correct()     # returns true if all is good, returns false if not
                    if status == False:
                        self.sudoku_flat[i] = self.sudoku_flat[i]+1
                    else:  # All is well, so we move on
                        i += 1

        return np.resize(self.sudoku_flat, [self.w_sudoku, self.w_sudoku])


if __name__ == "__main__":

    t1 = time.time()        # for finding out how long it took to find a solution
    print("Start the solver : ")

    # The sudoku we want to solve
    sudoku_to_solve = np.array([['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '5', '', '4', '', '', ''],
                                ['', '7', '', '1', '6', '9', '', '5', ''],
                                ['', '6', '', '', '', '', '', '3', ''],
                                ['', '8', '1', '', '', '', '4', '2', ''],
                                ['', '', '9', '4', '', '8', '5', '', ''],
                                ['9', '', '', '8', '', '5', '', '', '1'],
                                ['', '1', '', '', '', '', '', '7', ''],
                                ['', '5', '', '6', '', '3', '', '4', '']])


    s = SolveSudoku(sudoku_to_solve)        # pass the sudoku on to our class
    solution = s.solve()                    # solve it and get the solution

    print("Time to solve : {} seconds".format(time.time()-t1))
    print("Our solution is : \n"+str(solution))





# Examples of Sudokus we can solve with this solver:

# empty sudoku to fill in
# sudoku_to_solve = np.array([['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', ''],
#                            ['', '', '', '', '', '', '', '', '']])

# easy sudoku
# sudoku_to_solve = np.array([['1', '', '', ''],
#                             ['', '', '2', ''],
#                             ['', '', '', '3'],
#                             ['', '', '', '2'],
#                             ])

# medium sudoku
# sudoku_to_solve = np.array([['', '', '', '2', '6', '', '7', '', '1'],
#                            ['6', '8', '', '', '7', '', '', '9', ''],
#                            ['1', '9', '', '', '', '4', '5', '', ''],
#                            ['8', '2', '', '1', '', '', '', '4', ''],
#                            ['', '', '4', '6', '', '2', '9', '', ''],
#                            ['', '5', '', '', '', '3', '', '2', '8'],
#                            ['', '', '9', '3', '', '', '', '7', '4'],
#                            ['', '4', '', '', '5', '', '', '3', '6'],
#                            ['7', '', '3', '', '1', '8', '', '', '']])

# medium sudoku
# sudoku_to_solve = np.array([['', '', '', '2', '6', '', '7', '', '1'],
#                            ['6', '8', '', '', '7', '', '', '9', ''],
#                            ['1', '9', '', '', '', '4', '5', '', ''],
#                            ['8', '2', '', '1', '', '', '', '4', ''],
#                            ['', '', '4', '6', '', '2', '9', '', ''],
#                            ['', '5', '', '', '', '3', '', '2', '8'],
#                            ['', '', '9', '3', '', '', '', '7', '4'],
#                            ['', '4', '', '', '5', '', '', '3', '6'],
#                            ['7', '', '3', '', '1', '8', '', '', '']])


# hard : around 4 seconds to solve
# sudoku_to_solve = np.array([['', '', '', '', '', '', '', '', ''],
#                             ['', '', '', '5', '', '4', '', '', ''],
#                             ['', '7', '', '1', '6', '9', '', '5', ''],
#                             ['', '6', '', '', '', '', '', '3', ''],
#                             ['', '8', '1', '', '', '', '4', '2', ''],
#                             ['', '', '9', '4', '', '8', '5', '', ''],
#                             ['9', '', '', '8', '', '5', '', '', '1'],
#                             ['', '1', '', '', '', '', '', '7', ''],
#                             ['', '5', '', '6', '', '3', '', '4', '']])
