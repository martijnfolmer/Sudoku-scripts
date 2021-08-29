import time
import warnings
import numpy as np
import random
import collections
import copy
import cv2
warnings.simplefilter(action='ignore', category=FutureWarning)


class CreateSudoku:

    def __init__(self, difficulty):
        print("We are creating a sudoku!")
        self.sudoku_array = [[-1]*9]*9
        self.n_to_remove = int(difficulty*81)        # how many blocks to remove
        self.n_removed = 0
        print(f"We will remove {self.n_to_remove} numbers from the sudoku")

    def create_sudoku(self):
        # This function will create our sudoku, by first getting a filled in sudoku, then removing random numbers whilst
        #checking whether it can still be solved
        self.create_filled_in()

        # start removing elements, which will allow us to
        num_removed = 0
        for n in range(self.n_to_remove):
            kn = 0
            while True:
                xid, yid = self.get_non_empty(self.sudoku_array)
                valcur = self.sudoku_array[xid][yid]
                self.sudoku_array[xid][yid] = -1
                check = self.solve(self.sudoku_array)
                if check[0] == True:        # we were able to solve, so we continue
                    num_removed +=1
                    break
                else:
                    self.sudoku_array[xid][yid] = valcur    # reset
                    kn += 1
                    if kn>200:
                        break
        print(f"Total number of blocks removed : {num_removed}")
        self.n_removed = num_removed

    ############################################################
    # Functions for the creation of the initial filled in sudoku
    ############################################################
    def get_non_empty(self, sudoku_arr):

        all_vals_to_solve = [vals for vals in zip(*np.where(np.array(sudoku_arr) != -1))]
        if len(all_vals_to_solve)==0:
            print("ERROR : too many empty spaces in sudoku already")

        val = random.choice(all_vals_to_solve)
        return val[0], val[1]

    def shift(self, row, shift_n):

        '''
        :param row: The array which represents the row of which we want to shift the values
        :param shift_n: How much to shift the values
        :return: the array which is the shifted row to fill into the sudoku
        '''
        row = collections.deque(row)
        row.rotate(shift_n)
        row = list(row)
        return row

    def shuffle_rows (self, row_min, row_max):

        shuffle_index = list(range(row_min, row_max+1))
        random.shuffle(shuffle_index)
        sudoku_copy = copy.deepcopy(self.sudoku_array)
        for i in range(3):
            sudoku_copy[row_min+i] = self.sudoku_array[shuffle_index[i]]
        self.sudoku_array = copy.deepcopy(sudoku_copy)

    def shuffle_columns(self, column_min, column_max):
        shuffle_index = list(range(column_min, column_max+1))
        random.shuffle(shuffle_index)
        sudoku_copy = copy.deepcopy(self.sudoku_array)
        for j in range(3):
            for i in range(9):
                sudoku_copy[i][column_min+j] = self.sudoku_array[i][shuffle_index[j]]
        self.sudoku_array = copy.deepcopy(sudoku_copy)

    def create_filled_in(self):
        print("Creating the initial filled in sudoku")

        # get the top row, randomly arranged unique numbers between 1-9
        toprow = list(range(1,10))
        random.shuffle(toprow)
        self.sudoku_array[0] = [val for val in toprow]

        # shift the values alternatingly by 3,3,1, and fill in the rest of the sudoku
        for i_s, shift_c in enumerate([3,3,1,3,3,1,3,3]):
            toprow = self.shift(toprow, shift_c)
            self.sudoku_array[i_s+1] = [val for val in toprow]

        self.shuffle_rows(0, 2)
        self.shuffle_rows(3, 5)
        self.shuffle_rows(6, 8)
        self.shuffle_columns(0, 2)
        self.shuffle_columns(3, 5)
        self.shuffle_columns(6, 8)

        # we can flip horizontally and vertically at random
        if random.random()<=0.5:
            np.flip(self.sudoku_array, axis=0)
        if random.random()<=0.5:
            np.flip(self.sudoku_array, axis=1)
        # randomly rotate it 90 degrees
        for i in range(random.randint(0, 3)):
            np.rot90(self.sudoku_array)

    ############################################################
    # functions for solving the sudoku
    ############################################################
    def check_missing_num(self, arr):

        # check if there is only a single possible value missing in the passed array
        # the array represents a row, a column or a block
        there = [0]*9
        for val in arr:
            there[val-1] += 1
        if np.bincount(there)[0] == 1:
            return there.index(0)+1
        else:
            return -1

    def check_if_single_solution(self, sudoku_arr, idx, idy):

        # this function checks whether there is only one solution for a particular array (so only 1 missing number in
        # a row, column or block at location (idx, idy))
        all_column_val = [coor[idy] for coor in sudoku_arr]     # the column it is in
        all_column_val = [coor for coor in all_column_val if coor > 0]
        val = self.check_missing_num(all_column_val)
        if val != -1:
            return val

        all_row_val = sudoku_arr[idx]                           # the row it is in
        all_row_val = [coor for coor in all_row_val if coor > 0]
        val = self.check_missing_num(all_row_val)
        if val != -1:
            return val

        sudoku_check = np.array(sudoku_arr)

        all_block_val = sudoku_check[(idx//3)*3:(idx//3)*3+3, (idy//3)*3:(idy//3)*3+3]
        all_block_val = list(np.reshape(all_block_val, 9))
        all_block_val = [coor for coor in all_block_val if coor>0]
        val = self.check_missing_num(all_block_val)
        if val !=-1:
            return val

        coll_and_row_val = all_column_val
        coll_and_row_val.extend(all_row_val)
        coll_and_row_val.extend(all_block_val)
        coll_and_row_val = list(set(coll_and_row_val))
        coll_and_row_val = [coor for coor in coll_and_row_val if coor>0]
        val = self.check_missing_num(coll_and_row_val)
        return val

    def solve(self, sudoku_arr):
        tosolve = copy.deepcopy(sudoku_arr)     # copy it, so we can pass the solution on


        # first attempt to solve using traditional methods, which minimizes the values needed for backpropagation
        while True:
            all_vals_to_solve = [vals for vals in zip(*np.where(np.array(tosolve) == -1))]
            num_solved = 0
            for vals in list(all_vals_to_solve):
                val_found = self.check_if_single_solution(tosolve, vals[0], vals[1])

                if val_found != -1:
                    tosolve[vals[0]][vals[1]] = val_found
                    num_solved += 1

            if num_solved == 0:
                break

        # use backpropagation in order to solve
        all_vals_to_backpropogate = [vals for vals in zip(*np.where(np.array(tosolve) == -1))]
        kn = 0
        while True:
            if kn<0:    # means we didn't find a solution
                break
            if kn>=len(all_vals_to_backpropogate):  # means we found a solution
                break

            id_to_check = all_vals_to_backpropogate[kn]
            val = tosolve[id_to_check[0]][id_to_check[1]]

            if val == -1:
                tosolve[id_to_check[0]][id_to_check[1]] = 1
            else:
                tosolve[id_to_check[0]][id_to_check[1]] = val + 1

            if tosolve[id_to_check[0]][id_to_check[1]] >9:      # means we can't find a solution, so we backpropagate
                tosolve[id_to_check[0]][id_to_check[1]] = -1
                kn-=1
            else:
                check = self.check_solution(tosolve)
                if check[0]:       # move onto next
                    kn+=1

        if kn<0 or kn>len(all_vals_to_backpropogate):
            return [False, tosolve]
        else:
            return [True, tosolve]

    ############################################################
    # functions check the validity of the current solution
    ############################################################
    def check_only_unique(self, arr):
        arr_to_check = [coor for coor in arr if coor > 0]
        if len(set(arr_to_check)) == len(arr_to_check):
            return True
        else:
            return False

    def check_column(self, sudoku_arr, column_index):
        arr_to_check = [coor[column_index] for coor in sudoku_arr]
        return [self.check_only_unique(arr_to_check), arr_to_check]

    def check_row(self, sudoku_arr, row_index):
        arr_to_check = sudoku_arr[row_index]
        return [self.check_only_unique(arr_to_check), arr_to_check]

    def check_block(self, sudoku_arr, row1, row2, column1, column2):
        arr_to_check = np.array(sudoku_arr)[row1:row2+1,column1:column2+1]
        arr_to_check = list(np.reshape(arr_to_check, (9)))

        return [self.check_only_unique(arr_to_check), arr_to_check]

    def check_solution(self, sudoku_arr):

        '''

        :return: true if solution is valid (solution is considered valid if none of the filled in values break a rule), or false if invalid
        '''

        # check rows
        for icheck in range(len(sudoku_arr)):
            correct = self.check_row(sudoku_arr, icheck)
            if not correct[0]:
                return [False, "row", icheck, correct[1]]

        # check columns
        for icheck in range(len(sudoku_arr)):
            correct = self.check_column(sudoku_arr, icheck)
            if not correct[0]:
                return [False, "column", icheck, correct[1]]

        # check all the 3x3 blocks
        for x in range(0, 3):
            for y in range(0, 3):
                correct = self.check_block(sudoku_arr, x*3,x*3+2,y*3,y*3+2)
                if not correct[0]:
                    return [False, 'block', [x,y], correct[1]]

        # If we got to this point, the solution is valid
        return [True, "", 0, 0]

    ############################################################
    # visualisation functions
    ############################################################
    def plot_our_sudoku_num(self, img, sudoku_arr):

        # This function will plot the numbers of the sudoku in the appropriate window
        for i in range(len(sudoku_arr)):
            for j in range(len(sudoku_arr[0])):
                if sudoku_arr[i][j]!=-1:
                    cv2.putText(img, str(sudoku_arr[i][j]), (25+i*100,75+j*100), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,0), thickness=3)
        return img

    def show_our_sudoku(self):

        # this function will display the sudoku and its solution in sepperate windows
        rows = 900
        cols = 900
        sudoku_show = np.ones((rows, cols, 3), dtype="uint8")*255  # 3channel

        # draw our lines
        cv2.rectangle(sudoku_show, (0,0), (rows,cols),(0,0,0), thickness=10)
        for i in range(3):
            for j in range(3):
                cv2.rectangle(sudoku_show, (i*300, j*300), ((i+1)*300, (j+1)*300), (0, 0, 0), thickness=10)
        for i in range(9):
            for j in range(9):
                cv2.rectangle(sudoku_show, (i*100, j*100), ((i+1)*100, (j+1)*100), (0, 0, 0), thickness=5)

        sudoku_show_solution = copy.deepcopy(sudoku_show)
        sudoku_sol = self.solve(self.sudoku_array)

        # plot the values
        self.plot_our_sudoku_num(sudoku_show, self.sudoku_array)
        self.plot_our_sudoku_num(sudoku_show_solution, sudoku_sol[1])

        sudoku_show = cv2.resize(sudoku_show, (500, 500))
        sudoku_show_sol = cv2.resize(sudoku_show_solution, (500,500))

        cv2.namedWindow('Generated Sudoku')  # Create a named window
        cv2.moveWindow('Generated Sudoku', 40, 30)
        cv2.imshow('Generated Sudoku', sudoku_show)

        cv2.namedWindow('Solution to Sudoku')
        cv2.moveWindow('Solution to Sudoku', 600, 30)
        cv2.imshow('Solution to Sudoku', sudoku_show_sol)
        cv2.waitKey(-1)


if __name__ == "__main__":
    t1 = time.time()
    s = CreateSudoku(0.6)           # percentage of the spaces we want to be empty
    n_attempts = 0
    while True:
        s.create_sudoku()
        check = s.solve(s.sudoku_array)      # Checking, make sure that the created sudoku is actually solvable

        if s.n_removed < 0.9*s.n_to_remove:   # checking whether we are actually removing enough
            print(f"ERROR : not enough blocks where created, we must try again. Total number of attempts :{n_attempts+1}")
        elif check[0]:  # means we were able to create a sudoku, which our solver was able to solve
            print(f"Succes! Creation took : {time.time()-t1}. Total number of attempts : {n_attempts+1}")
            break
        else:
            # This particular error message should never occur, but just in case we created a sudoku that can't be
            # solved by backpropagation
            print(f"ERROR : sudoku was not able to be created, trying again. Total number of attempts : {n_attempts+1}")
        n_attempts +=1

    print("Below is the array form of our sudoku : -1 is an empty space")
    for line in s.sudoku_array:
        print(line)

    s.show_our_sudoku()



