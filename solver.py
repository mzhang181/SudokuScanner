from typing import List

def backTrack(grid: List[List[int]], row = 0, col = 0) -> bool:
    def can_place(grid: List[List[int]], guess: int, row: int, col: int) -> bool:
        if guess in grid[row]:
            return False
        for i in range(9):
            if grid[i][col] == guess:
                return False
        region = ((row // 3) * 3, (col // 3) * 3)
        for y in range(3):
            for x in range(3):
                if grid[region[0] + y][region[1] + x] == guess:
                    return False
        return True
    
    if col == 9:
        if row == 8:
            return True
        row += 1
        col = 0
    
    if grid[row][col]:
        return backTrack(grid, row, col + 1)

    for i in range(1, 10):
        if can_place(grid, i, row, col):
            grid[row][col] = i
            if backTrack(grid, row, col + 1):
                return True
            grid[row][col] = 0
        
    return False