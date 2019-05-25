import numpy as np
import queue
towalk=1000000
deque=queue.Queue()

maze=np.array([[1,2,1,1,1,1,1,1,0,1],#0 means channel，1 means wall，2 means entry，3 means exit
              [0,0,0,0,0,0,1,0,0,1],
              [0,1,0,1,1,0,1,1,0,1],
              [0,1,0,0,0,0,0,0,0,0],
              [1,1,0,1,1,0,1,1,1,1],
              [0,0,0,0,1,0,0,0,0,1],
              [0,1,1,1,1,1,1,1,0,1],
              [0,0,0,0,1,0,0,0,0,0],
              [0,1,1,1,1,0,1,1,1,0],
              [0,0,0,0,1,0,0,0,3,1]])
d=maze
n,m=maze.shape
sx,sy=0,1#index of entry
ex,ey=9,8#index of exit

dx=[1,0,-1,0]
dy=[0,1,0,-1]

def bfs():
    for i in range(n):
        for j in range(m):
            d[i][j]=towalk
    cur_list=[sx,sy]
    d[sx][sy]=0
    deque.put(cur_list)
    while deque:
        walked=deque.get()
        if (walked[0]==ex and walked[1])==ey:
            break
        for i in range(4):
            nx=walked[0]+dx[i]
            ny=walked[1]+dy[i]
            if (nx>=0 and nx<n and ny>=0 and ny<m and maze[nx][ny]!=1 and d[nx][ny]==towalk):
                cur_list=[nx,ny]
                deque.put(cur_list)
                d[nx][ny]=d[walked[0]][walked[1]]+1
    print(d)
    return d[ex][ey]
def test():
    result=bfs()
    print(result)
test()