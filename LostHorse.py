import numpy as np
import math
import time
# ---------------------------------------------------------------------------------------------#
# Reads file with board and tranform to matrix
def boardToMatrix( fileName ):
    rowNum = 0
    colNum = 0
    with open( fileName, 'r') as fp :
        rowNum = len(fp.readlines())
        fp.seek(0, 0)
        colNum = len(fp.readline().rstrip())
        print(rowNum, colNum)
        board = np.zeros((rowNum,colNum))
        linec = 0
        fp.seek(0, 0)
        lines = fp.readlines()
        for line in range(len(lines)):
            lin = lines[line].rstrip()
            squares = list(lin)
            for square in range(len(squares)):
                if squares[square] == "x":
                    board[line][square] = 1
                elif squares[square]  == "C":
                    start = (line,square)
                    board[line][square] = 4
                elif squares[square]  == "S":
                    end = (line,square)
                    board[line][square] = 5
            linec += 1
        
        return start, end, board, rowNum, colNum
# ---------------------------------------------------------------------------------------------#
# Build a graph from the board
def buildBoardGraph( board ):
    nodos = set()
    arestas = {}

    limitRow , limitCol = board.shape

    for row in range(len(board)):
        for col in range(len(board[row])):
            lms = legalMoves(board, row, col, limitRow-1, limitCol-1)
            for lm in range(len(lms)):
                if lms[lm][0] != -1:
                    nodos.add("N" + str(row) + "_" + str(col) )
                    nodos.add("N" + str(lms[lm][0]).replace('.0','') + "_" + str(lms[lm][1]).replace('.0','')) 
                    arestas[ (("N" + str(row) + "_" + str(col)),("N" + str(lms[lm][0]).replace('.0','') + "_" + str(lms[lm][1]).replace('.0',''))) ] = 1

    return nodos, arestas

# ---------------------------------------------------------------------------------------------#
# Return de possible legal moves from one spot (considering infinite board)
def legalMoves(board , row, col, limitRow, limitCol):
    lm = np.zeros((8,2))
    upRigth = [row-2,col+1]
    upLeft  = [row-2,col-1]
    leftUp = [row-1,col-2]
    leftDown = [row+1,col-2]
    downLeft = [row+2,col-1]
    downRight = [row+2,col+1]
    rightDown = [row+1,col+2]
    rigthUp = [row-1,col+2]

    # Checa upRight
    if upRigth[0] < 0:
        upRigth[0] = (limitRow + 1) - abs(upRigth[0])
    if upRigth[1] > limitCol :
        upRigth[1] = 0 
    if board[upRigth[0]][upRigth[1]] != 1.0 :
        lm[0] = upRigth; 
    else:
        lm[0] = [-1,-1]
    
    # Checa upLeft
    if upLeft[0] < 0 :
        upLeft[0] = (limitRow + 1) - abs(upLeft[0])
    if upLeft[1] < 0 :
        upLeft[1] = limitCol
    if board[upLeft[0]][upLeft[1]] != 1.0 :
        lm[1] = upLeft
    else :
        lm[1] = [-1,-1]
    
    # Checa leftUp
    if leftUp[0] < 0 :
        leftUp[0] = limitRow 
    if leftUp[1] < 0 :
        leftUp[1] = (limitCol + 1) - abs(leftUp[1])
    if board[leftUp[0]][leftUp[1]] != 1.0 :
        lm[2] = leftUp
    else :
        lm[2] = [-1,-1]

    # Checa leftDown
    if leftDown[0] > limitRow :
        leftDown[0] = 0 
    if leftDown[1] < 0 :
        leftDown[1] = (limitCol + 1) - abs(leftDown[1])
    if board[leftDown[0]][leftDown[1]] != 1.0 :
        lm[3] = leftDown
    else :
        lm[3] = [-1,-1]

    # Checa downLeft
    if downLeft[0] > limitRow :
        downLeft[0] = downLeft[0] - (limitRow + 1) 
    if downLeft[1] < 0 :
        downLeft[1] = limitCol 
    if board[downLeft[0]][downLeft[1]] != 1.0 :
        lm[4] = downLeft
    else :
        lm[4] = [-1,-1]

    # Checa downRight
    if downRight[0] > limitRow :
        downRight[0] = downRight[0] - (limitRow + 1)
    if downRight[1] > limitCol :
        downRight[1] = 0
    if board[downRight[0]][downRight[1]] != 1.0 :
        lm[5] = downRight
    else :
        lm[5] = [-1,-1]

    # Checa rightDown
    if rightDown[0] > limitRow :
        rightDown[0] = 0 
    if rightDown[1] > limitCol :
        rightDown[1] =  rightDown[1] - (limitCol + 1)
    if board[rightDown[0]][rightDown[1]] != 1.0 :
        lm[6] = rightDown 
    else :
        lm[6] = [-1,-1]

    # Checa rigthUp
    if rigthUp[0] < 0 :
        rigthUp[0] = limitRow
    if rigthUp[1] > limitCol :
        rigthUp[1] = rigthUp[1] - (limitCol + 1)
    if board[rigthUp[0]][rigthUp[1]] != 1.0 :
        lm[7] = rigthUp 
    else :
        lm[7] = [-1,-1]

    return lm
# ---------------------------------------------------------------------------------------------#
# Read graph from file (example)
def readG( name ) : 
  nodos = set()
  arestas = {}

  with open(name, 'r') as fp :
    for lin in fp :
      lin = lin.rstrip()
      words = lin.split()
      nodos.add( words[0] )
      nodos.add( words[2] )
      arestas[ ( words[0], words[2] ) ] = int(words[3])
      arestas[ ( words[2], words[0] ) ] = int(words[3])

  return nodos, arestas
# ---------------------------------------------------------------------------------------------#
# Breadth First Search Solution
def findNeighbors(node,E):
    neighbours = []
    for e in E.keys():
        n1, n2 = e
        if n1 == node:
            neighbours.append(n2)
    return neighbours

def breadthFirstSearch(start,E):
    q = []
    q.append(start)

    visited = {}
    visited[start] = True

    prev = {}
    while len(q) > 0:
        node = q.pop(0)
        neighbors = findNeighbors(node,E)
        for neighbour in neighbors:
            if neighbour not in visited:
                q.append(neighbour)
                visited[neighbour] = True
                prev[neighbour] = node
    return prev

def reconstructPath(start, end, prev):
    path = []

    node = end
    while(node in prev or node == start):
        path.append(node)
        if node == start:
            break
        node = prev[node]

    path.reverse()

    print(path)
    if path[0] == start:
        return path
    else :
        return []

def BFS(start, end, E):
    rowS, colS = start
    rowE, colE = end
    s = "N"+ str(rowS) + "_" + str(colS)
    e = "N"+ str(rowE) + "_" + str(colE)

    prev = breadthFirstSearch(s,E)

    path = reconstructPath(s, e, prev)
    return len(path)-1

# ---------------------------------------------------------------------------------------------#
# dijkstra Algorithym

def vertexMinDist(Q, dist):
    min = None
    for n in Q:
        if min not in dist:
            min = n
        if dist[n] < dist[min]:
            min = n
    return min

def Dijkstra( start, end, V, E ) :

    Q = set()

    dist = {}
    prev = {}

    for v in V:
        dist[v] = math.inf
        prev[v] = None
        Q.add(v)

    dist[start] = 0

    while len(Q) != 0 :
        u = vertexMinDist(Q, dist)

        Q.remove(u)

        if u == end:
            break

        neighbours = findNeighbors(u,E)
        for n in neighbours:
            alt = dist[u] + 1
            if alt < dist[n]:
                dist[n] = alt
                prev[n] = u

    return dist, prev

def buildPath(start, end, prev):
    path = []
    u = end
    if prev[u] != None or u == start:
        while u != None:
            path.append(u)
            u = prev[u]
    path.reverse()
    return path

def minDijkstra(start, end , V, E):
    rowS, colS = start
    rowE, colE = end
    s = "N"+ str(rowS) + "_" + str(colS)
    e = "N"+ str(rowE) + "_" + str(colE)

    dist , prev = Dijkstra(s, e, V, E)
    path = buildPath(s, e, prev)
    print(path)
    return len(path)-1


# ---------------------------------------------------------------------------------------------#
# aStar algorithym

def aStarMin(start, end, rowN, colN, V, E):
    rowS, colS = start
    rowE, colE = end
    s = "N"+ str(rowS) + "_" + str(colS)
    e = "N"+ str(rowE) + "_" + str(colE)

    #prev = aStarNotAcuratte(s, e, V, E)
    prev = aStar(s, e, rowN, colN, V, E)
    path = buildPath(s, e, prev)
    #path = reconstruct_path(prev, e)
    print(path)
    return len(path)-1


def aStarNotAcuratte(start, end, V, E):

    open = set()
    open.add(start)

    prev = {}

    gscore = {}
    fscore = {}
    for v in V:
        gscore[v] = math.inf
        fscore[v] = math.inf
        prev[v] = None
    
    gscore[start] = 0
    fscore[start] = h(start, end)
    
    while len(open) != 0:
        u = getMinFscore(open,fscore)
        if u == end:
            return prev
        
        open.remove(u)
        neighbours = findNeighbors(u, E)
        for n in neighbours:
            tentativeGscore = gscore[u] + 1
            if tentativeGscore < gscore[n]:
                prev[n] = u
                gscore[n] = tentativeGscore
                fscore[n] = gscore[n] + h(n, end)
                if n not in open:
                    open.add(n)
    
    return {}

def aStar(start, end, rowN, colN, V, E):

    open = set()
    open.add(start)

    prev = {}

    gscore = {}
    fscore = {}
    for v in V:
        gscore[v] = math.inf
        fscore[v] = math.inf
        prev[v] = None
    
    gscore[start] = 0
    fscore[start] = hInf(start, end, rowN, colN)
    
    while len(open) != 0:
        u = getMinFscore(open,fscore)
        if u == end:
            return prev
        
        open.remove(u)
        neighbours = findNeighbors(u, E)
        for n in neighbours:
            tentativeGscore = gscore[u] + hInf(u, n, rowN, colN)
            if tentativeGscore < gscore[n]:
                prev[n] = u
                gscore[n] = tentativeGscore
                fscore[n] = gscore[n] + hInf(n, end, rowN, colN)
                if n not in open:
                    open.add(n)
    
    return {}

def getMinFscore(open,fscore):
    min = None
    for n in open:
        if min not in open:
            min = n
        if fscore[n] < fscore[min]:
            min = n
    return min

def h(node, end):
    node = node.replace("N","").replace("_", " ").split()
    end = end.replace("N","").replace("_", " ").split()
    rowN = int(node[0])
    colN = int(node[1])
    rowE = int(end[0])
    colE = int(end[1])
    return math.dist([colN, rowN], [colE, rowE])

def hInf(node, end, rowN, colN):
    n, e = translatePointsCenter(node , end, rowN, colN)
    return math.dist(n, e)

def translatePointsCenter(toCenter , target, rowN, colN):
    center = [(colN/2.0)-1, (rowN/2.0)-1]
    toCenter = toCenter.replace("N","").replace("_", " ").split()
    target = target.replace("N","").replace("_", " ").split()
    rowTC = int(toCenter[0])
    colTC = int(toCenter[1])
    rowT = int(target[0])
    colT = int(target[1])

    coldif = center[0] - colTC
    rowdif = center[1] - rowTC
    
    colTT = colT + coldif
    rowTT = rowT + rowdif

    if colTT >= colN:
        colTT = colTT - colN

    if rowTT >= rowN:
        rowTT = rowTT - rowN
    
    if colTT < 0:
        colTT = colTT + colN

    if rowTT < 0:
        rowTT = rowTT + rowN

    targetTranslated= [colTT,rowTT]

    return center, targetTranslated

def reconstruct_path(prev, end):
    n = end
    total_path = []
    total_path.append(n)
    while n in prev.keys():
        n = prev[n]
        total_path.append(n)
    total_path.reverse()
    return total_path

# ---------------------------------------------------------------------------------------------#
# main executioon

start_time = time.time()

start, end, board, rowNum, colNum = boardToMatrix("casos/caso100.txt")

V, E = buildBoardGraph(board)

#print(V, "\n\n", E)

print(len(V), len(E))

#print(BFS(start, end, E))

# dist , prev = Dijkstra("N4_10" , "N14_34", V, E)

# print(buildPath("N4_10","N14_34", prev))

# print(minDijkstra(start, end, V, E))

#print (aStarMin(start, end, V, E))

print (aStarMin(start, end, rowNum, colNum, V, E))

print("--- %s seconds ---" % (time.time() - start_time))