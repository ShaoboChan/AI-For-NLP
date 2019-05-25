from collections import deque
sentence='I am happy and sad'
words=sentence.split()
dic=[["happy", "glad"], ["glad", "good"],["sad", "sorrow"]]
d=[]
been=[]
for i in range(len(dic)):
    if dic[i][0] in words:
        d.append(dic[i])
        been.append(i)
while len(been)!=len(dic):
    for i in range(len(dic)):
        if i not in been:
            for j in range(len(d)):
                if d[j][-1]==dic[i][0]:
                    d[j].append(dic[i][1])
                    been.append(i)
print(d)
for i in range(len(d[0])):
    for j in range(len(d[1])):
        print(words[0]+' '+words[1]+' '+d[0][i]+' '+words[3]+' '+d[1][j])




