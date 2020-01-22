def iou(box_max,box2):
    x1 = max(box_max[0], box2[0])
    x2 = min(box_max[1], box2[1])
    y1 = max(box_max[2], box2[2])
    y2 = min(box_max[3], box2[3])
    w = max(0.0, x2 - x1 + 1)
    h = max(0.0, y2 - y1 + 1)
    inter=w*h
    return inter/(area(box_max)+area(box2)-inter)
def area(box):

    return ((box[1]-box[0]+1)*(box[3]-box[2]+1))
def NMS(dets,thre):
    #dets:list of lists;list[i]=[x0,x1,y0,y1,score]
    D=[]
    while dets:
        box_max=max(dets,key=lambda x:x[4])
        D.append(box_max)
        dets.remove(box_max)
        for i in dets:
            if iou(box_max,i)>=thre:
                dets.remove(i)
    return D

if __name__ == '__main__':

    dets = [[10, 30, 11, 30, 0.95],
             [5, 9, 5, 9, 0.5],
             [5,22, 3, 15, 0.45],
             [11,27, 11, 29, 0.7],
             [4, 16, 4, 16, 0.78]]
    thre = 0.4

    result = NMS(dets, thre)
    print(result)
