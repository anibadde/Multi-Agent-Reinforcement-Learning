def minRooms(intervals):
    times = []
    for elem in intervals:
        times.append((elems[0], 1))
        times.append((elems[1], -1))
    times.sort()
    counter = 0
    maximum = 0
    for i in range(len(times)):
        counter += times[i][1]
        if counter > maximum:
            maximum = counter
    return maximum


def join(mapA, mapB):
    mapC = {}
    for (idVal, name) in mapA:
        if idVal in mapB:
            mapC[idVal] = (name, mapB[idVal])
        else:
            mapC[idVal] = (name, None)
    return mapC

