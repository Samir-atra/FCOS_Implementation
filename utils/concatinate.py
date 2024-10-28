def concat(list1, list2):
    if type(list1[0]) == list and type(list2) == list:
        alist = list1
        alist.append(list2)
    elif type(list2) == int:
        alist = [list1]
        alist.append(list2)
    return alist

# check if need to put all the boxes of a single image in a list instead of keeping them separate.
# concat([1.08, 187.69, 611.59, 285.84],51)