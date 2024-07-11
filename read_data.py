def read():
    names = []
    non_names = []
    with open('data/names.txt','r') as name_file, open('data/non_names.txt','r') as non_name_file:
        for line in name_file:
            names.append(line.strip())
        for line in non_name_file:
            non_names.append(line.strip())

    print(len(names),len(non_names))
    return names,non_names


