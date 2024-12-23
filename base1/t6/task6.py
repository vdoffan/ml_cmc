def check(x: str, file: str):
    x = x.lower()
    my_dict = {}
    string = x.split(' ')

    for item in string:
        if item in my_dict.keys():
            my_dict[item] += 1
        else:
            my_dict[item] = 1
    
    my_dict = dict(sorted(my_dict.items(), key=lambda item: item[0]))

    with open(file, 'w') as f:
        for item in my_dict.keys():
            f.write(f'{item} {my_dict[item]}\n')
