with open('environment.yml', 'r') as file:
    lines = file.readlines()

with open('environment_clean.yml', 'w') as file:
    for line in lines:
        equal_index = line.find('=')
        if equal_index != -1:
            modified_line = line[:equal_index]
            file.write(modified_line + '\n')
        else:
            file.write(line)