
Fidelity_list = """your output"""
epoch = 1
with open('output.txt', 'a+') as file:
    file.write('the epoch is: ' + str(epoch) + '\n')
    for fidelity in Fidelity_list:
        file.write(str(fidelity) + '\n')
epoch += 1
