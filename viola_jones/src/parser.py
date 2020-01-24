inputfile = open('/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/data/logs.txt')
f = open('/Users/lorenzocioni/Documents/Sviluppo/Workspace/AdaBoost/data/out.txt', 'w')

for line in inputfile:
    if line.startswith( 'Trained WeakClassifier:' ):
        line = line.replace('Trained WeakClassifier:', '')
        line = line.strip()
        line = line.split(', ')
        f.write('c:')

        elements = line[2].split(' ')
        f.write(elements[1] + ',')
        elements = line[1].split(' ')
        f.write(elements[1] + ',')
        elements = line[4].split(' ')
        f.write(elements[1] + ',')
        elements = line[0].split(' ')
        f.write(elements[1] + ',')
        elements = line[5].split(' ')
        f.write(elements[1] + ',')
        elements = line[3].split(' ')
        f.write(elements[1] + '\n')

f.close()
