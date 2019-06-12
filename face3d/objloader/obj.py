class objloader(object):
    vertSize = 0
    textSize = 0
    def __init__(self, objFilePath):
        self.vertComIndex = []
        self.vertData = [] 
        temp_vertices = []
        self.vertices = []  
        temp_textCoords = []
        temp_normals = []
        with open(objFilePath) as objFile:
            for line in objFile.readlines():
                data = line.split()
                if(data[0] == 'vt'):
                    text = [float(data[1]), -float(data[2])]
                    temp_textCoords.append(text)
                elif(data[0] == 'vn'):
                    norm = [float(data[1]), float(data[2]), float(data[3])]
                    temp_normals.append(norm)
                elif(data[0] == 'v'):
                    pos = [float(data[1]), float(data[2]), float(data[3])]
                    temp_vertices.append(pos)
                    self.vertices.append(float(data[1]))
                  
                    self.vertices.append(float(data[2]))
                    
                    self.vertices.append(float(data[3]))
                elif(data[0] == 'f'):
                    for vert in data[1:]:
                        pos = vert.split('/')
                        comIndex = [int(pos[0])-1, int(pos[1])-1, -1] 
                        self.vertComIndex.append(comIndex)
        self.vertSize = len(temp_vertices)
        self.textSize = len(temp_textCoords)
        for index in self.vertComIndex:
            vert = [temp_vertices[index[0]] if index[0] >= 0 else None, \
                          temp_textCoords[index[1]] if index[1] >= 0 else None, \
                          temp_normals[index[2]] if index[2] >= 0 else None]
            self.vertData.append(vert)
            
    def save(self, filename):
        temp_vertices = [None] * self.vertSize
        temp_textCoords = [None] * self.textSize
        for x, y in zip(self.vertComIndex, self.vertData):
            # print(type(y.position))
            if (x[0]< 0 or x[0] >= self.vertSize):
                print("pos index",x[0] )
            if (x[1] < 0 or x[1] >= self.textSize):
                print("textCoordIndex",x[1] )
            temp_vertices[x[0]] = y[0]
            temp_textCoords[x[1]] = y[1]
            
        with open(filename,'w') as outfile:
#            for vert in temp_vertices:   #
                # print(type(vert))
#                 line = 'v ' + str(vert[0]) + ' ' + str(vert[1]) + ' ' + str(vert[2]) + '\n'
#                 outfile.write(line)
            for i in range(self.vertSize):
                line = 'v ' + str(self.vertices[i * 3]) + ' ' + str(self.vertices[i * 3 + 1]) + ' ' + str(self.vertices[i * 3 + 2]) + '\n'
                outfile.write(line)
            for text in temp_textCoords:
                line = 'vt ' + str(text[0]) + ' ' + str(text[1]) + '\n'
                outfile.write(line)
            count = 0
            for index in self.vertComIndex:
                # print(index.posIndex, index.textCoordIndex)
                if (count % 4 == 0):
                    line = 'f ';
                line += str(index[0] + 1) + '/' + str(index[1] + 1) + '/'\
                + str(index[2] + 1) + ' '
                if(count % 4 == 3):
                    line += '\n'
                    outfile.write(line)
                count += 1
