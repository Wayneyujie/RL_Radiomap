with open("coordinates5.txt", "r") as file:
    for line in file:
        # 每行读取x, y
        x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
        

        x1=(x -555.375)/2.165
        y1=(y-202.703)/1.984

        x1=(x -555.375)/2.165
        y1=(y-202.703)/1.984

        radiox=(x1+600)/647*66
        radioy=(y1+34.4)/251.5*26
        cellindex_x=int(radiox)
        cellindex_y=int(radioy)
        x= cellindex_x
        y= cellindex_y
        
        

        
        # 打印转换后的结果
        print(f"x1: {cellindex_x}, y1: {cellindex_y}")