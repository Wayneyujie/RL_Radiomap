def transform_data(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    transformed_lines = []
    for line in lines:
        parts = line.strip().split()
        x, y = map(float, parts[:2])
        x1 = -136.5 + 10 * x
        y1 = -38 + 10 * y
        transformed_lines.append(f"{x1:.3f} {y1:.3f} {parts[2]}\n")  # 将第三列数据直接添加到新文件中

    with open(output_file, 'w') as f:
        f.writelines(transformed_lines)

if __name__ == "__main__":
    # import sys
    # foldername = sys.argv[1]
    # input_file = "path_point.txt"  # 输入文件名
    #input_file = foldername
    input_file = "./results/traj.txt"  # 输入文件名
    output_file = "./carla/carla_traj.txt"  # 输出文件名
    transform_data(input_file, output_file)
