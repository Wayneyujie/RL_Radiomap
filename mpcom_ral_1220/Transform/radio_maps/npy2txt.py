import numpy as np

def npy3d_to_txt_2d(npy_file_path, txt_file_path, delimiter=' ', fmt='%.6f'):
    """
    将3D的.npy文件转换为2D的.txt文件
    
    参数:
    npy_file_path: .npy文件路径
    txt_file_path: 输出的.txt文件路径
    delimiter: 分隔符，默认为空格
    fmt: 数据格式，默认为保留6位小数
    """
    try:
        # 加载.npy文件
        data = np.load(npy_file_path)
        
        print(f"原始数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        
        # 处理3D数组
        if data.ndim == 3:
            # 如果第一个维度是1，直接压缩成2D
            if data.shape[0] == 1:
                data_2d = data[0, :, :]  # 或者使用 data.squeeze()
                print(f"压缩后的2D形状: {data_2d.shape}")
            else:
                # 如果第一个维度不是1，让用户选择切片
                print("警告：第一个维度不是1，将使用第一个切片")
                data_2d = data[0, :, :]
            
            # 保存为.txt文件
            np.savetxt(txt_file_path, data_2d, delimiter=delimiter, fmt=fmt)
            
            print(f"转换成功！文件已保存为: {txt_file_path}")
            print(f"最终形状: {data_2d.shape}")
            
        elif data.ndim == 2:
            # 如果是2D数组，直接保存
            np.savetxt(txt_file_path, data, delimiter=delimiter, fmt=fmt)
            print(f"转换成功！文件已保存为: {txt_file_path}")
            print(f"数据形状: {data.shape}")
        
        else:
            # 1D数组或其他情况
            np.savetxt(txt_file_path, data, delimiter=delimiter, fmt=fmt)
            print(f"转换成功！文件已保存为: {txt_file_path}")
            print(f"数据形状: {data.shape}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {npy_file_path}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

# 使用示例
if __name__ == "__main__":
    npy_file = "radio_case7.npy"
    txt_file = "data_2d.txt"
    npy3d_to_txt_2d(npy_file, txt_file)