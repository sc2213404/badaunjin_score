import os


def save_directory_structure(directory, output_file):
    """
    递归遍历目录，并将目录结构保存到指定的输出文件中
    :param directory: 目标目录路径
    :param output_file: 输出文件路径
    """

    def traverse_dir(current_dir, indent_level=0):
        """
        递归遍历并打印目录和文件结构
        :param current_dir: 当前处理的目录
        :param indent_level: 缩进级别，用于显示文件结构层次
        """
        try:
            items = os.listdir(current_dir)  # 列出当前目录的所有文件和子目录
        except PermissionError:
            # 如果没有权限访问某些文件夹，跳过
            return

        items.sort()  # 排序以使输出结构更清晰
        for item in items:
            item_path = os.path.join(current_dir, item)
            indent = " " * 4 * indent_level  # 每一层次缩进4个空格
            if os.path.isdir(item_path):
                # 如果是目录，打印目录名并递归进入子目录
                output_file.write(f"{indent}[DIR] {item}/n")
                traverse_dir(item_path, indent_level + 1)
            else:
                # 如果是文件，打印文件名
                output_file.write(f"{indent}[FILE] {item}/n")

    # 打开输出文件，并写入目录结构
    with open('directory_structure.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(f"Directory structure of {directory}:/n")
        traverse_dir(directory)


# 使用该函数
#directory_to_check = "F:/软件工程/AI技术赛道（更新）/action"  # 替换为你要遍历的目录路径
directory_to_check = "F:/软件工程/AI技术赛道（更新）/add/mark_all"  # 替换为你要遍历的目录路径
save_directory_structure(directory_to_check, 'directory_structure.txt')

print(f"目录结构已保存到 directory_structure.txt")
