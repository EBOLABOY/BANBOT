"""
修复导入路径脚本 - 将相对导入改为绝对导入
"""

import os
import re
import glob

def fix_imports(directory):
    """
    修复指定目录下所有.py文件的导入路径
    
    参数:
        directory: 需要修复的目录
    """
    # 获取目录下所有.py文件
    py_files = glob.glob(os.path.join(directory, "*.py"))
    
    # 替换计数器
    total_replaced = 0
    files_modified = 0
    
    for file_path in py_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式替换导入语句
        # 将 "from utils." 替换为 "from src.utils."
        # 将 "from models." 替换为 "from src.models."
        # 将 "from features." 替换为 "from src.features."
        # 将 "from data." 替换为 "from src.data."
        
        new_content, count1 = re.subn(r'from utils\.', r'from src.utils.', content)
        new_content, count2 = re.subn(r'from models\.', r'from src.models.', new_content)
        new_content, count3 = re.subn(r'from features\.', r'from src.features.', new_content)
        new_content, count4 = re.subn(r'from data\.', r'from src.data.', new_content)
        
        # 计算替换总数
        count = count1 + count2 + count3 + count4
        
        # 如果有替换，更新文件
        if count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"已修复 {file_path}，替换了 {count} 处导入")
            total_replaced += count
            files_modified += 1
    
    print(f"\n修复完成! 共修改了 {files_modified} 个文件，替换了 {total_replaced} 处导入。")

if __name__ == "__main__":
    # 修复models目录
    models_dir = os.path.join("src", "models")
    print(f"开始修复 {models_dir} 目录下的文件...")
    fix_imports(models_dir) 