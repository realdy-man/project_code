import csv

# 原始文件路径
input_file = 'F:\AAA_JIQIXUEXI\project_code\output.csv'
# 修复后文件路径
output_file = 'F:\AAA_JIQIXUEXI\project_code\output_fixed.csv'

fixed_count = 0

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    
    for i, line in enumerate(infile, 1):
        # 去除行末尾的换行符
        line = line.rstrip('\n')
        
        # 分割行
        parts = line.split(',')
        
        # 处理字段数量异常的情况
        if len(parts) == 2:
            # 正常行，直接写入
            writer.writerow(parts)
        elif len(parts) > 2:
            # 字段数量过多，将前n-1个字段合并为一个字段，最后一个字段保留
            fixed_line = [','.join(parts[:-1]), parts[-1]]
            writer.writerow(fixed_line)
            fixed_count += 1
        elif len(parts) == 1:
            # 只有一个字段，可能缺少分类
            # 这里简单处理，给一个默认分类"其他"
            writer.writerow([parts[0], "其他"])
            fixed_count += 1
        else:
            # 空行，跳过
            continue

print(f"修复完成！总共有 {fixed_count} 行被修复")
print(f"新文件已保存为: {output_file}")
print("修复规则:")
print("1. 字段数量>2: 合并前n-1个字段为文本内容，保留最后一个字段为分类")
print("2. 字段数量=1: 保留文本内容，添加默认分类'其他'")