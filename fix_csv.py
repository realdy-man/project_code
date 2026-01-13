import csv

# 原始文件路径
input_file = 'F:\AAA_JIQIXUEXI\project_code\output.csv'
# 临时文件路径
output_file = 'F:\AAA_JIQIXUEXI\project_code\output_fixed.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    
    for i, line in enumerate(infile, 1):
        # 去除行末尾的换行符
        line = line.rstrip('\n')
        
        # 处理第129行的特殊情况
        if i == 129:
            # 分割成3个部分，将前两部分合并为一个字段
            parts = line.split(',')
            if len(parts) >= 3:
                fixed_line = [','.join(parts[:-1]), parts[-1]]
                writer.writerow(fixed_line)
                print(f"已修复第{i}行: {fixed_line}")
                continue
        
        # 正常处理其他行
        parts = line.split(',')
        if len(parts) == 2:
            writer.writerow(parts)
        else:
            print(f"第{i}行字段数量异常: {len(parts)}个字段")
            writer.writerow(parts)

print(f"修复完成，新文件已保存为: {output_file}")