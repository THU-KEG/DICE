def multiply_by_100(row):
    # 解析输入的行并将每个值乘以100
    values = [float(x.strip()) * 100 for x in row.split('&')[1:]]
    # 格式化输出新的行
    new_row = " & ".join([f"{val:.1f}" for val in values])
    return new_row

def main():
    # 输入数据行
    row = input("请输入一行数据：").strip()

    # 将每个值乘以100并输出新的行
    new_row = multiply_by_100(row)
    print(new_row)

if __name__ == "__main__":
    main()
