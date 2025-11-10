llava = []

# 使用 'with' 语句打开文件，这样在读取完成后会自动关闭文件
with open('/home/bingxing2/ailab/caijinyu/LISA/llava.txt', 'r', encoding='utf-8') as file:
    # 使用 for 循环逐行读取文件
    for line in file:
        # 使用 strip() 方法去除行末的换行符和可能存在的空白字符，然后添加到列表中
        llava.append(line.strip().replace("base_model.model.", ""))


lisa=[]
with open('/home/bingxing2/ailab/caijinyu/LISA/lisa.txt', 'r', encoding='utf-8') as file:
    for line in file:
        lisa.append(line.strip())


# 将列表转换为集合
set_llava = set(llava)
set_lisa = set(lisa)

# 找出仅在 llava 中存在的元素
only_in_llava = set_llava - set_lisa

# 找出仅在 lisa 中存在的元素
only_in_lisa = set_lisa - set_llava

print("仅在 llava 中存在的内容:", only_in_llava) # 全是lora参数
print(sum(['lora' in item for item in only_in_llava]),len(only_in_llava)) # 100% lora参数
print("仅在 lisa 中存在的内容:", only_in_lisa)