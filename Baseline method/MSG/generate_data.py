import os
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 如首次使用 NLTK，请取消注释以下行下载所需资源
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# 1. 加载 GloVe 词表
def load_glove_vocab(glove_path):
    vocab = set()
    with open(glove_path, 'r', encoding='latin-1', errors='ignore') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    return vocab

# 2. 文本预处理函数
def preprocess_text(text, glove_vocab):
    # 转小写
    text = text.lower()

    # 替换常见缩写
    abbreviations = {
        "can't": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)

    # 去除特殊符号和数字，仅保留字母和空格
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 分词
    words = nltk.word_tokenize(text)

    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # 仅保留 GloVe 中的词
    words = [w for w in words if w in glove_vocab]

    return words

# 新建文件夹（如果不存在）
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
glove_path = '.vector_cache/glove.6B.200d.txt'
print("加载 GloVe 词汇表...")
glove_vocab = load_glove_vocab(glove_path)
print(f"共加载 GloVe 词数：{len(glove_vocab)}")

# 读取原始数据
with open("original data/api_id_mapping.json", "r", encoding="utf-8") as f:
    apis = json.load(f)

with open("original data/shuffle_mashup_details.json", "r", encoding="utf-8") as f:
    mashups = json.load(f)

# 初始化输出
api_category = []
api_description = []
api_name = []

category_list_set = set()
mashup_category = []
mashup_description = []
mashup_name = []
mashup_used_api = []
used_api_set = set()

# 处理 API 数据
for api in apis:
    # API 名称
    api_name.append(str(api["id"]))

    # API 描述
    api_des = api["details"].get("description", "")
    if api_des == "":
        api_des = 'no description available'
        api_des = preprocess_text(api_des, glove_vocab)
    else:
        api_des = preprocess_text(api_des, glove_vocab)
    api_description.append(api_des)

    # API 类别
    tags = api["details"].get("tags", "")
    # tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    if not tags.strip():
        tag_list = ["NULL"]
    else:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    api_category.append(tag_list)

# 处理 Mashup 数据
# 处理 Mashup 数据
for mashup in mashups:
    mashup_name.append(str(mashup["id"]))

    # 类别处理成列表
    cats = mashup.get("categories", "")
    if not cats.strip():
        cat_list = ["NULL"]
    else:
        cat_list = [cat.strip() for cat in cats.split(",") if cat.strip()]
    mashup_category.append(cat_list)

    for c in cat_list:
        category_list_set.add(c)

    # 描述
    mashup_des = mashup.get("description", "")
    if mashup_des == "":
        mashup_des = 'no description available'
        mashup_des = preprocess_text(mashup_des, glove_vocab)
    else:
        mashup_des = preprocess_text(mashup_des, glove_vocab)
    mashup_description.append(mashup_des)

    # 使用的 API
    api_ids = mashup.get("api_info", [])
    mashup_used_api.append([str(aid) for aid in api_ids])
    used_api_set.update(api_ids)


# 最后生成 category_list
category_list = sorted(list(category_list_set))
used_api_list = sorted([str(i) for i in used_api_set])


# 写入输出文件
def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


write_json("data/api_category.json", api_category)
write_json("data/api_description.json", api_description)
write_json("data/api_name.json", api_name)
write_json("data/category_list.json", category_list)
write_json("data/mashup_category.json", mashup_category)
write_json("data/mashup_description.json", mashup_description)
write_json("data/mashup_name.json", mashup_name)
write_json("data/mashup_used_api.json", mashup_used_api)
write_json("data/used_api_list.json", used_api_list)
