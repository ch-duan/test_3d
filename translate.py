from googletrans import Translator, LANGUAGES
import os

# 初始化谷歌翻译器
translator = Translator()

def translate_and_rename(directory):
    for filename in os.listdir(directory):
        if not filename.startswith('.'):  # 忽略隐藏文件
            # 翻译文件名（不包括文件扩展名）
            name, extension = os.path.splitext(filename)
            translated_name = translator.translate(name, dest='en').text

            # 创建新的文件名并重命名
            new_filename = f"{translated_name}{extension}"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed '{filename}' to '{new_filename}'")

# 调用函数，传递你的目录
translate_and_rename("img2")