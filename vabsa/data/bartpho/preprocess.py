{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import os\nimport re\nimport json\nimport string\nimport emoji\nimport pandas as pd\n\ndef normalize_money(sent):\n    return re.sub(r'\\d+((,|.)\\d)?[k|m|b](/)?', 'giá ', sent)\n\ndef normalize_time(sent):\n    return re.sub(r'\\d+(\\s)?(h|giờ)(\\s)?(\\d+)?', 'giờ ', sent)\n\ndef normalize_hastag(sent):\n    return re.sub(r'#+\\w+', 'tag', sent)\n\ndef normalize_HTML(text):\n    return re.sub(r'<[^>]*>', '', text)\n\ndef normalize_website(sent):\n    result = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'website', sent)\n    return re.sub(r'\\w+(\\.(com|vn|me))+((\\/+([\\.\\w\\_\\-]+)?)+)?', 'website', result)\n\ndef normalize_emoji(sent):\n    return emoji.demojize(sent)\n\ndef normalize_elongate(sent):\n    patern = r'(.)\\1{1,}'\n    result = sent\n    while(re.search(patern, result) != None):\n        repeat_char = re.search(patern, result)\n        result = result.replace(repeat_char[0], repeat_char[1])\n    return result\n\ndef remove_number(sent):\n    return re.sub(r'[0-9]+', '', sent)\n\ndef remove_punct(text):\n    '''\n    This funtion replaces punctuations in texts for easier handling\n    '''\n    text = text.replace(\";\", \",\").replace(\"“\", \" \").replace(\"”\", \" \")\n    text = \"\".join(\n        [\n            c\n            if c.isalpha() or c.isdigit() or c in [\",\",\".\"]\n            else \" \"\n            for c in text\n        ]\n    )\n    text = \" \".join(text.split())\n    return text\n\ndef normalize_acronyms(sent):\n    text = sent\n    replace_list = {\n        'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ', 'okey': ' ok ', 'ôkê': ' ok ',\n        'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',' tks ': u' cám ơn ',\n        'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',\n        '^_^': 'tích cực', ':)': 'tích cực', ':(': 'tiêu cực', '❤️': 'tích cực', '👍': 'tích cực',\n        '🎉': 'tích cực', '😀': 'tích cực', '😍': 'tích cực', '😂': 'tích cực', '🤗': 'tích cực',\n        '😙': 'tích cực', '🙂': 'tích cực', '😔': 'tiêu cực', '😓': 'tiêu cực', '⭐': 'star', \n        '*': 'star', '🌟': 'star','kg ': u' không ','not': u' không ', u' kg ': u' không ',\n        '\"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',\n        u' kô ': u' không ', '\"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ',\n        'khong': u' không ', u' hok ': u' không ','he he': ' tích cực ','hehe': ' tích cực ',\n        'hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',' lol ': ' tiêu cực ',\n        ' cc ': ' tiêu cực ','cute': u' dễ thương ','huhu': ' tiêu cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',\n        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',\n        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích cực ', 'store': u' cửa hàng ',\n        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',\n        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',\n        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',\n        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',\n        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',\n        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',\n        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',\n        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',\n        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',\n        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích cực ', \" view \": \" tầm nhìn \", \n        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback '\n    }\n    \n    for k, v in replace_list.items():\n        text = text.replace(k, v)\n    return text\n\ndef normalize(sent):\n    #result = standardize_sentence_typing(sent)\n    result = normalize_hastag(sent)\n    result = normalize_website(result)\n    result = normalize_HTML(result)\n    result = normalize_acronyms(result)\n    result = normalize_emoji(result)\n    result = result.lower()\n    result = normalize_time(result)\n    result = normalize_money(result)\n    result = normalize_elongate(result)\n    result = normalize_acronyms(result)\n    result = remove_number(result)\n    result = remove_punct(result)\n    result = result.replace(\",.\", \".\")\n    result = re.sub(r'\\s+', ' ', result).strip() # Remove extra whitespace\n    return result\n\ndef tokenize(sent, f):\n    return f(sent)","metadata":{"_uuid":"a8e5152f-4ed8-42ed-8c4f-57ce5ee92b71","_cell_guid":"7900b856-38b4-4a7c-bcd4-9ad1fae8020e","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}