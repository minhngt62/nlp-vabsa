from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
import warnings
import string
warnings.filterwarnings("ignore")
import re

device = 'cuda'

model = RobertaForSequenceClassification.from_pretrained("checkpoints\\roberta", num_labels=4).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def normalize_money(sent):
    return re.sub(r'[0-9]+[.,0-9][k-m-b]', 'giá', sent)

def normalize_hastag(sent):
    return re.sub(r'#+\w+', 'tag', sent)

def normalize_website(sent):
    result = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'website', sent)
    return re.sub(r'\w+(\.(com|vn|me))+((\/+([\.\w\_\-]+)?)+)?', 'website', result)

def nomalize_emoji(sent):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', sent)

def normalize_elongate(sent):
    patern = r'(.)\1{1,}'
    result = sent
    while(re.search(patern, result) != None):
        repeat_char = re.search(patern, result)
        result = result.replace(repeat_char[0], repeat_char[1])
    return result

def remove_number(sent):
    return re.sub(r'[0-9]+', '', sent)

def normalize_acronyms(sent):
    text = sent
    replace_list = {
        'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' tích cực ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' tích cực ','hehe': ' tích cực ','hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',
        ' lol ': ' tiêu cực ',' cc ': ' tiêu cực ','cute': u' dễ thương ','huhu': ' tiêu cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích cực ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích cực ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback '
    }
    for k, v in replace_list.items():
        text = text.replace(k, v)
    return text


def normalize(sent):
    result = normalize_money(sent)
    result = normalize_hastag(result)
    result = normalize_website(result)
    result = nomalize_emoji(result)
    result = normalize_elongate(result)
    result = normalize_acronyms(result)
    result = remove_number(result)
    result = result.lower()
    return result.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()
def preprocess_data(sent):
    sent = normalize(sent)   
    return sent 

predictions = []

aspect_dict = {"AMBIENCE#GENERAL" : "nhận xét chung ngoại cảnh","DRINKS#PRICES": "giá đồ uống", "DRINKS#QUALITY": "chất lượng đồ uống", 
                   "DRINKS#STYLE&OPTIONS": "các lựa chọn và phong cách đồ uống" , "FOOD#PRICES": "giá đồ ăn","FOOD#QUALITY": "chất lượng đồ ăn","FOOD#STYLE&OPTIONS": "các lựa chọn và phong cách đồ ăn","LOCATION#GENERAL": "nhận xét chung vị trí",
                   "RESTAURANT#GENERAL": "nhận xét chung về nhà hàng", "RESTAURANT#MISCELLANEOUS": "khía cạnh khác của nhà hàng","RESTAURANT#PRICES": "giá nhà hàng","SERVICE#GENERAL": "nhận xét chung dịch vụ"}

test_sentence = "đây là trong những quán mà mình thích vì vị trà đậm và thơm cũng như mùi vị đặc trưng hơn hẳn những quán khác nè trà sữa trân châu sợi giá trà sữa pha khá ngon vị trà chát và mùi hương khá rõ không quá ngọt rất đúng với gu mình trà đào giá vị trà đào ở đây cũng đặc biệt hơn hẳn những quán khác không phải chua ngọt như thưởng thấy mà có mùi trà rất ngon cà phê đá xay giá món đá xay ở đây uống cũng ngon không kém trà nè mùi vị thơm hương cà phê vị đắng kết hợp hoàn hảo với độ béo ngọt của whiping cream không quá đắng cũng không quá ngọt hay lạt lẽo mà dịu nhẹ thơm và dễ uống lắm trà vải thiết quan âm giá trà vải có mùi vị rất thơm ngon mùi vải mà vẫn nghe rõ vị trà có chút vị chát nhẹ mùi trà thơm rất thích không phải chỉ toàn vị syrup vải ngọt gắt như nhiều chỗ khác do trà ở đây pha khá đậm nên bạn nào uống mà đang đói sẽ dễ say nha hoặc ban đêm có thể khó ngủ à cảnh báo trước trà thiết quan âm late giá ly này thì vị trà rất đậm nên cảm giác hơi nhạt và chát phần kem sữa bên trên béo béo uống chung thì vừa trà sữa giá ly này chụp ké chứ không có thử v"
test_sentence = preprocess_data(test_sentence)

polarity_dict = {0: "none", 1: "positive", 2: "negative", 3: "neutral"}
for i in tqdm(aspect_dict.keys()):

    test_encodings = tokenizer(test_sentence, str(aspect_dict[i]),truncation=True, padding=True, return_tensors='pt').input_ids.to(device)
    logits = model(test_encodings).logits

    prediction = np.argmax(logits.cpu().detach().numpy(), axis=-1)[0]
    if int(prediction) != 0:
       predictions.append(str(i) + " - " + str(polarity_dict[int(prediction)]))
    else:
        continue

print("Testing Sentence: ", test_sentence)
print("Prediction: ", ', '.join(str(x) for x in predictions))