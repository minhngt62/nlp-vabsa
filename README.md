# Vietnamese Aspect-based Sentiment Analysis
Understanding the comments, feedback, and opinions of users, clients, and customers has always been crucial in all fields of life. Unfortunately, manually handling the vast amount of textual content is now unachievable. This problem leads to the emergence of the research field sentiment analysis (SA), which helps automatically analyze the opinions of users. With an increase in the complexity of products, companies also need to recognize comments on smaller aspects of the products; therefore, the SA system has to analyze more detailed parts of users' feedback to recognize  more fine-grained aspect-level opinions and sentiments. This field of study is named as Aspect-Based Sentiment Analysis (ABSA), which includes many a variety of helping tasks.

**For more details, please read the report provided in folder `./docs`.**

<p align="center">
  <img src="https://github.com/minhngt62/nlp-vabsa/blob/main/assests/2-Figure1-1.png" />
</p>

## Natural Language Processing - DSAI K65: Group 04
1. Nguyễn Tống Minh (Email: minh.nt204885@sis.hust.edu.vn)
2. Nguyễn Xuân Thái Hòa
3. Ngô Thị Thu Huyền
4. Nguyễn Nhật Quang

---
To reproduce the experiments in notebooks, we notice that the file paths should be added with the corresponding parent directories. Also, when re-running the code, providing a cell `%cd` to `vabsa` will eliminate the issues of importing source code. For more details, please follow the section below.

## Guidelines 

1. Install required libraries:
   ```
   !pip install -r requirements
   ```
   Some methods will require more, specified libraries that are `requirements.txt` located in `vabsa/<method_name>`. Please install them as well if you need to re-run the programs.
2. Download the checkpoint directories for the methods from: [CHECKPOINTS](https://husteduvn-my.sharepoint.com/personal/minh_nt204885_sis_hust_edu_vn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fminh%5Fnt204885%5Fsis%5Fhust%5Fedu%5Fvn%2FDocuments%2F2022%2E2%5FNLP%5FCheckpoints&view=0). Put each checkpoint directory under `./checkpoints` for usage. For GatedCNN, please download [this](https://public.vinai.io/word2vec_vi_words_300dims.zip) and put each to the corresponding folder. 
4. To run the demo: You may want to change the input review, please do that at the module itself
   ```
   !python vabsa\roberta\vabsa\roberta\infer_1_sentence.py
   ```

