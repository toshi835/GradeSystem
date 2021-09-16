# GradeSystem

# データについて
エッセイ（学習者が書いた作文）は以下にある

- A1, A2, B1の3段階

`/home/lr/kawamoto/m1/GradeSystemForEssay/cefrj/`

テキストブックは以下にある

- A1, A2, B1, B2の4段階(C1,C2もあるが数が少ないため利用していない)

`/home/lr/kawamoto/m1/GradeSystemForEssay/textbook/`

# プログラムの動かし方

0. (仮想環境かなにかでpip install -r requirements.txt)


1. 前処理  `python run.py -mode prepro -data essay`
   

2. 学習時   `python run.py -mode train -data essay -clf lr -out ../model/essay/test.pkl`


3. テスト時  `python run.py -mode test -data essay -clf lr -out ../model/essay/test.pkl`

GECの結果を使用するときは -gec [statgec or nngec]

MLPで学習、テストを行いたいときは -clf nn

textbookのときは -data textbook

です。今後詳しく変更していきます。