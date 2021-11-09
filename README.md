# GradeSystem

## Data
- essay
    - A1, A2, B1の3段階
    - データのパス
      `/home/lr/kawamoto/m1/GradeSystemForEssay/essay/`
- textbook
    - A1, A2, B1, B2, C(C1とC2)の5段階
    - データのパス  
    `/home/lr/kawamoto/m1/GradeSystemForEssay/textbook/`

## How to use

### Preprocessing 
```
python run.py -mode prepro -data essay
```

### Training
```
python run.py -mode train -data essay -clf lr -model ../models/essay/test.pkl
or 
python run.py -mode train -data essay -clf bert -epoch 1 -gpus 1 -model ../models/essay/debert
```

### Test
```
python run.py -mode test -data essay -clf lr -model ../models/essay/test.pkl
```

## Option

- `-data {essay, textbook}`  
どのデータを使用するか指定
- `-gec {stat, nn, correct}`  
データがEssayのとき、GECのどの結果を使用するかを指定  
`stat`は統計的GEC, `nn`はNN GEC, `correct`は正解文  
指定しない場合GECの結果は使用しない
- `-clf {lr nn}`  
分類器に利用するモデルを指定  
`lr`はロジスティック回帰、`nn`はMLP
- `-gpus n`  
nは使用したいGPUの番号を入れる
- `-epoch 3000`  
clfにnnを選択した場合の学習するepochの回数を指定
- `-model MODEL_PATH`  
train時に保存、test時に使用するモデルのパスを指定