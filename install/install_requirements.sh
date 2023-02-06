#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install torch==1.13
pip install pandas
pip install scikit-learn
pip install -r requirements.txt

# maceb install
apt-get install -y build-essential openjdk-8-jdk python3-dev curl git automake
pip install konlpy "tweepy<4.0.0"
/bin/bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
