#!/bin/bash
pip install -r requirements.txt

# maceb install
apt-get install -y build-essential openjdk-8-jdk python3-dev curl git automake
pip install konlpy "tweepy<4.0.0"
/bin/bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# py-hanspell install
git clone https://github.com/ssut/py-hanspell.git
cd py-hanspell
python setup.py install
cd ..