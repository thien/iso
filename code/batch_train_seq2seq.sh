python3 train.py --ep 10 -ul f -r 16 --dataset subtitles --model seq2seq
python3 train.py --ep 10 -ul f -r 16 --dataset amazon --model seq2seq
python3 train.py --ep 10 -ul f -r 16 --dataset penn --model seq2seq
python3 train.py --ep 10 -r 16 --dataset amazon --model bowman
python3 train.py --ep 10 -r 16 --dataset subtitles --model bowman