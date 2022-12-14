python ../../learn/training.py \
../../mimicdata/mimic3/train_full.csv \
../../mimicdata/mimic3/vocab.csv \
full \
conv_attn \
200 \
--filter-size 10 \
--num-filter-maps 50 \
--dropout 0.2 \
--patience 10 \
--lr 0.0001 \
--public-model \
--test-model model.pth \
--vocab-dicts-file "../../mimic3_vocab_dicts.json"