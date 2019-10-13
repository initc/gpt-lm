mkdir -p tmp_data/data-bin
python async_preprocessing.py --data tmp_data/train.txt --out tmp_data/data-bin --prefix train --workers 6
python async_preprocessing.py --data tmp_data/valid.txt --out tmp_data/data-bin --prefix valid --workers 6