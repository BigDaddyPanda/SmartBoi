echo "del old model"
rm models/model.h5
rm models/model.json
rm models/vocabulary.p

python sentiment.py

echo "copy Model to Web App"
cp models/model.h5 /Users/alex/Documents/py/myflaskapp/models/test1_model.h5
cp models/model.json /Users/alex/Documents/py/myflaskapp/models/test1_model.json
cp models/vocabulary.p /Users/alex/Documents/py/myflaskapp/models/test1_vocabulary.p
cp acc.png /Users/alex/Documents/py/myflaskapp/models/test1_acc.png
cp loss.png /Users/alex/Documents/py/myflaskapp/models/test1_loss.png
cp model.png /Users/alex/Documents/py/myflaskapp/models/test1_model.png

