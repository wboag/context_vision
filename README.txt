
svm/

    ** most of the scripts for our Context-SVM and BBox-SVM

    ** to run 5-fold cross-validation for Context-SVM

        cd svm
        bash cross_val.sh  # If you only run one thing, it should be this




    ** to run Context-SVM

        python train.py --pred data/cv/1-train.txt --model svm/models/1-fold.model

        python predict.py --pred data/cv/1-test.txt --model svm/models/1-fold.model --out svm/output/1-fold.pred

        python evaluate.py --pred svm/output/1-fold.pred > data/results/1-fold.results





data/

    ImageNet data - /data1/wboag/cv/project/final/data/imagenet/ILSVRC2013_DET_bbox_val

    Caffe predictions - /data1/wboag/cv/project/final/data/pred/log/allImage_Predictions_newAnnotations_noDupla.log




collection/

    script to find "tricky" images where Context-SVM corrects Caffe - /data1/wboag/cv/project/final/collection/findImprovement.py

    runs pretrained Caffe model - /data1/wboag/cv/project/final/collection/myDetect.py



