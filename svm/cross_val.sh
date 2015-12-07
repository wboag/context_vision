rm -f models/*.model
rm -f output/*.pred


for i in {1..5} ; do
    python train.py --pred "../data/cv/$i-train.txt" --model "models/$i-fold.model"

    echo ""

    python predict.py --pred "../data/cv/$i-test.txt" --model "models/$i-fold.model" --out "output/$i-fold.pred"

    echo ""

    python evaluate.py --pred "output/$i-fold.pred" > "../data/results/$i-fold.results"
done


echo ""
echo "results stored in data/results/"
