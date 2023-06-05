for LR in 0.1
do
    for EPOCHS in 100
    do

        python main.py \
        --model_num 3 \
        --total_epoch ${EPOCHS} \
        --lr ${LR} \
        --batches_train 128 \
        --batches_eval 100

        python ensemble.py \
        --model_num 3 \
        --total_epoch ${EPOCHS} \
        --lr ${LR} \
        --batches_train 128 \
        --batches_eval 100

    done
done
