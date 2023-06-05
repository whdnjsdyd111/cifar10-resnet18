for LR in 0.01 0.005
do
    for EPOCHS in 50 60 70
    do

        #python main.py \
        #--model_num 3 \
        #--total_epoch ${EPOCHS} \
        #--lr ${LR} \
        #--batches_train 128 \
        #--batches_eval 100

        python ensemble.py \
        --model_num 3 \
        --total_epoch ${EPOCHS} \
        --lr ${LR} \
        --batches_train 128 \
        --batches_eval 100

    done
done
